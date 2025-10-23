from itertools import chain

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch

from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

load_dotenv()

tools = [TavilySearch()]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

#react_prompt = hub.pull("hwchase17/react")

output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

format_instructions = output_parser.get_format_instructions()
react_prompt_with_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
).partial(format_instructions=format_instructions)


agent = create_react_agent(
    llm=llm,
    prompt=react_prompt_with_instructions,
    tools=tools)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

extract_output = RunnableLambda(
    lambda x: x["output"] #a ideia aqui é extrair o campo 'output' do dicionário retornado pelo agente
)
parse_output = RunnableLambda(
    lambda x: output_parser.parse(x) #aqui temos o parser que converte o output em um objeto do tipo AgentResponse
)

chain = agent_executor | extract_output | parse_output # a ideia aqui é retirar o output da resposta "final" do agente por meio do extract_output e depois parsear esse output para o formato desejado com o parse_output


def main():

    try:
        result = chain.invoke(
            input={
                "input": "Search for 3 jobs postings for an AI Engineer using langchain in the bay area on linkedin and list their details."
            }
        )
        print(result)
    except Exception as e:
        import traceback

        print("Erro ao executar:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
