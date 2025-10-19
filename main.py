from itertools import chain
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_tavily import TavilySearch
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

tools=[TavilySearch()]

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

react_prompt=hub.pull("hwchase17/react")

agent= create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

agent_executor= AgentExecutor(agent=agent, tools=tools, verbose=True,
    handle_parsing_errors=True)

chain=agent_executor

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
    
if __name__== "__main__":
    main()
