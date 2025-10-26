from dotenv import load_dotenv
from langchain.tools import tool


load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text."""
    print(f"get_text_length enter with: {text}")
    text = text.strip("'\n").strip(
        '"'
        ) # striping away non alphabetic characters just in case

    return len(text)


if __name__ == "__main__":
    print("Hello, LangChain with React!")
    print(get_text_length(text="Dog"))
