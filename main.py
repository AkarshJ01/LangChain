import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Hello from langchain!")

    information = """
    Steven Paul Jobs (February 24, 1955 – October 5, 2011) was an American businessman...
    """

    summary_template = """
    Given the information {information} create: 
    1. A short summary 
    2. 2 Interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOllama(temperature=0, model="gemma4:e4b")
    chain = summary_prompt_template | llm

    langfuse_handler = CallbackHandler()

    response = chain.invoke(
        input={"information": information},
        config={"callbacks": [langfuse_handler]}
    )
    print(response.content)

if __name__ == "__main__":
    main()