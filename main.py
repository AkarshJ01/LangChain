import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from langfuse.langchain import CallbackHandler

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage


from langchain_tavily import TavilySearch

# @tool
# def search(query: str) -> str:
#     """
#     Tool that searchs the internet
#     Args :
#         query : The query to search for
#     Returns :
#         The search result
#     """
#     print (f"Seach for {query}")
#     return "Tokyo weather is great"

MODEL = "llama3.1:8b"

llm = ChatOllama(temperature=0, model= MODEL)
tools = [TavilySearch()]
agent = create_agent(model = llm , tools = tools)

 
def main():
    print(f"Hello from {MODEL}! \n")

    langfuse_handler = CallbackHandler()

    result = agent.invoke({"messages" : [HumanMessage(content=f"Is the ai model {MODEL} in any way related to china?")]},
                          config={"callbacks": [langfuse_handler]})
    print(f"{result["messages"][-1].content} \n")

 
if __name__ == "__main__":
    main()