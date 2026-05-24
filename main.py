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

llm = ChatOllama(temperature=0, model="llama3.1:8b")
tools = [TavilySearch()]
agent = create_agent(model = llm , tools = tools)

 
def main():
    print("Hello from langchain!")

    langfuse_handler = CallbackHandler()

    result = agent.invoke({"messages" : [HumanMessage(content="What is the average package of computer science btech in nit warangal ")]},
                          config={"callbacks": [langfuse_handler]})
    print(result["messages"][-1].content)

 
if __name__ == "__main__":
    main()