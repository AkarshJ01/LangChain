import asyncio
import os
import ssl
from typing import Any, List, Dict

import certifi 
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

# Change this line:
from langfuse.langchain import CallbackHandler


from logger import (Colors, log_error, log_header, log_info , log_success , log_warning)

load_dotenv()


# To make sure there is no ssl error on using TAVILY
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(model = "nomic-embed-text:latest")

vectorstore = PineconeVectorStore(index_name= os.environ['INDEX_NAME'], embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_crawl = TavilyCrawl()
tavily_map = TavilyMap(max_depth = 5, max_breadth = 20, max_pages = 1000)

async def main():
    """Main async function"""
    # Beautification
    log_header("Ingestion Pipline")
    log_info("Tavily crawling in 'https://docs.langchain.com/oss/python/langchain/overview'", Colors.PURPLE)

    # langfuse_handler = CallbackHandler()

    res = tavily_crawl.invoke({
         # "url" : "https://www.nitw.ac.in/",
        # "url" : "https://github.com/AkarshJ01",
        "max_depth" : 1,
        "extract_depth" : "advanced",
        "instructions": "LangChain"
    },
    # config = {"callbacks": [langfuse_handler]}
    )

    all_docs = [Document(page_content=result.get('content') or "", metadata = {"source": result.get('url')}) for result in res['results']]
    log_success(f"Successfully Crawlled {len(all_docs)} Documents !!!")



if __name__ == "__main__":
    asyncio.run(main())
