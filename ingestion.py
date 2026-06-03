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
tavily_map = TavilyMap(max_depth = 2, max_breadth = 20, max_pages = 1000)


