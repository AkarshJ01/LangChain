
#test asdf a asdfas d
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

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    client_kwargs={"timeout": 60.0}
)
vectorstore = PineconeVectorStore(index_name= os.environ['INDEX_NAME'], embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_crawl = TavilyCrawl()
tavily_map = TavilyMap(max_depth = 5, max_breadth = 20, max_pages = 1000)


# vectorstore = PineconeVectorStore(
#     index_name="langchain-project-index", embedding=embeddings
# )

# Local Vector space
from langchain_chroma import Chroma

chroma = Chroma(persist_directory="chroma_db", embedding_function=embeddings)






async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )


    semaphore = asyncio.Semaphore(2)

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        async with semaphore:
                    try:
                        await vectorstore.aadd_documents(batch)
                        log_success(
                            f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
                        )
                    except Exception as e:
                        log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
                        return False
                    return True

    # Process batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )






async def main():
    """Main async function"""

    vectorstore.delete(delete_all=True)



    # Beautification
    log_header("Ingestion Pipline")
    log_info("Tavily crawling in 'https://docs.langchain.com/oss/python/langchain/overview'", Colors.PURPLE)

    # langfuse_handler = CallbackHandler()




    # Retriving data into all_docs using Tavily CRAWL

    # res = tavily_crawl.invoke({
    #     "url" : "https://docs.langchain.com/",
    #     # "url" : "https://www.nitw.ac.in/",
    #     # "url" : "https://github.com/AkarshJ01",
    #     "max_depth" : 1,
    #     "extract_depth" : "advanced",
    #     # "instructions": "LangChain"
    # },
    # # config = {"callbacks": [langfuse_handler]}
    # )


    res = tavily_crawl.invoke({
        "url" : "https://python.langchain.com/v0.2/docs/introduction/", # <-- Live, text-dense documentation
        "max_depth" : 1,
        "extract_depth" : "advanced",
    })

    all_docs = [Document(page_content=result.get('raw_content') or "", metadata = {"source": result.get('url')}) for result in res['results']]
    log_success(f"Successfully Crawlled {len(all_docs)} Documents !!!")






    # Text Splitting

    # Add this temporary log right before text_splitter:
    log_info(f"Sample content length from first doc: {len(all_docs[0].page_content)}")
    log_info(f"Sample text snippet: {all_docs[0].page_content[:200]}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)    
    splitted_docs = text_splitter.split_documents(all_docs)

    log_success(f"Number of chunks split : {len(splitted_docs)} from {len(all_docs)}")



    #Embedding and 
    await index_documents_async(splitted_docs, batch_size=32)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Pages crawled: {len(all_docs)}")
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")



if __name__ == "__main__":
    asyncio.run(main())
