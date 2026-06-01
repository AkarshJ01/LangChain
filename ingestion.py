import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings 
from langchain_pinecone import PineconeVectorStore


load_dotenv()


if __name__ == "__main__":
    print(f"Langchain model ")


    # Loading model
    loader = TextLoader("/Users/akarshjaiswal/Desktop/LangChain/mediumblog1.txt")
    document = loader.load()
    
    #Text splitting
    text_splitter = CharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    #Embedding
    embeddings = OllamaEmbeddings(model = "nomic-embed-text:latest")

    #Creating a vector space
    PineconeVectorStore.from_documents(texts, embeddings , index_name = os.environ['INDEX_NAME'])
