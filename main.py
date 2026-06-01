import os 

from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_pinecone import PineconeVectorStore

load_dotenv()

MODEL = "qwen3-coder:30b"

embeddings = OllamaEmbeddings(model = "nomic-embed-text:latest")
llm = ChatOllama(model = MODEL)

# Get the vector DataBase from pinecone
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriver = vectorstore.as_retriever(search_kwargs = {"k":3})

prompt_template = ChatPromptTemplate.from_template("""
Answer the question only on the basis of the context :
{context}

Question : {query}

Provide a detailed answer :
""")



def format_doc(docs):
    """Format into a single line string"""
    return "\n\n".join(doc.page_content for doc in docs)



def retrival_chain (query:str):
    """Simple Retrival Chain"""

    # Step 1 get the relavent documents and format it
    docs = retriver.invoke(query)
    context = format_doc(docs)

    # Step 2 Create the prompt from our format to give to the llm cause we have gotten the most relavent context
    message = prompt_template.format_messages(context = context , query = query)

    # Step 3 invoke and use the llm
    response = llm.invoke(message)

    print(response.content)

    return response.content



if __name__ == "__main__":
    print("Retriving ...")

    # QUERY
    query = "What strategies can a person use to hide their true goals from opponents?"

    retrival_chain(query)


