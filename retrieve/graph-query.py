import os
import asyncio
import cassio
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.graph_vectorstores import CassandraGraphVectorStore
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_ENDPOINT = os.environ["ASTRA_DB_ENDPOINT"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ASTRA_DB_ID = os.environ["ASTRA_DB_ID"]
GRAPH_NODE_TABLE = os.environ["GRAPH_NODE_TABLE"]
KEYSPACE="demo"

SYSTEM_PROMPT = """You are a helpful assistant that uses the provided context to answer the question.
If you cannot find the answer in the context, say that you don't know."""

# Initialize embeddings and LLM using OpenAI
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

# Initialize Astra connection using Cassio
cassio.init(database_id=ASTRA_DB_ID, token=ASTRA_DB_APPLICATION_TOKEN)
store = CassandraGraphVectorStore(embeddings, table_name=GRAPH_NODE_TABLE,keyspace=KEYSPACE)

def format_docs(docs):

    return "\n\n".join(doc.page_content for doc in docs)

def get_traversal_result(question, k=10, depth=3):

    # Create a graph traversal retriever
    traversal_retriever = store.as_retriever(
        search_type="mmr_traversal",
        search_kwargs={
            "k": k,
            "depth": depth,
            "lambda_mult": 0.25,
            "fetch": 50
        }
    )

    # Retrieve documents
    retrieved_docs = traversal_retriever.get_relevant_documents(question)
    context = format_docs(retrieved_docs)
    print(context)

    # Construct a prompt for the LLM
    # You can structure the prompt however you like, here's a simple inline format:
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Use llm.predict() which is synchronous
    answer = llm.predict(prompt)
    usage_metadata = {}  # Replace with actual usage metadata if available

    return answer, usage_metadata, retrieved_docs

# Example usage:
if __name__ == "__main__":
    question = "what is IMPS and how much time does it take for the money to be credited"
    answer, usage_metadata, docs = get_traversal_result(question, k=3, depth=2)
    print("Answer:", answer)
    # print("Usage Metadata:", usage_metadata)
    # print("Retrieved Documents:")
    # for doc in docs:
    #     print(f"- {doc.metadata.get('source')}: {doc.page_content[:200]}...")
