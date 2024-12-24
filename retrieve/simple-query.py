import os
from dotenv import load_dotenv
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_ENDPOINT = os.environ["ASTRA_DB_ENDPOINT"]
ASTRA_DB_KEYSPACE = "demo"
ASTRA_DB_COLLECTION = "simple_demo"  # same as in pdf_loader.py

import os
from dotenv import load_dotenv
from astrapy import DataAPIClient
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_ENDPOINT = os.environ["ASTRA_DB_ENDPOINT"]
ASTRA_DB_KEYSPACE = "demo"
ASTRA_DB_COLLECTION = "simple_demo" # Ensure this matches what you used during indexing

# Initialize OpenAI embeddings and model
embedding = OpenAIEmbeddings()
llm = OpenAI()

# Connect to Astra DB collection
client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
database = client.get_database(ASTRA_DB_ENDPOINT)
collection = database.get_collection(ASTRA_DB_COLLECTION)

# Define the prompt template
ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

context: {context}
Question: "{question}"
Answer:"""
)


def retrieve_documents(query: str, top_k: int = 4):
    # Embed the query
    query_vector = embedding.embed_query(query)
    # Perform vector search using Astra DB's vector indexing
    # Here we simply sort by the vector field. Make sure your collection
    # is properly configured for vector search.
    cursor = collection.find(
        sort={"$vector": query_vector},
        projection={"content": True},
    )
    documents = list(cursor)
    # Take top_k documents
    return [doc['content'] for doc in documents[:top_k] if 'content' in doc]


def answer_question(question: str):
    context_documents = retrieve_documents(question, top_k=4)
    if not context_documents:
        return "I couldn't find any relevant content."
    context = "\n\n".join(context_documents)
    print(context)

    chain = LLMChain(llm=llm, prompt=ANSWER_PROMPT)
    response = chain({"context": context, "question": question})
    return response["text"]


if __name__ == "__main__":
    # Example usage:
    question = "what is IMPS and how much time does it take for the money to be credited"
    answer = answer_question(question)
    print("Question:", question)
    print("Answer:", answer)
