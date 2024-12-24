import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_astradb import AstraDBVectorStore

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_ENDPOINT = os.environ["ASTRA_DB_ENDPOINT"]
ASTRA_DB_KEYSPACE = "demo"
ASTRA_DB_COLLECTION = "simple_demo"  # change this to your collection name

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def embed_and_store_text(text: str, filename: str, title: str, category: str):
    # Initialize embeddings and vector store
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = AstraDBVectorStore(
        embedding=embedding,
        collection_name=ASTRA_DB_COLLECTION,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_ENDPOINT,
        namespace = ASTRA_DB_KEYSPACE
    )

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    chunks = text_splitter.split_text(text)

    # Create Document objects with metadata
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "filename": filename,
                "title": title,
                "category": category
            }
        ) for chunk in chunks
    ]

    # Add documents to the vector store
    vectorstore.add_documents(documents)
    print(f"PDF '{filename}' indexed with {len(documents)} chunks.")

if __name__ == "__main__":
    # Example usage:
    script_dir = os.path.dirname(__file__)  # should be .../Demos/ingest
    pdf_path = os.path.join(script_dir, '..', 'assets', 'impa copy.pdf')
    pdf_path = os.path.abspath(pdf_path)
    title = "IMPS"
    category = "Operational"  # Assign a category as needed
    text = extract_text_from_pdf(pdf_path)
    embed_and_store_text(text, pdf_path, title, category)
    print("PDF content processed and stored successfully.")
