import fitz  # PyMuPDF
import os
from astrapy import DataAPIClient
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=1000, overlap=250):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# Initialize AstraDB client
db = DataAPIClient(os.environ["ASTRA_DB_APPLICATION_TOKEN"]).get_database(os.environ["ASTRA_DB_ENDPOINT"])


# Create a collection (if not already created)
collection_name = "simple_vectorize"
collection = db.get_collection(collection_name, keyspace="demo")

# Extract text from PDF
script_dir = os.path.dirname(__file__)  # should be .../Demos/ingest
pdf_path = os.path.join(script_dir, '..', 'assets', 'impa copy.pdf')
pdf_path = os.path.abspath(pdf_path)
pdf_text = extract_text_from_pdf(pdf_path)

# Chunk the extracted text
chunks = chunk_text(pdf_text, chunk_size=1000, overlap=250)

# Prepare documents for insertion
documents = [{"$vectorize": chunk} for chunk in chunks]

# Insert documents into the collection
res = collection.insert_many(documents)
print(f"Inserted {len(res.inserted_ids)} chunks.")