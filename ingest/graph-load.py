import os
import json
import warnings
import cassio
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI, OpenAIEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.graph_vectorstores import CassandraGraphVectorStore
from langchain_community.graph_vectorstores.extractors import (
    KeybertLinkExtractor,
    GLiNERLinkExtractor,
)

warnings.filterwarnings("ignore", lineno=0)

ASTRA_DB_APPLICATION_TOKEN = os.environ["ASTRA_DB_APPLICATION_TOKEN"]
ASTRA_DB_ENDPOINT = os.environ["ASTRA_DB_ENDPOINT"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ASTRA_DB_ID = os.environ["ASTRA_DB_ID"]
GRAPH_NODE_TABLE = os.environ["GRAPH_NODE_TABLE"]
KEYSPACE="demo"

# Initialize embeddings and LLM using OpenAI
# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings()

# Initialize Astra connection using Cassio
cassio.init(database_id=ASTRA_DB_ID, token=ASTRA_DB_APPLICATION_TOKEN)
store = CassandraGraphVectorStore(embeddings, table_name=GRAPH_NODE_TABLE, keyspace=KEYSPACE)


def main():
    """
    Main function to load, process, and visualize documents from a PDF file.

    This function loads documents from a specified PDF, transforms and cleans them,
    splits them into chunks, and adds them to a graph vector store.
    It also visualizes the documents as a text-based graph.
    """
    try:
        # Specify the path to your PDF file
        # Make sure this path is correct and points to an existing PDF file.
        script_dir = os.path.dirname(__file__)  # should be .../Demos/ingest
        pdf_path = os.path.join(script_dir, '..', 'assets', 'impa copy.pdf')
        pdf_path = os.path.abspath(pdf_path)

        # Load documents from the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # If you need to process large PDFs, consider using chunks of documents,
        # but often PDFs are loaded as a single large document.
        chunk_size = 10
        for i in range(0, len(documents), chunk_size):
            print(f"Processing documents {i + 1} to {i + chunk_size}...")
            document_chunk = documents[i:i + chunk_size]

            # If link extraction from PDFs is not needed, you can skip this step.
            # However, if you still want to run it, it may just not yield many links.
            from langchain_community.graph_vectorstores.extractors import LinkExtractorTransformer
            transformer = LinkExtractorTransformer([
                # For HTML/web documents we used KeyBERT link extractor.
                # It might not be as useful for PDF text, but you can still run it.
                KeybertLinkExtractor(),
            ])
            document_chunk = transformer.transform_documents(document_chunk)

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1024,
                chunk_overlap=64,
            )
            document_chunk = text_splitter.split_documents(document_chunk)

            # Run Named Entity Recognition (NER) extraction if desired
            ner_extractor = GLiNERLinkExtractor(["Person", "Organization", "Location", "Event", "Genre", "Topic"])
            transformer = LinkExtractorTransformer([ner_extractor])
            document_chunk = transformer.transform_documents(document_chunk)

            # Add documents to the graph vector store
            store.add_documents(document_chunk)

            # Visualize the graph text for the current chunk
            # visualize_graph_text(document_chunk)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
