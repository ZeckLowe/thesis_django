import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
import hashlib
from pinecone import Pinecone
from datetime import date

load_dotenv()

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# EMBEDDINGS = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone Initialization
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("echo-openai")

# OpenAI Initialization
EMBEDDINGS = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=OPENAI_API_KEY)


def chunk_text(text, max_chunk_size=500):
    # Ensure each text ends with a newline to correctly split sentences
    if not text.endswith("\n"):
        text += "\n"

    # Split text into sentence
    sentences = text.split("\n")
    chunks = []
    current_chunk = ""

    # Iterate over sentence and assemble chunks
    for sentence in sentences:
        # Check if adding the current sentence exceeds the maximum chunk size
        if (len(current_chunk) + len(sentences) + 2 > max_chunk_size and current_chunk):
            # Add the current chunk to the list and start a new chunk
            chunks.append(current_chunk.strip())
            current_chunk = ""
        # Add the current sentence to the current chunk
        current_chunk += sentence.strip() + "\n"
    # Add any remaining text as the last chunk
    if (current_chunk):
        chunks.append(current_chunk.strip())

    return chunks # type: list[str]

def generate_embeddings(texts):
    """
    Generate embeddings for a list of text.
    """
    embedded = EMBEDDINGS.embed_documents(texts)

    print("Generating embeddings: Done!")
    return embedded

def generate_short_id(content):
    """
    Generate a short ID based on the content using SHA-256 hash.
    """
    hash_obj = hashlib.sha256()
    hash_obj.update(content.encode("utf-8"))

    return hash_obj.hexdigest()

def combine_vector_and_text(texts, meeting_title, date, text_embeddings):
    """
    Process a list of texts along with their embeddings.
    """
    data_with_metadata = []

    for doc_text, embedding in zip(texts, text_embeddings):
        if not isinstance(doc_text, str):
            doc_text = str(doc_text)

        if not isinstance(meeting_title, str):
            meeting_title = str(meeting_title)

        if not isinstance(date, str):
            date = str(date)

        text_id = generate_short_id(doc_text)
        data_item = {
            "id": text_id,
            "values": embedding,
            "metadata": {
                "text": doc_text,
                "title": meeting_title, # ADDED NEW METADATA
                "date": date}, # ADDED NEW METADATA
        }

        data_with_metadata.append(data_item)

    print("Combining vector and text: Done!")
    return data_with_metadata

def upsert_data_to_pinecone(data_with_metadata, namespace_name):
    """
    Upsert data with metadata into a Pinecone index.
    """
    index.upsert(vectors=data_with_metadata, namespace=namespace_name) # ADDED NAMESPACE
    print("Upserting vectors to Pinecone: Done!")

def Pinecone(texts, meeting_title):
    today = str(date.today()) # INITIALIZATION FOR DATE (DYNAMIC) BASED ON STORING
    namespace = 'USJ-R' # NAMESPACE DEFAULTED TO 'USJ-R' FOR ISOLATION (STATIC)

    chunked_text = chunk_text(texts=texts)
    chunked_text_embeddings = generate_embeddings(texts=chunked_text)
    data_with_meta_data = combine_vector_and_text(texts=chunked_text, meeting_title=meeting_title, date=today,  text_embeddings=chunked_text_embeddings)
    upsert_data_to_pinecone(data_with_metadata=data_with_meta_data, namespace_name=namespace)


# chunked_document = chunk_text_for_list(texts=text)
# chunked_document_embeddings = generate_embeddings(documents=chunked_document)
# data_with_meta_data = combine_vector_and_text(documents=chunked_document, doc_embeddings=chunked_document_embeddings)
# upsert_data_to_pinecone(data_with_metadata=data_with_meta_data)