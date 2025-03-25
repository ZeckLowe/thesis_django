import os
from langchain_openai import OpenAIEmbeddings
import hashlib
from pinecone import Pinecone
from datetime import date
import time
from pinecone import ServerlessSpec
from langchain_openai import ChatOpenAI
from .prompt_templates import prompt_templates
from firebase_admin import credentials, firestore
import firebase_admin


# Firestore Initialization
credential_path = r'C:\Users\user\OneDrive\Desktop\thesis_django\echo_backend\echo_chatbot\ServiceAccountKey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

if not firebase_admin._apps:
    cred = credentials.Certificate(r'C:\Users\user\OneDrive\Desktop\thesis_django\echo_backend\echo_chatbot\ServiceAccountKey.json')
    firebase_admin.initialize_app(cred)

db = firestore.Client()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Pinecone Initialization
pc = Pinecone(api_key=PINECONE_API_KEY)
index = ""
LLM = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")

# OpenAI Initialization
EMBEDDINGS = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=OPENAI_API_KEY)

def check_index(organization):
    """
    Check if an index exists in Pinecone. If not, create a new index.
    """
    index_name = organization
    index_list = pc.list_indexes()

    # If organization name does not exist, it creates new index
    if index_name not in index_list.names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
            deletion_protection="disabled"
        )
        # wait for index to be ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(index_name + " is created successfully.")
        index = pc.Index(index_name)

        return index
    else:
        print("Organization already exists.")
        return pc.Index(index_name)

def chunk_text_recursive(text, max_chunk_size=500):
    # Helper function for recursive chunking
    def recursive_chunk(sentences, current_chunk=""):
        # Base case: if no sentences are left, return the current chunk
        if not sentences:
            return [current_chunk.strip()] if current_chunk.strip() else []

        # Extract the next sentence
        sentence = sentences[0]
        remaining_sentences = sentences[1:]

        # Check if the sentence itself exceeds the max_chunk_size
        if len(sentence) > max_chunk_size:
            # Split the sentence into smaller parts
            split_parts = [
                sentence[i : i + max_chunk_size] for i in range(0, len(sentence), max_chunk_size)
            ]
            # Add the first part to the current chunk and handle the rest recursively
            return (
                [current_chunk.strip()] if current_chunk.strip() else []
            ) + split_parts + recursive_chunk(remaining_sentences, "")
        
        # Check if adding the current sentence exceeds the max_chunk_size
        if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            # Return the current chunk and continue with the next sentences
            return [current_chunk.strip()] + recursive_chunk(remaining_sentences, sentence.strip() + "\n")
        else:
            # Add the current sentence and continue recursively
            return recursive_chunk(remaining_sentences, current_chunk + sentence.strip() + "\n")

    # Ensure each text ends with a newline for sentence splitting
    if not text.endswith("\n"):
        text += "\n"

    # Split text into sentences by newline and filter out empty sentences
    sentences = [sentence for sentence in text.split("\n") if sentence.strip()]

    # Start recursive chunking
    return recursive_chunk(sentences)

def generate_embeddings(texts):
    """
    Generate embeddings for a list of text.
    """
    embedded = EMBEDDINGS.embed_documents(texts)

    print("Generating embeddings: Done!")
    return embedded

def generate_summary(texts, meeting_title, date):
    """
    Generate a summary of a transcript via OpenAI.
    """
    prompt = prompt_templates.summary_template().format(date=date, meeting_title=meeting_title, texts=texts)
    summary = LLM.invoke(prompt)
    print("Generating summary: Done!")

    return summary

def generate_short_id(content):
    """
    Generate a short ID based on the content using SHA-256 hash.
    """
    # Generate short id
    hash_obj = hashlib.sha256()
    hash_obj.update(content.encode("utf-8"))

    print("Generating short id: Done!")
    return hash_obj.hexdigest()

def combine_vector_and_text(texts, meeting_title, text_embeddings):
    """
    Process a list of texts along with their embeddings.

    Parameters:
    - texts (List[str]): List of chunked text
    - meeting_title (str): Title of the meeting
    - text_embeddings (List[List[float]]): Vector embeddings of the corresponding texts
    Output: List
    """
    # Date Today
    today = str(date.today())
    
    data_with_metadata = []

    # Creates list that contains id, values, and metadata
    for doc_text, embedding in zip(texts, text_embeddings):
        if not isinstance(doc_text, str):
            doc_text = str(doc_text)

        if not isinstance(meeting_title, str):
            meeting_title = str(meeting_title)

        if not isinstance(today, str):
            today = str(today)

        text_id = generate_short_id(doc_text)
        data_item = {
            "id": text_id,
            "values": embedding,
            "metadata": {"text": doc_text, "title": meeting_title, "date": today},
        }

        data_with_metadata.append(data_item)

    print("Combining vector and text: Done!")
    return data_with_metadata

def upsert_data_to_pinecone(data_with_metadata, namespace, index):
    """
    Upsert data with metadata into a Pinecone index.
    """
    index.upsert(vectors=data_with_metadata, namespace=namespace)
    time.sleep(2)
    print("Upserting vectors to Pinecone: Done!")

def store_summary_to_firestore(summary, organization, meeting_title):
    """
    Store the summary and its embeddings to Firestore.
    
    Parameters:
    - summary (str): The summary text to store.
    - organization (str): The organization's name associated with the meeting.
    - meeting_title (str): The title of the meeting to update.
    """
    try:
        summary_text = summary.content
        
        # Reference the 'meetings' collection and filter by organization and meeting title
        doc_ref = db.collection('Meetings').document(meeting_title)
        
        # Get the document reference
        doc = doc_ref.get()
        
        # Check if any documents match the query
        if not doc.exists:
            print("No matching meeting found for the specified organization and title.")
            return
        
        doc_ref.update({
                'meetingSummary': summary_text,  # Store the summary text
            })
        print(f"Updated document: {meeting_title} with summary and embeddings.")

        
        print("Summary and embeddings successfully stored in Firestore.")
    except Exception as e:
        print(f"An error occurred while storing the summary: {e}")


def PINECONE(texts, meeting_title, organization):
    today = str(date.today()) # INITIALIZATION FOR DATE (DYNAMIC) BASED ON STORING
    index = check_index(organization.lower().replace(" ","-"))
    namespace = meeting_title

    chunked_text = chunk_text_recursive(text=texts)
    chunked_text_embeddings = generate_embeddings(texts=chunked_text)
    data_with_meta_data = combine_vector_and_text(texts=chunked_text, meeting_title=meeting_title,  text_embeddings=chunked_text_embeddings)
    upsert_data_to_pinecone(data_with_metadata=data_with_meta_data, namespace=namespace, index=index)
    store_summary_to_firestore(meeting_title=meeting_title, organization=organization, summary=generate_summary(texts=texts, meeting_title=meeting_title, date=today))

