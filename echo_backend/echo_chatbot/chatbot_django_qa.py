import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import hashlib
from pinecone import Pinecone
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
import json
import ast
from rapidfuzz import fuzz
from datetime import datetime

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# EMBEDDINGS = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Pinecone Initialization
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("echo-openai")

# OpenAI Initialization
client=OpenAI(api_key=OPENAI_API_KEY)
LLM = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
EMBEDDINGS = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=OPENAI_API_KEY)


def get_query_embeddings(query):
    """
    This function returns a list of the embeddings for a given query
    """
    query_embeddings = EMBEDDINGS.embed_query(query)
    return query_embeddings

def fuzzy_match(title1, title2, threshold=80):
    """
    Perform a fuzzy match between two titles using RapidFuzz.
    Returns True if the similarity score is above the threshold.
    """
    similarity_score = fuzz.partial_ratio(title1.lower(), title2.lower())
    return similarity_score >= threshold

def extract_metadata_from_query(query):
    prompt = f"""
        You are a helpful assistant. Extract the meeting title and the meeting date from the following query.
        If the meeting title or date is not explicitly mentioned, return 'unknown'.
        If the date is mentioned as word, it should be formatted as 'YYYY-MM-DD'

        Query: {query}

        Provide the meeting title and date as a Python dictionary in this format:
        {{"meeting_title": "title_here", "date": "date_here"}}
        """
    
    response = LLM.invoke(prompt)
    metadata_str = response.content.strip()
    metadata_dict = ast.literal_eval(metadata_str)
    return metadata_dict

def query_pinecone_index(query_embeddings, meeting_title, date, top_k=2, include_metadata=True):
    """
    Query a Pinecone index.
    """
    filter_conditions = {}
    if date.lower() != 'unknown':
        filter_conditions['date'] = date

    query_response = index.query(
        vector=query_embeddings,
        top_k=top_k,
        include_metadata=include_metadata,
        namespace="USJ-R",
        filter=filter_conditions)
    
    filtered_matches = [] # ADDED FUZZY FILTER
    for match in query_response['matches']:
      if 'metadata' in match and 'title' in match['metadata']:
        metadata_title = match['metadata']['title']
        if fuzzy_match(meeting_title, metadata_title):
          filtered_matches.append(match)

    if not filtered_matches:
      return query_response

    query_response['matches'] = filtered_matches
    print(query_response)
    return query_response

def better_query_response(prompt):
    """
    This function returns a better response using LLM
    """
    better_answer = LLM.invoke(prompt)
    return better_answer

def Chatbot(_query):

    print(_query)
    _metadata = extract_metadata_from_query(_query)
    print(_metadata)
    _meeting_title = _metadata.get('meeting_title', 'unknown')
    _date = _metadata.get('date', 'unknown')

    _query_embeddings = get_query_embeddings(query=_query)
    _answers = query_pinecone_index(
       query_embeddings=_query_embeddings,
       meeting_title=_meeting_title,
       date=_date)
    
    _text_answer = " ".join([doc['metadata']['text'] for doc in _answers['matches']])
    _prompt = f"""You are a meeting facilitator.
        This Human will ask you a questions about the conversation of the meeting. 
        Use following piece of context to answer the question. 
        If you don't know the answer, just say you don't know. 
        Keep the answer within two to five sentences and concise.
        Context: {_text_answer}
        Question: {_query}"""
    _final_answer = better_query_response(prompt=_prompt)
    return _final_answer.content

# answer = Chatbot(_query="Provide details about the meeting on September 13, 2024")
# print(answer)

# query = "Who is the first speaker of the meeting?"
# query_embeddings = get_query_embeddings(query=query)
# answers = query_pinecone_index(query_embeddings=query_embeddings)

# text_answer = " ".join([doc['metadata']['text'] for doc in answers['matches']])

# prompt = f"""You are a meeting facilitator.
#         This Human will ask you a questions about the conversation of the meeting. 
#         Use following piece of context to answer the question. 
#         If you don't know the answer, just say you don't know. 
#         Keep the answer within 2 sentences and concise.
#         Context: {text_answer}
#         Question: {query}"""

# final_answer = better_query_response(prompt=prompt)
# print(final_answer.content)