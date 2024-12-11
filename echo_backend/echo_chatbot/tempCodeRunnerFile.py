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
print(OPENAI_API_KEY)