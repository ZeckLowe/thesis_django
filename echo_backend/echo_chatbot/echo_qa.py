import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import hashlib
from pinecone import Pinecone
from langchain_openai import OpenAI
from langchain_core.messages import HumanMessage, AIMessage
from sklearn.metrics.pairwise import cosine_similarity
import firebase_admin
import google.cloud
from firebase_admin import credentials, firestore
from .prompt_templates import prompt_templates
from google.cloud.firestore_v1.base_query import FieldFilter
from sentence_transformers import CrossEncoder
from collections import Counter
from fuzzywuzzy import fuzz

# Firestore Initialization
# credential_path = r'C:\Users\user\OneDrive\Desktop\thesis_django\echo_backend\echo_chatbot\ServiceAccountKey.json'
credential_path = r'C:\Codes\Django\thesis_django\echo_backend\echo_chatbot\ServiceAccountKey.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

if not firebase_admin._apps:
    # cred = credentials.Certificate(r'C:\Users\user\OneDrive\Desktop\thesis_django\echo_backend\echo_chatbot\ServiceAccountKey.json')
    cred = credentials.Certificate(r'C:\Codes\Django\thesis_django\echo_backend\echo_chatbot\ServiceAccountKey.json')
    firebase_admin.initialize_app(cred)

try:
    db = firestore.Client()
    print("*Firestore connected successfully!")
except Exception as e:
    print(f"Failed to connect to Firestore: {e}")

# API Keys Initialization
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

if not OPENAI_API_KEY:
    print("OpenAI API Key not found!")
if not PINECONE_API_KEY:
    print("Pinecone API Key not found!")

# Pinecone Initialization
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("*Pinecone connected successfully!")
except Exception as e:
    print(f"Failed to connect to Pinecone: {e}")


# OpenAI Initialization
try:
    client=OpenAI(api_key=OPENAI_API_KEY)
    LLM = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    EMBEDDINGS = OpenAIEmbeddings(model='text-embedding-3-small')
    print("*OpenAI connected successfully!")
except Exception as e:
    print(f"Failed to connect to OpenAI: {e}")

# CrossEncoder Initialization
try:
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
    print("*CrossEncoder connected successfully!")
except Exception as e:
    print(f"Failed to connect to CrossEncoder: {e}")

# Get Embeddings
def get_embeddings(text):
    """
    This function returns a list of the embeddings for a given query
    """
    text_embeddings = EMBEDDINGS.embed_query(text)
    print("Generating Embeddings: Done!")
    return text_embeddings

def resolve_namespace(query, summaries, user_id, session_id):
    """
    Resolves the namespace by selecting the most similar one using fuzzy matching (fuzzywuzzy).
    """
    # def ambiguous_fuzzy(query_embeddings, summaries):
    #     """
    #     Rank namespaces by semantic similarity to the query.
    #     """   
    #     # Compute similarity with meeting summaries
    #     summary_embeddings = {title: get_embeddings(summary) for title, summary in summaries.items()}
    #     print("Generated summary embeddings:", summary_embeddings)

    #     summary_similarities = {
    #         title: cosine_similarity([query_embeddings], [embedding])[0][0] for title, embedding in summary_embeddings.items()
    #     }
    #     print("Computed Summary Similarity:", summary_similarities)

    #     # Rank by similarity
    #     ranked_candidates = sorted(summary_similarities.items(), key=lambda x: x[1], reverse=True)
    #     print("\n🔹 Initial Ranking (Cosine Similarity):", ranked_candidates)
        
    #     score_diff = ranked_candidates[0][1] - ranked_candidates[1][1]
    #     print("Score difference:", score_diff)
        
    #     # Prepare input for re-ranking
    #     cross_encoder_inputs = [(summaries[title], query) for title, _ in ranked_candidates]

    #     # Compute cross-encoder scores
    #     scores = reranker.predict(cross_encoder_inputs)

    #     # Re-rank based on cross-encoder scores
    #     reranked_candidates = sorted(zip(ranked_candidates, scores), key=lambda x: x[1], reverse=True)
    #     print("\n🔹 Cross Encoder:", reranked_candidates)

    #     score_diff = reranked_candidates[0][1] - reranked_candidates[1][1]
    #     print("Score difference:", score_diff)

    #     if score_diff < 0.9:
    #         print("Ambiguous in Cross Encoder")
    #         return ""

    #     print("\n🔹 Re-ranked Candidates (Cross-Encoder):", reranked_candidates)
        
    #     return reranked_candidates[0][0][0]
    def store_primary_namespace(user_id, session_id, primary_namespace):
        """
        Store the primary namespace in Firestore.
        """
        doc_ref = db.collection("chatHistory").document(user_id).collection("session").document(session_id)
        try:
            doc_ref.update({
                'primary_namespace': primary_namespace
            })
        except Exception as e:
            print(f"Error updating chat history: {str(e)}")
        print(f"Stored primary namespace '{primary_namespace}' for user_id={user_id}, session_id={session_id}")

    def get_primary_namespace(user_id, session_id):
        """
        Fetch the primary namespace for the user and session.
        """
        doc_ref = db.collection("chatHistory").document(user_id).collection("session").document(session_id)
        doc_snapshot = doc_ref.get()
        try:
            if doc_snapshot.exists:
                primary_namespace = doc_snapshot.get('primary_namespace')
                if primary_namespace is None:
                    print(f"No 'namespace' field found in document for user_id={user_id}, session_id={session_id}")
                    return ""

                print(f"Primary Namespace Initialized: {primary_namespace}")
            else:
                print(f"No document found for user_id={user_id}, session_id={session_id}")
        except Exception as e:
            print(f"Error initializing chat history: {str(e)}")
        
        return primary_namespace

    def get_most_similar_namespace(query, summaries, user_id, session_id):
        """
        Rank namespaces by fuzzy matching (using fuzzywuzzy's token_set_ratio).
        """
        similarities = {
            title: (fuzz.token_set_ratio(query.lower(), f"{title}".lower()) + fuzz.token_set_ratio(query.lower(), f"{summary}".lower()))/2
            for title, summary in summaries.items()
        }

        print("Computed fuzzy similarities:", similarities)

        # Rank namespaces based on similarity score
        ranked_namespaces = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        print("Ranked namespaces:", ranked_namespaces)

        # Check for ambiguity
        if len(ranked_namespaces) > 1:
            diff = ranked_namespaces[0][1] - ranked_namespaces[1][1]
            if diff < 15:
                print("Ambiguous fuzzy match.")
                return ""
            
        store_primary_namespace(user_id, session_id, ranked_namespaces[0][0])
        return ranked_namespaces[0][0] if ranked_namespaces else ""

    primary_namespace = get_primary_namespace(user_id, session_id)
    if primary_namespace == "":
        primary_namespace = get_most_similar_namespace(query, summaries, user_id, session_id)
    print(f"Selected namespace: {primary_namespace}")
    return primary_namespace

def generate_followup_question(question, meeting_summaries):
    """
    Generate followup response based on the previous query.
    """
    followup_prompt = prompt_templates.followup_template().format(question=question, meeting_list=meeting_summaries)
    followup_response = LLM.invoke(followup_prompt)
    print("Generating followup question: Done!")

    return followup_response.content

# Get Relevant Documents
def query_pinecone_index(query_embeddings, meeting_title, index, top_k=5, include_metadata=True):
    """
    Query a Pinecone index.
    """
    # Build filter conditions directly for Pinecone
    filter_conditions = {}

    # Include date and meeting title if specified
    if meeting_title.lower() != 'unknown':
        filter_conditions['title'] = meeting_title

    # Query Pinecone using the build filter conditions
    query_response = index.query(
        vector=query_embeddings,
        filter=filter_conditions,
        top_k=top_k,
        include_metadata=include_metadata,
        namespace=meeting_title )

    print("Querying Pinecone Index: Done!")
    return " ".join([doc['metadata']['text'] for doc in query_response['matches']]), [doc['metadata']['date'] for doc in query_response['matches']], [doc['metadata']['title'] for doc in query_response['matches']]

def fetch_summaries_by_organization(organization):
        """
        Fetches summaries by organization
        """
        summaries = {}
        meetings_ref = db.collection("Meetings")
        query = meetings_ref.where(filter=FieldFilter("organization", "==", organization))
        docs = query.stream()

        for doc in docs:
            data = doc.to_dict()
            meeting_title = data.get("meetingTitle")
            summary = data.get("meetingSummary")
            if meeting_title and summary:
                summaries[meeting_title] = summary
        
        print(f"Fetched summaries for organization '{organization}': {summaries}")
        return summaries

def decomposition_query_process(question, text_answers, chat_history, text_date, text_title):
    """Implements decomposition query"""
    
    def decompose_question(question, chat_history):
        """
        Decomposes a complex question into smaller questions.
        """
        prompt = prompt_templates.decomposition_template().format(question=question, chat_history=chat_history)
        response = LLM.invoke(prompt)
        subquestions = response.content.split("\n")
        print("Decomposing Question: Done!")

        return subquestions
    
    def generate_qa_pairs(subquestions, context):
        """Generates QA pairs by answering each subquestion."""
        qa_pairs = []
        for subquestion in subquestions:
            context = context
            rag_prompt = prompt_templates.qa_template().format(context=context, subquestion=subquestion)
            answer = LLM.invoke(rag_prompt)
            qa_pairs.append((subquestion, answer))
        print("Generating QA Pairs: Done!")

        return qa_pairs
    
    def build_final_answer(question, context, chat_history, qa_pairs, text_date, text_title):
        """Builds a final answer by integrating the context and QA pairs."""
        qa_pairs_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs])
        # final_prompt = prompt_templates.final_rag_template().format(context=context, qa_pairs=qa_pairs_str, question=question)
        final_prompt = prompt_templates.final_rag_template_with_memory().format(context=context, qa_pairs=qa_pairs_str, question=question, chat_history=chat_history, text_date=text_date, text_title=text_title)
        final_response = LLM.invoke(final_prompt)
        print("Building Final Answer: Done!")

        return final_response
    
    subquestions = decompose_question(question)
    qa_pairs = generate_qa_pairs(subquestions, text_answers)
    print(qa_pairs)
    final_answer = build_final_answer(question, text_answers, chat_history, qa_pairs, text_date, text_title)

    return final_answer.content

def initialize_chat_history(user_id, session_id):
    """
    Initializes a chat history object.
    """
    chat_history = []
    doc_ref = db.collection("chatHistory").document(user_id).collection("session").document(session_id)
    doc_snapshot = doc_ref.get()
    try:
        if doc_snapshot.exists:
            messages = doc_snapshot.get('messages')
            if messages is None:
                print(f"No 'messages' field found in document for user_id={user_id}, session_id={session_id}")
                return chat_history
            messages = doc_snapshot.get('messages')

            for message in messages:
                chat_history.append(message)
            print(f"Chat History Initialized: {chat_history}")
        else:
            print(f"No document found for user_id={user_id}, session_id={session_id}")
    except Exception as e:
        print(f"Error initializing chat history: {str(e)}")
    
    return chat_history

def update_chat_history(user_id, session_id, chat_history):
    """
    Updates the chat history object.
    """
    doc_ref = db.collection("chatHistory").document(user_id).collection("session").document(session_id)
    try:
        doc_ref.update({
            'messages': chat_history
        })
    except Exception as e:
        print(f"Error updating chat history: {str(e)}")

def process_chat_history(chat_history):
    """
    Changes the chat history list into a HumanMessages and AIMessages Schema
    """
    process_chat_history = []
    for idx, message in enumerate(chat_history):
        if idx % 2 == 0:
            process_chat_history.append(HumanMessage(message))
        else:
            process_chat_history.append(AIMessage(message))

        
    return process_chat_history

def CHATBOT(query, user_id, session_id, organization):
    print(f"Question: {query}")
    print(f"Current User ID: {user_id}")
    print(f"Current Session ID: {session_id}")
    print(f"Organization: {organization}")
    index = pc.Index(organization.lower())

    chat_history = initialize_chat_history(user_id=user_id, session_id=session_id)
    summaries = fetch_summaries_by_organization(organization=organization)

    query_embeddings = get_embeddings(text=query)
    meeting_title = resolve_namespace(query=query, query_embeddings=query_embeddings, summaries=summaries)

    if meeting_title == "":
        print("AMBIGUOUS MATCH")
        response = generate_followup_question(query, summaries)
        
        chat_history.append(query)
        chat_history.append(response)
        update_chat_history(user_id, session_id, chat_history)
        return response

    text_answers, text_date, text_title = query_pinecone_index(query_embeddings=query_embeddings, meeting_title=meeting_title, index=index)
    print(f"Retrieved context: {text_answers}\nDate context: {text_date[0]}\nTitle Context: {text_title[0]}")

    # chat_history 

    response = decomposition_query_process(question=query, text_answers=text_answers, chat_history=process_chat_history(chat_history), text_date=text_date[0], text_title=text_title[0])

    chat_history.append(query)
    chat_history.append(response)
    update_chat_history(user_id, session_id, chat_history)

    print("User Query:", query)
    print("Chatbot Response:", response)

    return response

# CHATBOT(query="What is the project about?", user_id="qN3L5d7p5VZIHiHvDYy2o8b7ihX2", session_id="4", organization="SCS")