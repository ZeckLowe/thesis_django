import os
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import hashlib
from pinecone import Pinecone
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from google.cloud import firestore

# Firestore Initialization
db = firestore.Client()

# API Keys Initialization
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Pinecone Initialization
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("scs")

# OpenAI Initialization
client=OpenAI(api_key=OPENAI_API_KEY)
LLM = ChatOpenAI(temperature=0, model_name="gpt-4-turbo")
EMBEDDINGS = OpenAIEmbeddings(model='text-embedding-3-small')

# Firestore Conversation Memory
class FirestoreConversationMemory(ConversationBufferMemory):
    """
    Extends ConversationBufferMemory to Firestore
    """
    def __init__(self, user_id, session_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_id = user_id
        self.session_id = session_id

    def save_to_firestore(self, message):
        """
        Saves the current conversation buffer to Firestore
        """
        doc_ref = db.collection('chatHistory').document(self.user_id).collection(self.session_id)
        for i in message in enumerate(self.chat_memory.messages):
            doc_ref.document(f"message_{i}").set({
                "role": message.role,
                "content": message.context,
                "timestamp": message.created_at.isoformat() if hasattr(message, 'created_at') else None
            })

    def add_message(self, role, content):
        """
        Adds a new message to the buffer and saves it to firstore
        """
        self.chat_memory.add_message(role, content)
        self.save_to_firestore()

# Initialize Memory
def initialize_memory(user_id, session_id):
    """
    Initializes the conversation buffer memory for a user and session
    """
    def load_conversation_history(user_id, session_id):
        """
        Loads the conversation history from firestore
        """
        doc_ref = db.collection('chatHistories').document(user_id).collection(session_id)
        messages = doc_ref.stream()

        conversation_history = []
        for message in messages:
            data = message.to_dict()
            conversation_history.append({
                "role": data["role"],
                "content": data["content"],
                "timestamp": data["timestamp"]
            })

        return conversation_history

    # Load previous conversation history if it exists
    existing_history = load_conversation_history(user_id, session_id)

    # Initialize FirestoreConversationMemory
    memory = FirestoreConversationMemory(user_id=user_id, session_id=session_id)

    # Fill in the memory with previous message
    for message in existing_history:
        memory.chat_memory.add_message(message["role"], message["content"])

        return memory

# Get Namespaces
def get_meeting_titles():
    """
    Retrieves meeting titles for namespace
    """
    meeting_titles = []
    meetings_ref = db.collection('Meetings')

    try:
        documents = meetings_ref.stream()
        for doc in documents:
            data = doc.to_dict()
            if 'meetingTitle' in data:
                meeting_titles.append(data['meetingTitle'])
        print("Meeting Titles Retrieved:", meeting_titles)
    except Exception as e:
        print(f"Error retrieving meeting titles: {str(e)}")

    return meeting_titles

# Prompt Templates
class prompt_templates:
    def final_rag_template():
        prompt = """
            You are a meeting facilitator.
            This user will ask you a questions about the conversation of the meeting.
            Use following piece of context to answer the question.
            If you don't know the answer, just say you don't know.
            Keep the answer complete and concise.
            Context: {context}
            Here are some background questions and answers that will help you answer the question: {qa_pairs}
            Question: {question}
        """
        
        return ChatPromptTemplate.from_template(prompt)
    
    def decomposition_template():
        prompt = """
            Break the following user question into smaller, more specific questions.
            Provide these subquestions separated by newlines. 
            Do not rephrase if you see unknown terms.
            Question: {question}
            subquestions:
        """

        return ChatPromptTemplate.from_template(prompt)
    
    def qa_template():
        prompt = """
            Answer the question in the following context:\n{context}\n\nQuestion: {subquestion}
        """

        return ChatPromptTemplate.from_template(prompt)
    
    def conversation_history_template():
        prompt = """
            Conversation History: {memory}\nUser Query: {query}\nResponse: {initial_response}\nRefined Answer:
        """

        return ChatPromptTemplate.from_template(prompt)

# Get Embeddings
def get_embeddings(text):
    """
    This function returns a list of the embeddings for a given query
    """
    text_embeddings = EMBEDDINGS.embed_query(text)
    print("Generating Embeddings: Done!")
    return text_embeddings

# Get Namespace
def resolve_namespace(query_embeddings, namespaces):
    """
    Resolves the namespace by either selecting the most similar one or prompting the user for clarification.
    """
    def get_most_similar_namespace(query_embeddings, namespaces, threshold=0.05):
        """
        Rank namespaces by semantic similarity to the query.
        """
        # Get embeddings for each namespace in list
        namespace_embeddings = {ns: get_embeddings(ns) for ns in namespaces}
        print(namespace_embeddings)

        # Compute similarities
        similarities = {
            ns: cosine_similarity([query_embeddings], [embedding])[0][0] for ns, embedding in namespace_embeddings.items()
        }
        print(similarities.items)

        # Rank namespaes by similarity score
        ranked_namespaces = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        print(ranked_namespaces)

        # Check if the top two are close in similarity
        top_two = ranked_namespaces[:2]
        print(top_two)
        if len(top_two) > 1 and abs(top_two[0][1] - top_two[1][1]) < threshold:
            return None, top_two # Ambiguous case, return for user clarification
        
        return ranked_namespaces[0][0], ranked_namespaces

    def clarify_with_user(ambiguous_namespaces):
        """
        Ask the user to clarify when multiple namespaces are similar.
        """
        options = [ns[0] for ns in ambiguous_namespaces]
        print(options)
        print(f"Did you mean:\n1. {options[0]}\n2. {options[1]}")

        # Simulate user input for demonstration
        user_choice = int(input("Please choose 1 or 2: "))-1
        return options[user_choice]

    namespace, ranked = get_most_similar_namespace(query_embeddings, namespaces)
    print(namespaces, ranked)

    if namespace:
        print(f"Selected namespace: {namespace}")
        return namespace
    else:
        print("Ambiguity detected!")
        return clarify_with_user(ranked)

# Get Relevant Documents
def query_pinecone_index(query_embeddings, meeting_title, top_k=5, include_metadata=True):
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
        namespace=meeting_title) # Filter based on metadata

    print("Querying Pinecone Index: Done!")
    return " ".join([doc['metadata']['text'] for doc in query_response['matches']])

def decomposition_query_process(question, text_answers, memory):
    """Implements decomposition query"""

    def output_parser(output):
        """
        Helps parses the LLM output, prints it, and returns it.
        """
        print("\n" + output.content + "\n")

        return output.content

    def decompose_question(question):
        """
        Decomposes a complex question into smaller questions.
        """
        prompt = prompt_templates.decomposition_template().format(question=question)
        response = LLM.invoke(prompt)
        subquestions = response.content.split("\n")

        return subquestions
    
    def generate_qa_pairs(subquestions, context):
        """Generates QA pairs by answering each subquestion."""
        qa_pairs = []
        for subquestion in subquestions:
            context = context
            rag_prompt = prompt_templates.qa_template().format(context=context, subquestion=subquestion)
            answer = LLM.invoke(rag_prompt)
            qa_pairs.append((subquestion, answer))

        return qa_pairs
    
    def build_final_answer(question, context, qa_pairs):
        """Builds a final answer by integrating the context and QA pairs."""
        qa_pairs_str = "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs])
        final_prompt = prompt_templates.final_rag_template().format(context=context, qa_pairs=qa_pairs_str, question=question)
        final_response = LLM.invoke(final_prompt)

        return final_response
    
    memory.add_message(role="user", content=question)
    
    subquestions = decompose_question(question)
    qa_pairs = generate_qa_pairs(subquestions, text_answers)
    print(qa_pairs)
    final_answer = build_final_answer(question, text_answers, qa_pairs)

    memory.add_message(role="bot", content=final_answer.content)

    return output_parser(final_answer)

def CHATBOT(query, user_id, session_id):
    # Initialize memory
    memory = initialize_memory(user_id=user_id, session_id=session_id)

    namespaces = get_meeting_titles()
    if not namespaces:
        return "no meeting titles found in the database."
    # namespaces = ["Kickoff Meeting", "Project Meeting"]

    query_embeddings = get_embeddings(text=query)
    meeting_title = resolve_namespace(query_embeddings=query_embeddings, namespaces=namespaces)
    text_answers = query_pinecone_index(query_embeddings=query_embeddings, meeting_title=meeting_title)
    print(f"Retrieved context: {text_answers}")
    response = decomposition_query_process(question=query, text_answers=text_answers, memory=memory)

    print("User Query:", query)
    print("Chatbot Response:", response)