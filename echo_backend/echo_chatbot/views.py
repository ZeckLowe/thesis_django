from django.shortcuts import render
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from django.http import JsonResponse
from .chatbot_django_qa import Chatbot
from .chatbot_django_rps import Pinecone
import echo_qa
import json

@api_view(['POST'])
def ask_question(request):
    question = request.data.get('question')
    user_id = request.data.get('user_id')
    session_id = request.data.get('session_id')

    if not question or not user_id or not session_id:
        return Response({'error': 'Question, user_id, and session_id are required.'}, status=400)
    
    try:
        answer = Chatbot(question)
        return Response({'answer': str(answer)}, status=200)
    except Exception as e:
        return Response({'error': str(e)}, status=500)
    
@api_view(['POST'])
def store_transcript(request):
    transcript = request.data.get('transcript')
    meeting_title = 'Sample Meeting Title'

    if not transcript:
        return Response({'error': 'No transcript provided'}, status=400)
    if not meeting_title:
        return Response({'error': 'No meeting title provided'}, status=400)
    
    try:
        Pinecone(transcript, meeting_title)
    except Exception as e:
        return Response({'error': str(e)}, status=500)
    
    return Response({'message': 'Transcript stored successfully!'}, status=200)


@csrf_exempt
def initialize_conversation_memory(request):
    if request.method == "POST":
        try:
            data = request.data.get('data')
            user_id = data["user_id"]
            session_id = data["session_id"]
            messages = data["messages"]

            memory = echo_qa.FirestoreConversationMemory(user_id=user_id, session_id=session_id)

            for message in messages:
                role = message.get("role")
                content = message.get("content")
                memory.chat_memory.add_message(role, content)

            return Response({"message": "Conversation memory initialized successfully"}, status=200)
        
        except Exception as e:
            return Response({"error": "Invalid request method"}, status=400)