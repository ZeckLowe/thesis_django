from django.shortcuts import render
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from django.http import JsonResponse
from .chatbot_django_qa import Chatbot
from .chatbot_django_rps import Pinecone
from .echo_qa import CHATBOT
import json

@api_view(['POST'])
def ask_question(request):
    question = request.data.get('question')
    user_id = request.data.get('user_id')
    session_id = request.data.get('session_id')
    organization = request.data.get('organization')

    if not question or not user_id or not session_id:
        return Response({'error': 'Question, user_id, and session_id are required.'}, status=400)
    
    try:
        answer = CHATBOT(query=question, user_id=user_id, session_id=session_id, organization=organization)
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