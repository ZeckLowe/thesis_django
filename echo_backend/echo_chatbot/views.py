from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .chatbot_django_qa import Chatbot
from .chatbot_django_rps import Pinecone

@api_view(['POST'])
def ask_question(request):
    question = request.data.get('question')
    if not question:
        return Response({'error': 'No question provided'}, status=400)
    
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

