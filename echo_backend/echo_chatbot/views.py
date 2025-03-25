from django.shortcuts import render
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from django.http import JsonResponse
from .chatbot_django_qa import Chatbot
# from .chatbot_django_rps import Pinecone
from .echo_qa import CHATBOT
from .echo_rps import PINECONE
import json
import re
import torch
from pyannote.audio import Pipeline
import whisper
from rest_framework import status
import tempfile
import os

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
    meeting_title = request.data.get('meetingTitle')
    organization = request.data.get('organization')

    if not transcript:
        return Response({'error': 'No transcript provided'}, status=400)
    if not meeting_title:
        return Response({'error': 'No meeting title provided'}, status=400)
    if not organization:
        return Response({'error': 'No organization provided'}, status=400)
    
    try:
        PINECONE(transcript, meeting_title, organization)
    except Exception as e:
        return Response({'error': str(e)}, status=500)
    
    return Response({'message': 'Transcript stored successfully!'}, status=200)

import os
import tempfile
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .transcription_service import TranscriptionService

@api_view(['POST'])
def transcribe_audio(request):
    """API endpoint for audio transcription."""
    if 'audio' not in request.FILES:
        return Response({"error": "No audio file provided."}, status=status.HTTP_400_BAD_REQUEST)

    audio_file = request.FILES['audio']

    if not audio_file.name.endswith('.wav'):
        return Response({"error": "Invalid file format. Please upload a WAV file."}, status=status.HTTP_400_BAD_REQUEST)

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
        for chunk in audio_file.chunks():
            temp_audio_file.write(chunk)
        temp_audio_file.flush()
        temp_audio_file_path = temp_audio_file.name

    try:
        # Process the audio file
        transcription_service = TranscriptionService()
        transcript_output = transcription_service.process_audio_file(temp_audio_file_path)
        return Response(transcript_output, status=status.HTTP_200_OK)

    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return Response({"error": "An error occurred during transcription."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    finally:
        # Clean up temporary file
        try:
            os.remove(temp_audio_file_path)
        except Exception as e:
            print(f"Error deleting temp file: {e}")
