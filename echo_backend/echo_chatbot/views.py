from django.shortcuts import render
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from django.http import JsonResponse
from .chatbot_django_qa import Chatbot
# from .chatbot_django_rps import Pinecone
from .echo_qa import CHATBOT
from .echo_rps import PINECONE
# import json
# import re
# import torch
# from pyannote.audio import Pipeline
# import whisper
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
# from .transcription_service import TranscriptionService
import logging
from django.conf import settings
from pydub import AudioSegment  # Add this import
from .transcription_service import (
    transcribe_audio_file, 
    extract_speaker_embedding, 
    store_speaker_embedding, 
    pinecone_index,
    init_pinecone,
    pc_client,
    find_matching_speaker,
    convert_to_wav
)

logger = logging.getLogger(__name__)

@api_view(['POST'])
def transcribe_audio(request):
    """
    API endpoint to process and transcribe an uploaded audio file.
    Returns a structured response with identified speakers and their transcribed speech.
    
    Parameters:
    - audio: Audio file to transcribe (required)
    
    Returns:
    - 200: Successful transcription with transcript data
    - 400: Invalid request (missing file or wrong format)
    - 500: Server error during processing
    """
    # Validate file presence
    if 'audio' not in request.FILES:
        logger.warning("No audio file provided in request")
        return Response(
            {"error": "No audio file provided. Please upload an audio file using the 'audio' field."}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    audio_file = request.FILES['audio']
    
    # Validate file extension
    allowed_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.wma', '.aiff', '.alac', '.opus', '.webm', '.mp4']
    file_ext = os.path.splitext(audio_file.name)[1].lower()
    
    if file_ext not in allowed_extensions:
        logger.warning(f"Invalid file format uploaded: {audio_file.name}")
        return Response(
            {"error": f"Invalid file format. Supported formats: {', '.join(allowed_extensions)}"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    # Validate file size (e.g., 50MB max)
    max_size = getattr(settings, 'MAX_AUDIO_FILE_SIZE', 50 * 1024 * 1024)  # 50MB default
    if audio_file.size > max_size:
        logger.warning(f"File too large: {audio_file.size} bytes")
        return Response(
            {"error": f"File too large. Maximum size is {max_size//(1024*1024)}MB"}, 
            status=status.HTTP_400_BAD_REQUEST
        )

    # Ensure Pinecone is initialized
    try:
        if pinecone_index is None:
            init_pinecone()
        logger.info("Pinecone connection verified")
    except Exception as e:
        logger.error(f"Pinecone initialization error: {str(e)}")
        return Response(
            {"error": f"Could not connect to vector database: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    temp_audio_file_path = None
    temp_wav_file_path = None
    
    try:
        # Convert the uploaded file to WAV format
        logger.info(f"Processing audio file: {audio_file.name}")
        temp_wav_file_path = convert_to_wav(audio_file)
        
        logger.info(f"Starting transcription for file: {audio_file.name}")
        transcript_output = transcribe_audio_file(temp_wav_file_path)
        
        if not transcript_output:
            logger.info("Audio processed but no speech segments detected")
            return Response(
                {"message": "Audio processed, but no speech segments were detected."}, 
                status=status.HTTP_200_OK
            )
        
        logger.info(f"Successfully transcribed {len(transcript_output)} segments")
        
        # Get information about stored embeddings using the new API
        index_stats = pinecone_index.describe_index_stats()
        vector_count = index_stats['total_vector_count']
        
        return Response(
            {
                "status": "success",
                "transcript": transcript_output,
                "original_filename": audio_file.name,
                "segment_count": len(transcript_output),
                "embeddings_stored": vector_count
            }, 
            status=status.HTTP_200_OK
        )
        
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}", exc_info=True)
        return Response(
            {
                "status": "error",
                "error": "An error occurred during transcription",
                "details": str(e)
            }, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
    finally:
        # Clean up temporary files
        try:
            if temp_wav_file_path and os.path.exists(temp_wav_file_path):
                os.remove(temp_wav_file_path)
                logger.info(f"Removed temporary WAV file: {temp_wav_file_path}")
        except Exception as e:
            logger.error(f"Error deleting temp WAV file: {str(e)}")


@api_view(['POST'])
def store_single_speaker(request):
    """
    Test endpoint to extract and store embeddings for a single speaker.
    
    Parameters:
    - audio: Audio file with a single speaker (required)
    - speaker_name: Name of the speaker (required)
    
    Returns:
    - 200: Successfully extracted and stored embedding
    - 400: Invalid request
    - 500: Server error during processing
    """
    # Start with comprehensive error handling
    try:
        # Validate input parameters
        if 'audio' not in request.FILES:
            logger.warning("No audio file provided")
            return Response(
                {"error": "No audio file provided."}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        if 'speaker_name' not in request.data or not request.data['speaker_name'].strip():
            logger.warning("No speaker name provided")
            return Response(
                {"error": "Speaker name is required."}, 
                status=status.HTTP_400_BAD_REQUEST
            )
            
        speaker_name = request.data['speaker_name'].strip()
        audio_file = request.FILES['audio']
        
        logger.info(f"Processing single speaker request: {speaker_name}, file: {audio_file.name}")
        
        # Validate file extension
        allowed_extensions = ['.wav', '.mp3', '.ogg']
        file_ext = os.path.splitext(audio_file.name)[1].lower()
        
        if file_ext not in allowed_extensions:
            logger.warning(f"Invalid file format: {file_ext}")
            return Response(
                {"error": f"Invalid file format. Supported formats: {', '.join(allowed_extensions)}"}, 
                status=status.HTTP_400_BAD_REQUEST
            )
            
        # Ensure Pinecone is initialized
        try:
            if pinecone_index is None:
                logger.info("Initializing Pinecone...")
                init_pinecone()
            logger.info("Pinecone connection verified")
        except Exception as e:
            logger.error(f"Pinecone initialization error: {str(e)}")
            return Response(
                {"error": f"Could not connect to vector database: {str(e)}"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
        
        # Process the audio file
        temp_file_path = None
        try:
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_audio_file:
                for chunk in audio_file.chunks():
                    temp_audio_file.write(chunk)
                temp_file_path = temp_audio_file.name
                
            logger.info(f"Audio saved to temporary file: {temp_file_path}")
            
            # Extract embedding - try a flexible approach
            try:
                # Try extracting from the middle of the audio to avoid silence
                embedding = extract_speaker_embedding(temp_file_path, 1.0, 6.0)  # Extract from 1s to 6s
                logger.info("Successfully extracted embedding from 1s-6s")
            except Exception as e1:
                logger.warning(f"Failed to extract embedding from time slice: {str(e1)}")
                try:
                    # Try with the beginning of the file if the middle didn't work
                    embedding = extract_speaker_embedding(temp_file_path, 0.0, 5.0)  # Extract from start to 5s
                    logger.info("Successfully extracted embedding from 0s-5s")
                except Exception as e2:
                    logger.error(f"Failed to extract embedding from beginning: {str(e2)}")
                    raise ValueError("Could not extract valid speaker embedding from audio. The file may be too short or contain no speech.")
            
            # Store the embedding
            logger.info(f"Storing embedding for speaker: {speaker_name}")
            success = store_speaker_embedding(speaker_name, embedding)
            
            if success:
                # Get updated index stats
                index_stats = pinecone_index.describe_index_stats()
                vector_count = index_stats['total_vector_count']
                
                logger.info(f"Successfully stored embedding. Total vectors: {vector_count}")
                return Response({
                    "status": "success",
                    "message": f"Successfully stored embedding for speaker '{speaker_name}'",
                    "total_embeddings": vector_count,
                    "dimension": len(embedding) if embedding else "unknown"
                })
            else:
                logger.error("Failed to store embedding")
                return Response({
                    "status": "error",
                    "message": "Failed to store embedding in the vector database"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"Error processing single speaker: {str(e)}", exc_info=True)
            return Response({
                "status": "error",
                "message": f"Error: {str(e)}"
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        finally:
            # Clean up
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Temporary file {temp_file_path} removed")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {str(e)}")
    
    except Exception as outer_e:
        logger.error(f"Unhandled exception in store_single_speaker: {str(outer_e)}", exc_info=True)
        return Response({
            "status": "error",
            "message": f"Server error: {str(outer_e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_pinecone_stats(request):
    """
    Get statistics about the Pinecone index, including vector count
    and dimension.
    
    Returns:
    - 200: Index statistics
    - 500: Error connecting to Pinecone
    """
    try:
        if pinecone_index is None:
            init_pinecone()
            
        # Get index statistics
        index_stats = pinecone_index.describe_index_stats()
        
        # Get index information using the new API
        indexes = pc_client.list_indexes().indexes() if hasattr(pc_client, 'list_indexes') else []
        index_info = next((idx for idx in indexes if idx.name == settings.PINECONE_INDEX_NAME), None)
        
        status_info = {
            "status": "success",
            "index_name": settings.PINECONE_INDEX_NAME,
            "total_vectors": index_stats['total_vector_count'],
            "dimension": 512,  # This should match PINECONE_DIMENSION in transcription_service.py
            "namespaces": index_stats.get('namespaces', {})
        }
        
        # Add additional info if available
        if index_info:
            status_info["hosting_type"] = getattr(index_info, 'hosting', 'serverless')
            status_info["status"] = getattr(index_info, 'status', 'unknown')
            
        return Response(status_info)
        
    except Exception as e:
        logger.error(f"Error getting Pinecone stats: {str(e)}", exc_info=True)
        return Response({
            "status": "error",
            "message": f"Failed to get Pinecone stats: {str(e)}"
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['POST'])
def test_voice_similarity(request):
    """
    Test endpoint to check if a voice matches any known speaker.
    
    Parameters:
    - audio: Audio file with speech to identify (required)
    
    Returns:
    - Matching speaker info if found
    """
    try:
        # Validate input parameters
        if 'audio' not in request.FILES:
            return Response({"error": "No audio file provided."}, status=status.HTTP_400_BAD_REQUEST)
            
        audio_file = request.FILES['audio']
        
        # Save to temporary file
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1].lower()) as temp_file:
                for chunk in audio_file.chunks():
                    temp_file.write(chunk)
                temp_file_path = temp_file.name
            
            # Extract embedding from the audio
            embedding = extract_speaker_embedding(temp_file_path, 1.0, 6.0)
            
            # Search for similar speakers in Pinecone
            matching_speaker = find_matching_speaker(embedding, threshold=0.6)
            
            if matching_speaker:
                return Response({
                    "status": "success",
                    "matched_speaker": matching_speaker,
                    "message": f"Voice matches speaker: {matching_speaker}"
                })
            else:
                return Response({
                    "status": "not_found",
                    "message": "No matching speaker found in database"
                })
                
        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error in test_voice_similarity: {str(e)}", exc_info=True)
        return Response({"status": "error", "message": f"Error: {str(e)}"}, 
                       status=status.HTTP_500_INTERNAL_SERVER_ERROR)

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import logging
from .utils import send_meeting_invites

logger = logging.getLogger(__name__)

@csrf_exempt
def send_meeting_email(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            meeting_title = data.get('meetingTitle', 'Untitled Meeting')
            meeting_date = data.get('date', 'N/A')
            start_time = data.get('startTime', 'N/A')
            end_time = data.get('endTime', 'N/A')
            agendas = data.get('agendas', [])
            participant_emails = data.get('participants', [])
            organization = data.get('organization', '')

            response = send_meeting_invites(meeting_title, meeting_date, start_time, end_time, agendas, participant_emails, organization)

            if "error" in response:
                return JsonResponse(response, status=400)

            return JsonResponse(response, status=200)

        except Exception as e:
            logger.error(f"Error in send_meeting_email: {e}")
            return JsonResponse({'error': f'An error occurred: {str(e)}'}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)
