import re
import torch
from pyannote.audio import Pipeline
import whisper
import os
from django.conf import settings
import numpy as np
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from pydub import AudioSegment
import tempfile
import logging



# Ensure FFMPEG_DIR exists before adding it to PATH
FFMPEG_DIR = getattr(settings, "FFMPEG_DIR", None)
if FFMPEG_DIR:
    os.environ["PATH"] += os.pathsep + FFMPEG_DIR
else:
    print("Warning: FFMPEG_DIR is not set in settings.py")

# Check if CUDA is available and create proper torch device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the pipeline and whisper model at the module level
pipeline = Pipeline.from_pretrained(
    settings.DIARIZATION_MODEL,
    use_auth_token=settings.HUGGING_FACE_TOKEN, 
).to(device)


whisper_model = whisper.load_model(settings.WHISPER_MODEL).to(device)

load_dotenv() 


# Global Pinecone client and index
pc_client = None
pinecone_index = None

def init_pinecone():
    """Initialize Pinecone client and create index if it doesn't exist"""
    global pc_client, pinecone_index
    
    try:
        pc_client = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        existing_indexes = pc_client.list_indexes().names()
        
        if settings.PINECONE_INDEX_NAME not in existing_indexes:
            print(f"Creating new Pinecone index: {settings.PINECONE_INDEX_NAME}")
            pc_client.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=settings.PINECONE_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.PINECONE_CLOUD,
                    region=settings.PINECONE_REGION
                )
            )
            time.sleep(10)
            print(f"Index {settings.PINECONE_INDEX_NAME} created successfully")
        
        pinecone_index = pc_client.Index(settings.PINECONE_INDEX_NAME)
        print(f"Connected to Pinecone index: {settings.PINECONE_INDEX_NAME}")
        return pinecone_index
    
    except Exception as e:
        print(f"Error initializing Pinecone: {str(e)}")
        raise

# Initialize Pinecone once
try:
    pinecone_index = init_pinecone()
except Exception as e:
    print(f"Warning: Could not initialize Pinecone. Error: {str(e)}")
    pinecone_index = None


def clean_text(text):
    """Clean transcribed text by removing noise and non-English segments."""
    # Remove non-English characters (keeping basic punctuation)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def merge_consecutive_segments(segments, max_gap=settings.MAX_GAP_BETWEEN_SEGMENTS):
    """Merge consecutive segments from the same speaker if they're close in time"""
    if not segments:
        return segments
    
    merged = []
    current = segments[0]
    
    for next_segment in segments[1:]:
        if (current['speaker'] == next_segment['speaker'] and 
            next_segment['start'] - current['end'] <= max_gap):
            # Merge segments
            current['text'] += ' ' + next_segment['text']
            current['end'] = next_segment['end']
        else:
            merged.append(current)
            current = next_segment
    
    merged.append(current)
    return merged

def transcribe_segment(audio_file_path, start, end):
    """Transcribe a specific segment of the audio file."""
    audio = whisper.load_audio(audio_file_path)
    audio_segment = audio[int(start * 16000):int(end * 16000)]
    audio_tensor = torch.tensor(audio_segment).to(device)
    
    result = whisper_model.transcribe(
        audio_tensor,
        language="english",
        temperature=0.0,
        no_speech_threshold=0.3,
        condition_on_previous_text=True
    )
    return clean_text(result["text"])

def extract_name(introduction):
    """Extract speaker name from an introduction phrase."""
    patterns = [
        r"(?:my name is|i am|this is|i'm|name's) ([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)",
        r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-Z]+)?)\s+(?:here|speaking)",
    ]
    
    introduction = clean_text(introduction.lower())
    
    for pattern in patterns:
        match = re.search(pattern, introduction, re.IGNORECASE)
        if match:
            return match.group(1).title().strip()
    return None


def extract_speaker_embedding(audio_file_path, start, end):
    """Extract speaker embedding for a specific segment."""
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
    
    # Load pretrained model for speaker embedding extraction - use PyAnnote model for 512 dimensions
    embedding_model = PretrainedSpeakerEmbedding(
        "pyannote/embedding",  # This model produces 512-dimensional embeddings
        use_auth_token=settings.HUGGING_FACE_TOKEN
    ).to(device)
    
    audio = whisper.load_audio(audio_file_path)
    # Ensure audio is long enough
    if int(end * 16000) > len(audio):
        end = len(audio) / 16000  # Convert to seconds
        print(f"Adjusted end time to {end:.2f}s due to audio length")
    
    audio_segment = audio[int(start * 16000):int(end * 16000)]
    
    # Ensure we have enough audio data to extract an embedding
    # min_samples = 16000 * 1  
    min_samples = int(16000 * 0.5)
    if len(audio_segment) < min_samples:
        print(f"Warning: Audio segment too short ({len(audio_segment)/16000:.2f}s), minimum 2s required")
        if len(audio) >= min_samples:
            # Use the first few seconds instead
            audio_segment = audio[:min_samples]
            print("Using first 2 seconds of audio instead")
        else:
            raise ValueError(f"Audio file too short: {len(audio)/16000:.2f}s, minimum 2s required")
    
    # Convert to mono if needed and ensure proper shape
    if len(audio_segment.shape) == 1:
        audio_segment = np.expand_dims(audio_segment, axis=0)  # Add channel dimension
    
    audio_tensor = torch.tensor(audio_segment).float().to(device)
    
    # Ensure proper shape: (channels, samples)
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Extract embedding - the model expects (batch, channel, samples)
    with torch.no_grad():
        embedding = embedding_model(audio_tensor.unsqueeze(0))  # Add batch dimension
    
    # Convert to numpy array if it's a tensor
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.squeeze(0).cpu().numpy()
    
    # Normalize the embedding
    embedding = embedding / np.linalg.norm(embedding)
    
    # Verify the dimensions match what we expect
    if embedding.size != 512:
        print(f"Warning: Embedding dimension is {embedding.size}, expected 512")
        if embedding.size < 512:
            # Pad with zeros if needed
            padding = np.zeros(512 - embedding.size)
            embedding = np.concatenate([embedding, padding])
            print("Padded embedding to 512 dimensions")
        else:
            # Truncate if too large
            embedding = embedding[:512]
            print("Truncated embedding to 512 dimensions")
    
    # Return as list for database storage
    return embedding.tolist()

def store_speaker_embedding(name, embedding):
    """Store speaker embedding in Pinecone."""
    try:
        # Fix for nested embeddings - flatten if necessary
        if len(embedding) == 1 and isinstance(embedding[0], list):
            embedding = embedding[0]  # Take the first embedding
            print(f"Flattened nested embedding, new length: {len(embedding)}")
        
        # Check embedding dimensionality
        if len(embedding) != settings.PINECONE_DIMENSION:
            print(f"Embedding dimension mismatch: got {len(embedding)}, expected {settings.PINECONE_DIMENSION}")
            
            # Handle dimension mismatch
            if len(embedding) < settings.PINECONE_DIMENSION:
                # Pad with zeros
                padding = [0.0] * (settings.PINECONE_DIMENSION - len(embedding))
                embedding = embedding + padding
                print(f"Padded embedding to {len(embedding)} dimensions")
            else:
                # Truncate
                embedding = embedding[:settings.PINECONE_DIMENSION]
                print(f"Truncated embedding to {len(embedding)} dimensions")
        
        # Create a unique ID for this embedding
        import uuid
        vector_id = f"{name.replace(' ', '_')}_{uuid.uuid4()}"
        
        # Upsert the vector to Pinecone
        pinecone_index.upsert(
            vectors=[{
                "id": vector_id,
                "values": embedding,
                "metadata": {"speaker_name": name, "timestamp": time.time()}
            }]
        )
        
        print(f"Stored embedding for {name} in Pinecone (vector_id: {vector_id}, dimension: {len(embedding)})")
        return True
        
    except Exception as e:
        print(f"Failed to store embedding for {name} in Pinecone: {str(e)}")
        print(f"Problematic embedding sample: {embedding[:5] if isinstance(embedding, list) else embedding}")
        return False


def find_matching_speaker(embedding, threshold=0.5): #lower threshold if needed
    """Find matching speaker from stored embeddings using Pinecone."""
    try:
        # Query for closest embedding using cosine similarity
        results = pinecone_index.query(
            vector=embedding,
            top_k=1,
            include_metadata=True
        )
        
        # Check if any matches were found
        if results['matches'] and len(results['matches']) > 0:
            top_match = results['matches'][0]
            similarity = top_match['score']
            
            if similarity > threshold:
                name = top_match['metadata']['speaker_name']
                print(f"Found matching speaker: {name} with similarity: {similarity:.4f}")
                return name
        
        return None
        
    except Exception as e:
        print(f"Error finding matching speaker in Pinecone: {str(e)}")
        return None

def transcribe_audio_file(audio_file_path):
    """Process audio file to identify speakers and transcribe their speech."""
    try:
        print("Diarization pipeline loading...")
        diarization = pipeline(audio_file_path)
        print("Diarization successful.")
        
        print("All raw segments from diarization:")
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            print(f"Raw segment: {segment.start:.2f} to {segment.end:.2f}, speaker: {speaker}")
        # First pass: collect segments and identify speakers
        segments = []
        speaker_names = {}
        speaker_embeddings = {}
        introduction_phase = True
        min_segment_length = settings.MIN_SEGMENT_LENGTH
        ignore_seconds = settings.IGNORE_FIRST_SECONDS
        INTRODUCTION_PHASE_MAX_TIME = 20

        # Debug the output structure of diarization
        first_track = next(diarization.itertracks(yield_label=True), None)
        if first_track:
            print(f"Sample track: {first_track}")
            print(f"Type: {type(first_track)}")
        
        for segment, track, speaker in diarization.itertracks(yield_label=True):
            try:
                # End introduction phase based on the time
                if segment.start > INTRODUCTION_PHASE_MAX_TIME:
                    introduction_phase = False

                # Skip segments that are too short or in the ignored initial period
                if segment.start < ignore_seconds:
                    continue

                # if segment.end - segment.start < min_segment_length:
                #     continue
                # To this
                if segment.end - segment.start + 0.001 < min_segment_length:  # Add a small epsilon
                    continue

                print(f"Processing segment: {segment.start:.2f} to {segment.end:.2f}, speaker: {speaker}")
                
                # Transcribe the segment
                text = transcribe_segment(audio_file_path, segment.start, segment.end)
                
                if not text:
                    print(f"No text transcribed for segment: {segment.start:.2f} to {segment.end:.2f}")
                    continue
                
                print(f"Transcribed text: {text[:50]}...")
                
                try:
                    # Extract speaker embedding with proper error handling
                    embedding = extract_speaker_embedding(audio_file_path, segment.start, segment.end)
                    
                    # Store embedding with speaker ID for later processing
                    if speaker not in speaker_embeddings:
                        speaker_embeddings[speaker] = []
                    speaker_embeddings[speaker].append(embedding)
                except Exception as e:
                    print(f"Failed to extract embedding for segment {segment}: {str(e)}")
                    embedding = None

                # Extract name during introduction phase 
                if introduction_phase:
                    name = extract_name(text)
                    if name:
                        speaker_names[speaker] = name
                        print(f"Extracted name: {name} for speaker {speaker}")
                
                # If no name extracted from introduction, try to find a match in stored embeddings
                if speaker not in speaker_names:
                    if embedding is not None:
                        matching_name = find_matching_speaker(embedding)
                        if matching_name:
                            speaker_names[speaker] = matching_name
                            print(f"Found matching speaker: {matching_name}")
                    if speaker not in speaker_names:
                        speaker_names[speaker] = f"Speaker {len(speaker_names) + 1}"
                        print(f"Assigned default name: {speaker_names[speaker]}")

                segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'speaker': speaker_names.get(speaker, f"Speaker {len(speaker_names) + 1}"),
                    'text': text
                })

                if introduction_phase and (len(speaker_names) >= 3 or segment.end > 60):
                    introduction_phase = False
                    
            except Exception as e:
                import traceback
                print(f"Error processing segment {segment}: {e}")
                print(traceback.format_exc())
                continue  # Skip any problematic segments

        print(f"Total segments processed: {len(segments)}")
        
        # Store embeddings for named speakers
        for speaker, name in speaker_names.items():
            if not name.startswith("Speaker "):  # Only store named speakers
                for embedding in speaker_embeddings.get(speaker, []):
                    try:
                        store_speaker_embedding(name, embedding)
                        print(f"Stored embedding for {name}")
                    except Exception as e:
                        print(f"Failed to store embedding for {name}: {str(e)}")

        # Merge consecutive segments from the same speaker
        segments = merge_consecutive_segments(segments)
        print(f"After merging: {len(segments)} segments")

        # Format into list of objects
        transcript_output = []
        for segment in segments:
            transcript_output.append({
                segment['speaker']: segment['text']
            })

        # Clear CUDA cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return transcript_output

    except Exception as e:
        import traceback
        print(f"Error during transcription: {str(e)}")
        print(traceback.format_exc())
        raise e
    


logger = logging.getLogger(__name__)

def convert_to_wav(input_file, output_file=None, sample_rate=16000):
    """
    Convert any audio file to WAV format (16kHz, 16-bit, mono).
    
    Parameters:
    - input_file: Path to input audio file or file-like object
    - output_file: Optional path for output WAV file. If None, creates a temporary file.
    - sample_rate: Sample rate for output WAV file (default: 16000 Hz)
    
    Returns:
    - Path to the converted WAV file
    """
    try:
        logger.info(f"Converting audio file to WAV format")
        
        # Determine file extension from input
        if isinstance(input_file, str):
            file_ext = os.path.splitext(input_file)[1].lower()
        else:
            # For file-like objects or Django UploadedFile
            file_ext = os.path.splitext(input_file.name)[1].lower()
            
            # Save to a temporary file if it's a file-like object
            if not isinstance(input_file, str):
                temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                
                # Handle Django UploadedFile with chunks
                if hasattr(input_file, 'chunks'):
                    for chunk in input_file.chunks():
                        temp_input.write(chunk)
                else:
                    # Handle regular file-like objects
                    input_file.seek(0)
                    temp_input.write(input_file.read())
                    
                temp_input.close()
                input_file = temp_input.name
                logger.info(f"Saved uploaded file to temporary location: {input_file}")
        
        # Select the appropriate format based on file extension
        format_dict = {
            '.mp3': 'mp3',
            '.wav': 'wav',
            '.ogg': 'ogg',
            '.flac': 'flac',
            '.m4a': 'm4a',
            '.aac': 'aac',
            '.wma': 'wma',
            '.aiff': 'aiff',
            '.alac': 'alac',
            '.opus': 'opus',
            '.webm': 'webm',
            '.mp4': 'mp4'
        }
        
        input_format = format_dict.get(file_ext, None)
        if not input_format:
            raise ValueError(f"Unsupported audio format: {file_ext}")
        
        # Load the audio file
        audio = AudioSegment.from_file(input_file, format=input_format)
        
        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
            logger.info("Converted audio to mono")
        
        # Set the sample rate
        if audio.frame_rate != sample_rate:
            audio = audio.set_frame_rate(sample_rate)
            logger.info(f"Set sample rate to {sample_rate}Hz")
        
        # Export as WAV
        if not output_file:
            output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
        
        audio.export(output_file, format='wav')
        logger.info(f"Successfully converted audio to WAV: {output_file}")
        
        # Clean up temp input file if we created one
        if isinstance(input_file, str) and input_file.startswith(tempfile.gettempdir()):
            try:
                os.remove(input_file)
                logger.info(f"Removed temporary input file: {input_file}")
            except Exception as e:
                logger.warning(f"Could not remove temporary input file: {str(e)}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Error converting audio to WAV: {str(e)}", exc_info=True)
        raise
