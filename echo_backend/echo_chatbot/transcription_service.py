import os
import re
import tempfile
import torch
import whisper
from pyannote.audio import Pipeline
from django.conf import settings

class TranscriptionService:
    """Service for audio transcription and speaker diarization"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance is created"""
        if cls._instance is None:
            cls._instance = super(TranscriptionService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize only once"""
        if not TranscriptionService._initialized:
            # Configure environment
            if hasattr(settings, 'FFMPEG_PATH') and settings.FFMPEG_PATH:
                os.environ["PATH"] += os.pathsep + settings.FFMPEG_PATH
            
            # Set up device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"TranscriptionService initialized using device: {self.device}")
            
            # We don't initialize models until they're needed
            self._pipeline = None
            self._whisper_model = None
            
            # Patching torch.load on demand rather than globally
            self._original_torch_load = torch.load
            
            TranscriptionService._initialized = True
    
    def _patch_torch_load(self):
        """Apply patch to torch.load only when needed"""
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return self._original_torch_load(*args, **kwargs)
        
        torch.load = patched_torch_load
    
    def _restore_torch_load(self):
        """Restore original torch.load function"""
        torch.load = self._original_torch_load
    
    def _get_pipeline(self):
        """Lazy-load the diarization pipeline"""
        if self._pipeline is None:
            self._patch_torch_load()
            try:
                hugging_face_token = settings.HUGGING_FACE_TOKEN
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hugging_face_token
                ).to(self.device)
            finally:
                self._restore_torch_load()
        return self._pipeline
    
    def _get_whisper_model(self):
        """Lazy-load the whisper model"""
        if self._whisper_model is None:
            self._patch_torch_load()
            try:
                self._whisper_model = whisper.load_model("large-v2").to(self.device)
            finally:
                self._restore_torch_load()
        return self._whisper_model
    
    def clean_text(self, text):
        """Clean transcribed text by removing noise and non-English segments."""
        # Remove non-English characters (keeping basic punctuation)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def merge_consecutive_segments(self, segments, max_gap=2.0):
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

    def extract_name(self, introduction):
        """Extract speaker name from introduction text."""
        patterns = [
            r"(?:my name is|i am|this is|i'm|name's) ([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)",
            r"([A-Z][a-zA-Z]+(?:\s[A-Z][a-z]+)?)\s+(?:here|speaking)",
        ]
        
        introduction = self.clean_text(introduction.lower())
        
        for pattern in patterns:
            match = re.search(pattern, introduction, re.IGNORECASE)
            if match:
                return match.group(1).title().strip()
        return None

    def transcribe_segment(self, audio_file_path, start, end):
        """Transcribe a specific segment of audio."""
        whisper_model = self._get_whisper_model()
        
        audio = whisper.load_audio(audio_file_path)
        audio_segment = audio[int(start * 16000):int(end * 16000)]
        audio_tensor = torch.tensor(audio_segment).to(self.device)
        
        result = whisper_model.transcribe(
            audio_tensor,
            language="english",
            temperature=0.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=True
        )
        return self.clean_text(result["text"])

    def process_audio_file(self, audio_file_path):
        """Process an audio file and return the transcription with speaker diarization."""
        try:
            print("Diarization pipeline loading...")
            pipeline = self._get_pipeline()
            diarization = pipeline(audio_file_path)
            print("Diarization successful.")
            
            # First pass: collect segments and identify speakers
            segments = []
            speaker_names = {}
            introduction_phase = True
            min_segment_length = 1.0
            ignore_seconds = 2  # Ignore first 2 seconds
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start < ignore_seconds:
                    continue
                
                if turn.end - turn.start < min_segment_length:
                    continue
                
                text = self.transcribe_segment(audio_file_path, turn.start, turn.end)
                
                if not text:
                    continue
                
                if introduction_phase and "name is" in text.lower():
                    name = self.extract_name(text)
                    if name:
                        speaker_names[speaker] = name
                elif speaker not in speaker_names:
                    speaker_names[speaker] = f"Speaker {len(speaker_names) + 1}"
                
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker_names.get(speaker, f"Speaker {len(speaker_names) + 1}"),
                    'text': text
                })
                
                if introduction_phase and len(speaker_names) >= 3:
                    introduction_phase = False
            
            # Merge consecutive segments from the same speaker
            segments = self.merge_consecutive_segments(segments)
            
            # Format into list of objects
            transcript_output = []
            for segment in segments:
                transcript_output.append({
                    segment['speaker']: segment['text']
                })
            
            return transcript_output
            
        finally:
            # Clear CUDA cache after processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()