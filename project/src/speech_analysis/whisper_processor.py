"""
Whisper Integration for Speech-to-Text Processing
"""
import os
import tempfile
import whisper

class WhisperTranscriber:
    """Transcribe audio using OpenAI's Whisper model."""
    
    def __init__(self, model_name="base"):
        """Initialize the Whisper transcriber.
        
        Args:
            model_name: Whisper model to use (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        print(f"Loading Whisper {model_name} model...")
        self.model = whisper.load_model(model_name)
        print("Model loaded successfully!")
    
    def transcribe_file(self, audio_file_path):
        """Transcribe an audio file.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary with transcription results
        """
        result = self.model.transcribe(audio_file_path)
        return result
    
    def transcribe_audio_data(self, audio_data, sample_rate=16000):
        """Transcribe audio data.
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary with transcription results
        """
        # Create temporary file for audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save audio data to temporary file
            # This is a placeholder - in a real implementation you would save the audio data properly
            # For example with scipy.io.wavfile.write(temp_path, sample_rate, audio_data)
            
            # Transcribe the temporary file
            result = self.transcribe_file(temp_path)
            return result
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path) 