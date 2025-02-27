import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')

import os
import torch
# Suppress Flash Attention 2 warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
# Suppress HF text generation warnings
os.environ["HF_SUPPRESS_GENERATION_WARNINGS"] = "true"

import time
import yaml
import re
import shutil
from typing import Dict, List, Optional, Union, Tuple
from pydub import AudioSegment
import tempfile
import tqdm

def _init_bark(self) -> None:
        """Initialize the Bark TTS model."""
        print("Initializing Bark TTS model...")
        from transformers import AutoProcessor, BarkModel
        
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained("suno/bark")
        self.model = BarkModel.from_pretrained("suno/bark")
        
        # Set pad token ID to avoid warnings
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            print("Bark model loaded on GPU")
        else:
            print("Bark model loaded on CPU")
        
        # Define speaker presets
        self.speakers = {
            "Speaker 1": "v2/en_speaker_6",  # Male expert
            "Speaker 2": "v2/en_speaker_9",  # Female student
            "Speaker 3": "v2/en_speaker_3"   # Second expert
        }

def _init_coqui(self) -> None:
        """Initialize the Coqui TTS model."""
        print("Initializing Coqui TTS model...")
        try:
            from TTS.api import TTS
            
            # Initialize Coqui TTS with a multi-speaker model
            # Using VITS model which supports multi-speaker synthesis
            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            
            # Define speaker presets (speaker names for Coqui XTTS)
            self.speakers = {
                "Speaker 1": "p326",  # Male expert
                "Speaker 2": "p225",  # Female student
                "Speaker 3": "p330"   # Second expert
            }
            self.coqui_available = True
            
            # Store sample rate for later use
            self.sample_rate = 24000  # Default for XTTS
            
        except ImportError:
            print("WARNING: Coqui TTS module not found. Using fallback TTS instead.")
            print("To install Coqui TTS, run: pip install TTS")
            # Fall back to Bark if Coqui is not available
            self.model_type = "bark"
            self._init_bark()
            self.coqui_available = False
        except Exception as e:
            print(f"WARNING: Error initializing Coqui TTS: {str(e)}. Using fallback TTS instead.")
            # Fall back to Bark if there's an error with Coqui
            self.model_type = "bark"
            self._init_bark()
            self.coqui_available = False

def _generate_audio_bark(self, text: str, speaker: str) -> AudioSegment:
        """Generate audio using Bark TTS.
        
        Args:
            text: Text to convert to speech
            speaker: Speaker identifier
            
        Returns:
            AudioSegment containing the generated speech
        """
        try:
            # Prepare inputs
            inputs = self.processor(
                text=text,
                voice_preset=self.speakers[speaker],
                return_tensors="pt"
            )
            
            # Create attention mask if not present
            if "attention_mask" not in inputs:
                # Create attention mask (all 1s, same shape as input_ids)
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            
            # Move inputs to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate audio with specific generation parameters
            # Use only max_new_tokens, not max_length to avoid the warning
            speech_output = self.model.generate(
                **inputs,
                pad_token_id=self.model.config.pad_token_id,
                do_sample=True,
                temperature=0.7,
                max_new_tokens=250  # Use max_new_tokens instead of max_length
            )
            
            # Convert to audio segment
            audio_array = speech_output.cpu().numpy().squeeze()
            
            # Save to temporary file and load as AudioSegment
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Save as WAV
            import scipy.io.wavfile as wavfile
            wavfile.write(temp_path, rate=24000, data=audio_array)
            
            # Load as AudioSegment
            if not self.ffmpeg_available:
                print("WARNING: FFmpeg not available. Using silent audio as fallback.")
                audio_segment = AudioSegment.silent(duration=len(audio_array) * 1000 // 24000)
            else:
                try:
                    audio_segment = AudioSegment.from_wav(temp_path)
                except Exception as e:
                    print(f"Error loading audio segment: {str(e)}")
                    # Fallback to silent audio
                    audio_segment = AudioSegment.silent(duration=len(audio_array) * 1000 // 24000)
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_path}: {str(e)}")
            
            return audio_segment
        except Exception as e:
            print(f"Error in _generate_audio_bark: {str(e)}")
            # Return a silent segment as fallback
            return AudioSegment.silent(duration=1000)

def _generate_audio_coqui(self, text: str, speaker: str) -> AudioSegment:
        """Generate audio using Coqui TTS.
        
        Args:
            text: Text to convert to speech
            speaker: Speaker identifier
            
        Returns:
            AudioSegment containing the generated speech
        """
        try:
            # Create a temporary file to save the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate audio with Coqui TTS
            # For XTTS, we need to provide a reference audio file for the speaker
            # Since we don't have that, we'll use the built-in speaker IDs
            self.model.tts_to_file(
                text=text,
                file_path=temp_path,
                speaker=self.speakers[speaker],
                language="en"
            )
            
            # Load as AudioSegment
            if not self.ffmpeg_available:
                print("WARNING: FFmpeg not available. Using silent audio as fallback.")
                # Estimate duration based on text length (rough approximation)
                estimated_duration = len(text) * 60  # ~60ms per character
                audio_segment = AudioSegment.silent(duration=estimated_duration)
            else:
                try:
                    audio_segment = AudioSegment.from_wav(temp_path)
                except Exception as e:
                    print(f"Error loading audio segment: {str(e)}")
                    # Fallback to silent audio
                    estimated_duration = len(text) * 60  # ~60ms per character
                    audio_segment = AudioSegment.silent(duration=estimated_duration)
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_path}: {str(e)}")
            
            return audio_segment
        except Exception as e:
            print(f"Error generating audio with Coqui: {str(e)}")
            # Return a silent segment as fallback
            return AudioSegment.silent(duration=1000) 