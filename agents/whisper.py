import os
from typing import Optional , Any
from loggers import setup_logger
from agents.helpers import tempfile_creator

logging = setup_logger(__name__)

# Small model: loads fast and downloads quickly, at some accuracy cost vs. larger
# models. Bump to "small"/"base" if accuracy on noisy/accented audio is insufficient.
DEFAULT_MODEL_SIZE = "tiny"


class Whisper:
    model: Optional[Any] = None
    model_size_or_path: str = DEFAULT_MODEL_SIZE
    device: str = "cpu"
    compute_type: str = "int8"

    def __init__(
        self,
        model_size_or_path: str = DEFAULT_MODEL_SIZE,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """Initializes and loads the Faster-Whisper model into memory."""
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type

        try:
            from faster_whisper import WhisperModel

            logging.info(f"Loading Whisper model: {self.model_size_or_path}...")
            self.model = WhisperModel(
                self.model_size_or_path,
                device=self.device,
                compute_type=self.compute_type,
            )
            logging.info("Whisper model loaded successfully!")
        except Exception as e:
            logging.error(f"Failed to load Whisper Model: {e}")
            raise ValueError(f"Whisper Model could not be loaded: {e}")

    def transcript(
        self,
        audio_bytes: bytes,
        audio_format: str,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> tuple[str, Any]:
        """Transcribes raw audio bytes using the pre-loaded Whisper model."""
        if self.model is None:
            raise RuntimeError("Whisper model is not initialized.")

        # Create temporary file from audio bytes
        temp_file, audio_path = tempfile_creator(
            audio_bytes=audio_bytes, audio_formate=audio_format
        )

        try:
            # Transcribe
            segments, info = self.model.transcribe(
                audio_path, beam_size=beam_size, vad_filter=vad_filter
            )

            # Generator to string conversion
            transcript_final = " ".join(
                segment.text.strip() for segment in segments
            )
            return transcript_final, info

        finally:
            # Ensure temporary file is cleaned up if tempfile_creator returns a file object
            if hasattr(temp_file, "close"):
                temp_file.close()
            try:
                os.remove(audio_path)
            except OSError:
                pass


stt_model = Whisper(model_size_or_path=DEFAULT_MODEL_SIZE,
                        device="cpu",
                        compute_type="int8")
