import os
from types import SimpleNamespace
from typing import Optional , Any
from loggers import setup_logger
from agents.helpers import tempfile_creator
from config import settings
from model_manager import ensure_whisper_model

logging = setup_logger(__name__)

# "small" trades some load time/memory for noticeably better accuracy than "tiny",
# especially on Arabic and noisy/accented audio.
DEFAULT_MODEL_SIZE = "small"

# Used only as a fallback when the local model fails to load (e.g. no network/disk
# access to fetch model weights in a restricted cloud environment).
CLOUD_MODEL = "whisper-1"


class Whisper:
    model: Optional[Any] = None
    model_size_or_path: str = DEFAULT_MODEL_SIZE
    device: str = "cpu"
    compute_type: str = "int8"
    backend: str = "local"

    def __init__(
        self,
        model_size_or_path: str = DEFAULT_MODEL_SIZE,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        """Loads the local Faster-Whisper model. Falls back to the OpenAI Whisper
        API if the local model fails to load, instead of crashing the app."""
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type

        try:
            from faster_whisper import WhisperModel

            # Downloads to settings.WHISPER_MODEL_PATH only the first time; a
            # later restart finds the weights already on disk and loads
            # fully offline, without touching the network.
            local_model_path = ensure_whisper_model(
                self.model_size_or_path, settings.WHISPER_MODEL_PATH
            )

            logging.info(f"Loading Whisper model: {self.model_size_or_path}...")
            self.model = WhisperModel(
                local_model_path,
                device=self.device,
                compute_type=self.compute_type,
            )
            self.backend = "local"
            logging.info("Whisper model loaded successfully!")
        except Exception as e:
            logging.error(
                f"Failed to load local Whisper model, falling back to the "
                f"OpenAI Whisper API ({CLOUD_MODEL}): {e}"
            )
            from openai import OpenAI

            self.model = None
            self.backend = "cloud"
            self._openai_client = OpenAI(api_key=settings.GPT_API)

    def transcript(
        self,
        audio_bytes: bytes,
        audio_format: str,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> tuple[str, Any]:
        """Transcribes raw audio bytes using whichever backend loaded successfully."""
        # Create temporary file from audio bytes
        temp_file, audio_path = tempfile_creator(
            audio_bytes=audio_bytes, audio_formate=audio_format
        )

        try:
            if self.backend == "cloud":
                with open(audio_path, "rb") as audio_file:
                    response = self._openai_client.audio.transcriptions.create(
                        model=CLOUD_MODEL,
                        file=audio_file,
                        response_format="verbose_json",
                    )
                return response.text.strip(), SimpleNamespace(language=response.language)

            # Transcribe locally
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


class _LazyWhisper:
    """Defers constructing (downloading/loading) the real Whisper model until
    the first actual transcription call, instead of at import time -- so
    every process that imports this module (any agent import chain) doesn't
    pull a few hundred MB of Whisper weights into memory whether or not audio
    is ever used in that process."""

    _instance: Optional["Whisper"] = None

    def _get(self) -> "Whisper":
        if self._instance is None:
            self._instance = Whisper(
                model_size_or_path=settings.WHISPER_MODEL_SIZE or DEFAULT_MODEL_SIZE,
                device="cpu",
                compute_type="int8",
            )
        return self._instance

    def __getattr__(self, name):
        return getattr(self._get(), name)


stt_model = _LazyWhisper()
