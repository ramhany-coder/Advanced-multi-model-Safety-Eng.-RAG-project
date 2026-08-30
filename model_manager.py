"""Local-directory-first model loading.

Every model this project uses (embedding, Whisper, the PII multilingual
transformer, ...) is meant to live in a known local directory. On startup we
check that directory first; the model is only downloaded from the Hugging
Face Hub the first time it's missing, so a later restart finds it already on
disk and never touches the network again.
"""
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

from loggers import setup_logger

logger = setup_logger(__name__)

# snapshot_download writes finished files straight into local_dir as it goes,
# so a directory with *some* files in it can just mean a prior download was
# interrupted partway through -- not that the model is complete. This marker
# is only written after a download finishes, so its presence is what actually
# means "safe to load offline, don't redownload."
_DOWNLOAD_COMPLETE_MARKER = ".download_complete"

# sentence-transformers repos ship extra onnx/openvino/tensorflow copies of the
# same weights for other runtimes; HuggingFaceEmbeddings only ever loads the
# plain PyTorch ones, so skip the rest to save bandwidth and disk space.
SENTENCE_TRANSFORMER_IGNORE_PATTERNS = ["onnx/*", "openvino/*", "*.msgpack", "*.h5", "*.ot"]


def is_model_present(local_dir: str | Path) -> bool:
    """A model directory counts as present only once a prior download of it
    has fully completed (see `_DOWNLOAD_COMPLETE_MARKER`)."""
    path = Path(local_dir)
    return (path / _DOWNLOAD_COMPLETE_MARKER).is_file()


def ensure_model_downloaded(
    repo_id: str,
    local_dir: str | Path,
    allow_patterns: Optional[list[str]] = None,
    ignore_patterns: Optional[list[str]] = None,
) -> str:
    """Ensure a Hugging Face Hub model is available at `local_dir`.

    Downloads it with `snapshot_download` only the first time; every later
    call (including across app restarts) finds the directory already
    populated and returns immediately without contacting the network.
    """
    path = Path(local_dir)

    if is_model_present(path):
        logger.info("Model '%s' already present at %s, skipping download", repo_id, path)
        return str(path)

    logger.info("Model '%s' not found at %s, downloading...", repo_id, path)
    path.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(path),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    (path / _DOWNLOAD_COMPLETE_MARKER).touch()
    logger.info("Model '%s' downloaded to %s", repo_id, path)
    return str(path)


def ensure_whisper_model(model_size_or_id: str, local_dir: str | Path) -> str:
    """Ensure a faster-whisper (CTranslate2) model is available at `local_dir`.

    Uses faster_whisper's own downloader (it knows the right repo naming and
    file allow-list for CTranslate2 Whisper models), but only the first time --
    a `local_dir` that's already populated is returned as-is with no network
    call, and can be passed straight to `WhisperModel(...)`.
    """
    path = Path(local_dir)

    if is_model_present(path):
        logger.info(
            "Whisper model '%s' already present at %s, skipping download",
            model_size_or_id,
            path,
        )
        return str(path)

    logger.info("Whisper model '%s' not found at %s, downloading...", model_size_or_id, path)
    from faster_whisper.utils import download_model

    path.mkdir(parents=True, exist_ok=True)
    download_model(model_size_or_id, output_dir=str(path))
    (path / _DOWNLOAD_COMPLETE_MARKER).touch()
    logger.info("Whisper model '%s' downloaded to %s", model_size_or_id, path)
    return str(path)
