"""Integration tests for the voice STT pipeline (agents/whisper.py, agents/Audio/agent.py).

Runs the actual configured STT backend (local Faster-Whisper by default, OpenAI
Whisper API on load-failure fallback) against real English and Arabic speech
fixtures, plus the audio_transcription_agent node's error handling. Meant to run
in CI to catch pipeline regressions (broken audio decoding, backend wiring,
language detection). Assertions use a similarity ratio rather than exact-match,
since STT output varies slightly across model versions/hardware.
"""
import base64
import difflib
import json
from pathlib import Path

import pytest

from agents.Audio.whisper.whisper import stt_model
from agents.Audio.agent import audio_transcription_agent

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "audio"
REFERENCE = json.loads((FIXTURES_DIR / "reference.json").read_text(encoding="utf-8"))


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _load_audio(name: str) -> bytes:
    return (FIXTURES_DIR / name).read_bytes()


@pytest.mark.parametrize("fixture_name", REFERENCE.keys())
def test_stt_model_transcribes_sample(fixture_name):
    """The configured stt_model backend produces a close-enough transcript."""
    ref = REFERENCE[fixture_name]
    audio_bytes = _load_audio(fixture_name)

    transcript, info = stt_model.transcript(audio_bytes=audio_bytes, audio_format="mp3")

    assert transcript, f"{fixture_name}: got an empty transcript"
    assert info.language.lower().startswith(ref["lang_prefix"]), (
        f"{fixture_name}: expected language starting with '{ref['lang_prefix']}', got '{info.language}'"
    )
    ratio = _similarity(ref["text"], transcript)
    assert ratio >= ref["min_similarity"], (
        f"{fixture_name}: transcript too far from reference (similarity={ratio:.2f}).\n"
        f"Reference: {ref['text']!r}\nGot:       {transcript!r}"
    )


@pytest.mark.parametrize("fixture_name", REFERENCE.keys())
def test_audio_transcription_agent_end_to_end(fixture_name):
    """The LangGraph node wraps the STT backend correctly end to end (base64 in, state out)."""
    ref = REFERENCE[fixture_name]
    audio_b64 = base64.b64encode(_load_audio(fixture_name)).decode("ascii")

    result = audio_transcription_agent({"audio_bytes": audio_b64, "audio_format": "mp3"})

    assert result.get("audio_transcription_error") is None
    assert result["raw_audio_transcript"], f"{fixture_name}: empty raw_audio_transcript"
    assert result["detected_voice_language"].lower().startswith(ref["lang_prefix"])


def test_audio_transcription_agent_no_audio():
    result = audio_transcription_agent({"audio_bytes": "", "audio_format": "mp3"})
    assert result["raw_audio_transcript"] == ""
    assert "No audio provided" in result["audio_transcription_error"]


def test_audio_transcription_agent_invalid_base64():
    result = audio_transcription_agent({"audio_bytes": "not-valid-base64!!", "audio_format": "mp3"})
    assert result["raw_audio_transcript"] == ""
    assert "Could not decode audio" in result["audio_transcription_error"]
