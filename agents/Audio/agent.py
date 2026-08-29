import base64

from agents.Audio.whisper.whisper import stt_model


def audio_transcription_agent(state) -> dict:
    audio_b64 = state.get('audio_bytes', "")
    audio_format = state.get('audio_format', "")

    if not audio_b64:
        return {
            "raw_audio_transcript": "",
            "audio_transcription_error": "No audio provided.",
        }

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception as e:
        return {
            "raw_audio_transcript": "",
            "audio_transcription_error": f"Could not decode audio: {e}",
        }

    transcript_final, info = stt_model.transcript(
        audio_bytes=audio_bytes,
        audio_format=audio_format
    )
    return {
        "raw_audio_transcript": transcript_final.strip(),
        "detected_voice_language": info.language
    }
