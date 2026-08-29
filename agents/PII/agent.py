import base64
import io

from PIL import Image

from agents.PII.helpers import (
    PII_COVERAGE_SEVERITY,
    _get_presidio_engines,
    logger,
    redact_text_with_presidio,
)


def query_pii_agent(state) -> dict:
    query = state.get("query") or ""
    audio_transcript = state.get("raw_audio_transcript") or ""
    language_code = state.get("language_code") or "en"

    clean_query, query_coverage = redact_text_with_presidio(query, language_code)
    clean_audio_transcript, audio_coverage = redact_text_with_presidio(
        audio_transcript, language_code
    )

    worst_coverage = max(
        [query_coverage, audio_coverage], key=lambda c: PII_COVERAGE_SEVERITY[c]
    )

    return {
        "clean_query": clean_query,
        "clean_audio_transcript": clean_audio_transcript,
        "pii_language_used": language_code,
        "pii_coverage": worst_coverage,
    }


def image_pii_agent(state) -> dict:
    image = state.get("image_bytes")

    if not image:
        return {"image_bytes_cleaned": None}

    _, _, image_redactor = _get_presidio_engines()
    if image_redactor is None:
        # Fail closed: never forward an unredacted image to the vision LLM.
        return {
            "image_bytes_cleaned": None,
            "image_redaction_mode": "blocked_no_redactor",
        }

    try:
        image_data = base64.b64decode(image)
        pil_image = Image.open(io.BytesIO(image_data))

        red_result = image_redactor.redact(image=pil_image, fill="black")
        if red_result.mode != "RGB":
            # JPEG can't encode alpha; RGBA/P/etc inputs (e.g. PNG screenshots)
            # would otherwise raise here and fall into the fail-closed branch.
            red_result = red_result.convert("RGB")

        buffered = io.BytesIO()
        red_result.save(buffered, format="JPEG")
        clean_img_bytes_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "image_bytes_cleaned": clean_img_bytes_base64,
            "image_redaction_mode": "presidio_redacted"
        }
    except Exception as e:
        logger.error("Image PII redaction failed, blocking image: %s", e)
        return {
            "image_bytes_cleaned": None,
            "image_redaction_mode": "blocked_after_error",
        }
