import re
import threading
import queue

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

try:
    from presidio_analyzer.nlp_engine import TransformersNlpEngine
except Exception:
    TransformersNlpEngine = None

try:
    from presidio_image_redactor import ImageRedactorEngine
except Exception:
    ImageRedactorEngine = None

from loggers import setup_logger

logger = setup_logger(__name__)

# Presidio's AnalyzerEngine() loads a spaCy NLP model on construction, which can
# take a long time (or hang outright if the model/network dependency isn't
# available). Building it at import time would block the whole process before
# any request is served, so it's built lazily in a background thread the first
# time PII redaction is actually needed, with a bounded wait per call.
_PRESIDIO_INIT_TIMEOUT_SECONDS = 15
_presidio_result_queue: "queue.Queue" = queue.Queue(maxsize=1)
_presidio_init_started = threading.Event()
_presidio_engines = None  # (analyzer, anonymizer, image_redactor) once resolved


def _init_presidio_engines_worker() -> None:
    try:
        eng_analyzer = AnalyzerEngine()
        eng_anonymizer = AnonymizerEngine()
        eng_image = ImageRedactorEngine() if ImageRedactorEngine is not None else None
        _presidio_result_queue.put((eng_analyzer, eng_anonymizer, eng_image))
    except Exception as e:
        logger.error("Presidio engine initialization failed: %s", e)
        _presidio_result_queue.put((None, None, None))


def _get_presidio_engines():
    """Return (analyzer, anonymizer, image_redactor), starting lazy init on
    first call. Waits up to _PRESIDIO_INIT_TIMEOUT_SECONDS; if init hasn't
    finished yet, returns (None, None, None) for this call without giving up
    permanently, so a slow-but-eventually-successful load still gets used on
    a later call."""
    global _presidio_engines
    if _presidio_engines is not None:
        return _presidio_engines

    if not _presidio_init_started.is_set():
        _presidio_init_started.set()
        threading.Thread(target=_init_presidio_engines_worker, daemon=True).start()

    try:
        _presidio_engines = _presidio_result_queue.get(timeout=_PRESIDIO_INIT_TIMEOUT_SECONDS)
    except queue.Empty:
        logger.error(
            "Presidio engine initialization still not ready after %ss; "
            "falling back to regex-only redaction for this call.",
            _PRESIDIO_INIT_TIMEOUT_SECONDS,
        )
        return None, None, None

    return _presidio_engines


# Real NER-based PII coverage (names, locations, ...) for non-English text, via
# a multilingual transformer model. Deliberately scoped to Arabic only, not the
# full set of languages `language_detector` recognizes (ar/fr/es/de): loading
# more than one large transformer model copy at once was verified to segfault
# under memory pressure on an 8GB-RAM host, whereas a single extra model
# alongside the English engine was verified stable. Extend
# _MULTILINGUAL_LANGUAGES only on a host with headroom to match.
_MULTILINGUAL_TRANSFORMER_MODEL = "Davlan/xlm-roberta-base-ner-hrl"
_MULTILINGUAL_TOKENIZER_MODEL = "xx_ent_wiki_sm"
_MULTILINGUAL_LANGUAGES = ["ar"]
_MULTILINGUAL_INIT_TIMEOUT_SECONDS = 30
_multilingual_result_queue: "queue.Queue" = queue.Queue(maxsize=1)
_multilingual_init_started = threading.Event()
_multilingual_engine = None  # analyzer (or None) once resolved


def _init_multilingual_engine_worker() -> None:
    try:
        if TransformersNlpEngine is None:
            raise RuntimeError("spacy-huggingface-pipelines is not installed")
        models = [
            {
                "lang_code": lang,
                "model_name": {
                    "spacy": _MULTILINGUAL_TOKENIZER_MODEL,
                    "transformers": _MULTILINGUAL_TRANSFORMER_MODEL,
                },
            }
            for lang in _MULTILINGUAL_LANGUAGES
        ]
        nlp_engine = TransformersNlpEngine(models=models)
        eng_analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, supported_languages=_MULTILINGUAL_LANGUAGES
        )
        _multilingual_result_queue.put(eng_analyzer)
    except Exception as e:
        logger.error("Multilingual PII engine initialization failed: %s", e)
        _multilingual_result_queue.put(None)


def _get_multilingual_engine():
    """Return the multilingual analyzer (or None), lazily built the same way
    as _get_presidio_engines. Reuses the English pipeline's AnonymizerEngine --
    anonymization is language-agnostic, it only needs the analyzer results."""
    global _multilingual_engine
    if _multilingual_engine is not None:
        return _multilingual_engine

    if not _multilingual_init_started.is_set():
        _multilingual_init_started.set()
        threading.Thread(target=_init_multilingual_engine_worker, daemon=True).start()

    try:
        _multilingual_engine = _multilingual_result_queue.get(
            timeout=_MULTILINGUAL_INIT_TIMEOUT_SECONDS
        )
    except queue.Empty:
        logger.error(
            "Multilingual PII engine still not ready after %ss; "
            "falling back to pattern-only redaction for this call.",
            _MULTILINGUAL_INIT_TIMEOUT_SECONDS,
        )
        return None

    return _multilingual_engine


# Presidio is only configured with an English NLP model in this project, so its
# NER-based recognizers (PERSON, LOCATION, ...) only run reliably on English
# text. These patterns are a conservative, language-agnostic safety net used
# whenever the real engines are unavailable or fail outright, so a redaction
# failure never means "send the original text through unredacted."
_FALLBACK_PII_PATTERNS = [
    ("EMAIL_ADDRESS", re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")),
    ("IBAN_CODE", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE_NUMBER", re.compile(r"\+?\d[\d\-.\s()]{7,14}\d")),
]


def _fallback_regex_redact(text: str) -> str:
    redacted = text
    for label, pattern in _FALLBACK_PII_PATTERNS:
        redacted = pattern.sub(f"<{label}>", redacted)
    return redacted


def redact_text_with_presidio(text: str, language_code: str = "en") -> tuple[str, str]:
    """Redact PII from `text`.

    Returns (redacted_text, coverage):
      - "presidio_en":            full Presidio analysis ran against English-model
                                  NER + pattern recognizers.
      - "presidio_multilingual": full Presidio analysis ran against a multilingual
                                  transformer NER model (see _MULTILINGUAL_LANGUAGES
                                  for which languages this actually covers) +
                                  pattern recognizers.
      - "presidio_partial":      Presidio ran, but the input's language has neither
                                  of the above NER engines available, so only
                                  language-agnostic pattern recognizers (email,
                                  credit card, IBAN, ...) are trustworthy here --
                                  NER-based entities like names or addresses are
                                  not reliably covered.
      - "fallback_regex":        the Presidio engines were unavailable or raised,
                                  so a conservative regex-only scrubber was used
                                  instead.

    On any failure this always returns scrubbed text, never the raw input.
    """
    if not text:
        return "", "presidio_en"

    analyzer_engine, anonymizer_engine, _ = _get_presidio_engines()
    if analyzer_engine is None or anonymizer_engine is None:
        return _fallback_regex_redact(text), "fallback_regex"

    if language_code in _MULTILINGUAL_LANGUAGES:
        multilingual_analyzer = _get_multilingual_engine()
        if multilingual_analyzer is not None:
            try:
                results = multilingual_analyzer.analyze(text=text, language=language_code)
                anon = anonymizer_engine.anonymize(text=text, analyzer_results=results)
                return anon.text, "presidio_multilingual"
            except Exception as e:
                logger.error(
                    "Multilingual Presidio redaction failed, using fallback scrubber: %s",
                    e,
                )
                return _fallback_regex_redact(text), "fallback_regex"

    try:
        # Falls back to the English NLP model for any language without a
        # dedicated NER engine -- its NER results won't be meaningful for
        # non-English text, but its pattern recognizers (email, credit card,
        # IBAN, ...) are language-agnostic and still catch structured PII.
        results = analyzer_engine.analyze(text=text, language="en")
        anon = anonymizer_engine.anonymize(text=text, analyzer_results=results)
        coverage = "presidio_en" if language_code == "en" else "presidio_partial"
        return anon.text, coverage
    except Exception as e:
        logger.error("Presidio text redaction failed, using fallback scrubber: %s", e)
        return _fallback_regex_redact(text), "fallback_regex"


PII_COVERAGE_SEVERITY = {
    "presidio_en": 0,
    "presidio_multilingual": 0,
    "presidio_partial": 1,
    "fallback_regex": 2,
}
