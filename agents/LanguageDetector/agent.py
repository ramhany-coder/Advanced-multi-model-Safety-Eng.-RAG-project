from lingua import Language, LanguageDetectorBuilder

language_detector = LanguageDetectorBuilder.from_languages(
    Language.ARABIC,
    Language.ENGLISH,
    Language.FRENCH,
).build()


def local_language_detector_agent(state) -> dict:
    query = state.get("query", "")
    detected = language_detector.detect_language_of(query)

    if detected is None:
        return {
            "language": "English",
            "language_code": "en",
            "origin_en": True
        }

    lang_name = detected.name.capitalize()
    lang_code = detected.iso_code_639_1.name.lower()

    return {
        "language": lang_name,
        "language_code": lang_code,
        "origin_en": lang_code == "en"
    }
