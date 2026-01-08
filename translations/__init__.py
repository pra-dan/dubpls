# from .base import BaseTranslator
from .english_hindi import EnglishToHindiTranslator
from .english_french import EnglishToFrenchTranslator


def get_translator(language: str) -> BaseTranslator:
    """
    Return a translator instance for the requested language code.
    """
    normalized = (language or "").lower()
    registry = {
        "hi": EnglishToHindiTranslator,
        "fr": EnglishToFrenchTranslator,
    }
    translator_cls = registry.get(normalized, EnglishToHindiTranslator)
    return translator_cls()


def translate_segments(config: dict, json_path: str) -> None:
    """
    Common translation entry point. Leads to a language-specific translator.
    """
    target_language = config["target_language"]
    translator = get_translator(target_language)
    translator.translate_segments(json_path)

# only the names listed in __all__ will be imported when using "from translations import *"
# __all__ = ["BaseTranslator", "translate_segments", "get_translator"]
__all__ = ["translate_segments"]

