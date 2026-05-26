from .base import BaseTranslator
from .dynamic_translator import LLMDynamicTranslator

def get_translator(language: str) -> BaseTranslator:
    """
    Return a dynamic translator instance for the requested language code.
    """
    return LLMDynamicTranslator(target_language=(language or "hi"))


def translate_segments(json_path: str, config: dict) -> None:
    """
    Common translation entry point. Leads to a language-specific translator.
    """
    target_language = config.get("target_language", "hi")
    translator = get_translator(target_language)
    translator.translate_segments(json_path)

# only the names listed in __all__ will be imported when using "from translations import *"
__all__ = ["translate_segments"]
