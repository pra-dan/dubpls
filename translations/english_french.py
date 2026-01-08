from .base import BaseTranslator


class EnglishToFrenchTranslator(BaseTranslator):
    target_language: str = "fr"
    warmup_text = None  # No-op until implementation is added

    def translate_text(self, text: str) -> str:
        raise NotImplementedError("French translation not implemented yet.")

