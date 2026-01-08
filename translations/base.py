import json
from typing import Optional

from tqdm import tqdm


class BaseTranslator(self):
    warmup_text: Optional[str] = "if you see this, translation is working!"

    # @property
    # @abstractmethod
    # def target_language(self) -> str:
    #     return target_language

    @abstractmethod
    def translate_text(self, text: str) -> str:
        """
        Translate the provided text into the target language.
        """

    def translate_segments(self, json_path: str) -> None:
        self._warm_up()

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for segment in tqdm(data.get("segments", [])):
            src_text = segment.get("text", "")
            translated = self.translate_text(src_text)
            target_language = self.target_language
            segment[target_language] = translated

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def _warm_up(self) -> None:
        """
        Runs the translation in target lang once, before bombarding 
        the server with failing requests
        """
        if not self.warmup_text:
            return
        print("warm+test translation run")
        response = self.translate_text(self.warmup_text)
        if response == "":
            print("Translation module isn't working as intended")
        else:
            print(response)

