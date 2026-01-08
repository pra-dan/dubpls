import json
import requests

from .base import BaseTranslator


class EnglishToFrenchTranslator(BaseTranslator):
    """
    Hits a llama.cpp server with a French-capable model.
    """

    target_language: str = "fr"
    # Example FR-capable model; adjust as needed
    model_path: str = "/models/TowerInstruct-Mistral-7B-v0.2.Q6_K.gguf"

    def __init__(self, url: str = "http://localhost:8080/v1/chat/completions"):
        self.url = url

    def translate_text(self, text: str) -> str:
        headers = {"Content-Type": "application/json"}
        # You will likely want a more FR-specific prompt later
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Translate the following text from English into French. "
                        "Set the translation tone and formality using the fact that this is from a movie.\n"
                        f"English: {text}\nFrench:"
                    ),
                }
            ]
        }

        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Translation error (FR): {e}")
            return ""


