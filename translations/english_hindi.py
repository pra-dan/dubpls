import json
import requests

from .base import BaseTranslator


class EnglishToHindiTranslator(BaseTranslator):
    """
    Hits a llama.cpp server, that uses "sarvam-translate.Q3_K_M"
    """
    target_language = "hi"

    def __init__(self, url: str = "http://localhost:8080/v1/chat/completions"):
        self.url = url

    def translate_text(self, text: str) -> str:
        headers = {"Content-Type": "application/json"}
        data = {
            "messages": [
                {
                    "role": "system",
                    "content": f"Translate the text below to Hindi.: {text}",
                },
                {"role": "user", "content": f"{text}"},
            ]
        }

        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Translation error: {e}")
            return ""

