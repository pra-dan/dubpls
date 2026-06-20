import json
import requests

from .base import BaseTranslator


class EnglishToHindiTranslator(BaseTranslator):
    """
    Hits a llama.cpp server, that uses "sarvam-translate.Q3_K_M".
    """

    target_language = "hi"
    # Sarvam EN-HI model
    model_path = "/models/sarvam-translate.Q3_K_M.gguf"

    # def __init__(self, url):
    #     self.url = url

    def translate_text(self, segment) -> str:
        headers = {"Content-Type": "application/json"}
        
        # Handle case where segment might be a string (from warmup) or a dict (from pipeline)
        if isinstance(segment, dict):
            speech_text = segment.get("text", "")
            vp = segment.get("video_profile")
            if not vp:
                raise ValueError("Missing video_profile metadata during translation. Cannot proceed with limited context.")
            
            dubbing_register = segment.get("dubbing_register")
            if not dubbing_register:
                raise ValueError("Missing dubbing_register metadata during translation. Cannot proceed with limited context.")

            if hasattr(vp, "model_dump"):
                vp = vp.model_dump()
            
            setting_summary = vp.get("setting_summary", "Not available")
            formality_level = vp.get("formality_level", "informal")
            overall_tone = vp.get("tone", "neutral")
            dialogue_justification = segment.get("dialogue_justification", "Not available")
            
            system_prompt = (
                f"You are an expert Hindi dubbing translator. "
                f"Translate the English text into Hindi that matches the exact scene tone. "
                f"Setting: {setting_summary}. Formality: {formality_level}. Tone: {overall_tone}. \n"
                f"Dialogue Justification: {dialogue_justification}\n\n"
                f"Rules for translation:\n"
                f"1. DO NOT translate English idioms/slang literally (e.g., 'Whiskey Dick'). Use context-appropriate Hindi/Bambaiya slang, Tapori language, or Hinglish (e.g., 'सुमड़ी', 'चिकने', 'नुन्नू').\n"
                f"2. Strict Word Count Constraint: The Hindi translation word count MUST be within 1 word of the English source.\n"
                f"3. Strict Formality Rule: If the tone is aggressive/intimate, ALWAYS use 'तू/तेरा' instead of 'तुम/आप'. NEVER add unprompted respectful honorifics like 'साब' or polite terms like 'मैडम' in street slang contexts.\n"
                f"4. Do not sanitize vulgarity if the context is aggressive."
            )
        else:
            speech_text = segment
            system_prompt = "Translate the text below to Hindi."

        data = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": f"{speech_text}"},
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