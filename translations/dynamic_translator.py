import json
import requests
import os
import sys

# adding root to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import get_language_name
from .base import BaseTranslator

COMMON_CONTEXT_TEMPLATE = """\
=== VIDEO PROFILE ===
Video Type    : {video_type}
Genre         : {genre}
Genre Tags    : {genre_examples}
Maturity      : {maturity_rating}
Overall Tone  : {overall_tone}
Formality     : {formality_level}
Setting       : {setting_summary}

=== SCENE CONTEXT ===
Visual Context: {visual_context}

=== SPEAKER INFO ===
Speaker Gender : {audio_speaker_gender} (Ensure correct grammatical gender for self-referential words.)

Task: Translate the English text to {language_name}.
- The translation must be colloquial and match the genre/tone described above.
- STRICT LENGTH CONSTRAINT: The translation must match the duration/length of the source text perfectly.
- Do NOT include quotes, explanations, or notes.\
"""

class LLMDynamicTranslator(BaseTranslator):
    def __init__(self, target_language: str = "fr"):
        self.target_language = target_language
        self.language_name = get_language_name(target_language)
        self.model_path = "/models/TowerInstruct-Mistral-7B-v0.2.Q6_K.gguf"
        
        self.model_prompt_stencil = {
            "/models/TowerInstruct-Mistral-7B-v0.2.Q6_K.gguf": [
                {
                    "role": "user",
                    "content": (
                        "{context}\n\nTask: Translate this English movie dialogue into {language_name} for a {video_type} "
                        "({genre_examples}) — {maturity_rating}.\n"
                        "Rules for the translation:\n"
                        "- MUST be extremely crude, colloquial/informal {language_name} (slang).\n"
                        "- If there is swearing or aggression, translate it to heavy {language_name} slang.\n"
                        "- Do not translate literally if there is a more natural aggressive/comedic {language_name} slang term.\n"
                        "- Must strictly match the brevity/length of the English source.\n"
                        "English: {text}\n{language_name}:"
                    ),
                }
            ],
            "/models/Dolphin3.0-Llama3.1-8B.Q4_K_M.gguf": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert {language_name} dubbing translator for a {video_type} ({genre_examples}). "
                        "Your task is to translate English dialogue into highly colloquial, informal {language_name}. \n\n"
                        "Rules:\n"
                        "1. Do NOT reply to the text or provide notes. Only output the translation.\n"
                        "2. Use slang, informal phrasing.\n"
                        "3. STRICT LENGTH CONSTRAINT: The output length must closely match the input.\n"
                        "4. Use the context: {context}"
                    ),
                },
                {
                    "role": "user",
                    "content": "{text}"
                }
            ],
        }

    def translate_text(self, segment) -> str:
        headers = {"Content-Type": "application/json"}
        speech_text = segment.get("text", "")

        agc = segment.get("audio_gender_classification")
        speaker_gender = agc.get("label", "") if isinstance(agc, dict) else (agc if isinstance(agc, str) else "")


        vp = segment.get("video_profile")
        if not vp:
            raise ValueError("Missing video_profile metadata during translation. Cannot proceed with limited context.")

        dubbing_register = segment.get("dubbing_register")
        if not dubbing_register:
            raise ValueError("Missing dubbing_register metadata during translation. Cannot proceed with limited context.")

        if hasattr(vp, "model_dump"):
            vp = vp.model_dump()

        video_type       = vp.get("video_type", "movie")
        genre            = vp.get("genre", "unknown")
        genre_examples   = vp.get("genre_examples", "")
        maturity_rating  = vp.get("maturity_rating", "unknown")
        overall_tone     = vp.get("tone", "neutral")
        formality_level  = vp.get("formality_level", "informal")
        setting_summary  = vp.get("setting_summary") or "Not available."
        video_context = segment.get("video_context", "No visual context!")

        common_context = COMMON_CONTEXT_TEMPLATE.format(
            video_type=video_type,
            genre=genre,
            genre_examples=genre_examples,
            maturity_rating=maturity_rating,
            overall_tone=overall_tone,
            formality_level=formality_level,
            setting_summary=setting_summary,
            visual_context=video_context,
            audio_speaker_gender=speaker_gender,
            language_name=self.language_name,
        )

        messages = []
        stencil = self.model_prompt_stencil.get(self.model_path, self.model_prompt_stencil["/models/TowerInstruct-Mistral-7B-v0.2.Q6_K.gguf"])
        for msg in stencil:
            messages.append({
                "role": msg["role"],
                "content": msg["content"].format(
                    context=common_context,
                    text=speech_text,
                    video_type=video_type,
                    genre=genre,
                    genre_examples=genre_examples,
                    maturity_rating=maturity_rating,
                    language_name=self.language_name,
                ),
            })

        data = {"messages": messages}

        try:
            # Note: assuming self.url comes from BaseTranslator
            response = requests.post(self.url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Translation error ({self.target_language}): {e}")
            return ""
