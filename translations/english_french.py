import json
import requests

from .base import BaseTranslator

COMMON_CONTEXT_TEMPLATE = """
Context from Video: {visual_context}

Speaker Gender: {audio_speaker_gender} (Ensure correct grammatical gender for self-referential words).
Speaker Emotion: {audio_speaker_emotion} (Ensure the tone reflects this emotion).

Task: Translate the English text to French.
- The translation must be colloquial.
- STRICT LENGTH CONSTRAINT: The translation must match the duration/length of the source text perfectly.
- Do NOT include quotes, explanations, or notes.
"""

class EnglishToFrenchTranslator(BaseTranslator):
    """
    Hits a llama.cpp server with a French-capable model.
    """

    target_language = "fr"
    # Example FR-capable model; adjust as needed
    model_path = "/models/TowerInstruct-Mistral-7B-v0.2.Q6_K.gguf"
    # model_path = "/models/Dolphin3.0-Llama3.1-8B.Q4_K_M.gguf"

    # Each model has its own prompt format.
    model_prompt_stencil = {
        "/models/TowerInstruct-Mistral-7B-v0.2.Q6_K.gguf": [
            {
                "role": "user",
                "content": (
                    "{context}\n\nTask: Translate this English movie dialogue into French for an R-rated comedic action movie.\n"
                    "Rules for the translation:\n"
                    "- MUST be extremely crude, colloquial/informal French (argot/slang).\n"
                    "- ALWAYS use 'tu', 'te', 'toi'. NEVER use 'vous'. (e.g. use 't'es' instead of 'vous êtes').\n"
                    "- If there is swearing or aggression (e.g. 'fuck out', 'bitch'), translate it to heavy French slang (e.g. 'casse-toi', 'putain', 'merde', 'connard', 'connerie').\n"
                    "- Do not translate literally if there is a more natural aggressive/comedic French slang term.\n"
                    "- Must strictly match the brevity/length of the English source.\n"
                    "English: {text}\nFrench:"
                ),
            }
        ],
        "/models/Dolphin3.0-Llama3.1-8B.Q4_K_M.gguf": [
            {
                "role": "system",
                "content": (
                    "You are an expert French dubbing translator for an R-rated comedic action movie. Your task is to translate English dialogue into highly colloquial, informal French. \n\n"
                    "Rules:\n"
                    "1. Do NOT reply to the text or provide notes. Only output the translation.\n"
                    "2. Use slang, informal phrasing (argot), and NEVER use 'vous' (always use 'tu').\n"
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

    # def __init__(self, url):
    #     self.url = url

    def translate_text(self, segment) -> str:
        headers = {"Content-Type": "application/json"}

        speech_text = segment.get("text", "") 

        agc = segment.get("audio_gender_classification")
        speaker_gender = agc.get("label", "") if isinstance(agc, dict) else (agc if isinstance(agc, str) else "")
        
        aec = segment.get("audio_emotion_classification")
        speaker_emotion = aec.get("label", "") if isinstance(aec, dict) else (aec if isinstance(aec, str) else "")

        video_context = segment.get("video_context", "No visual context!")
        common_context = COMMON_CONTEXT_TEMPLATE.format(visual_context=video_context, audio_speaker_gender=speaker_gender, audio_speaker_emotion=speaker_emotion) 

        # build the messages with the actual text
        messages = []
        for msg in self.model_prompt_stencil[self.model_path]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"].format(context=common_context, text=speech_text)
            })

        data = {
            "messages": messages
        }

        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(data)); print(data)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Translation error (FR): {e}")
            return ""

if __name__=='__main__':
    """
    curl -s \
        --request POST \
        --url http://127.0.0.1:8080/v1/chat/completions \
        --header "Content-Type: application/json" \
        --data '{
            "messages": [
                {
                    "role": "user",
                    "content": "Translate the following text from English into French. Keep the tone as informal and comedy. Keep the word count strictly between 10 and 12.\nEnglish: Mr. Wilson, you appear to have soiled yourself while on duty.\nFrench:"
                }
            ]
        }'
    """
