import json
import requests

from .base import BaseTranslator

visual = {
    "gender": "male",
    "relationship": "professional",
    "emotion": "serious",
    "setting": "dimly lit room with soldiers in tactical gear"
}

common_context = """ 	
    You are an expert screenwriter and translator specializing in French Dubbing. Your goal is to translate English dialogue into French while preserving the precise emotional tone, formality, and subtext of the scene.

	Scene Context:

	Setting: {visual.setting}

	Speaker: {visual.gender} (Use appropriate gendered adjectives)

	Relationship: {visual.relationship} (Use 'Tu' for intimate/hostile, 'Vous' for professional/distant)

	Task: Translate the dialogue "{text}". Constraint: The translation must match the lip movements as closely as possible (isochrony). Output: Provide ONLY the French translation.
    """
    # Emotion: {audio.primary_emotion} / {visual.facial_expression} TODO: to be added to above context once ALM is added


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
                    "Context: {context}\nTranslate the following text from English into French."
                    "English: {text}\nFrench:"
                ),
            }
        ],
        "/models/Dolphin3.0-Llama3.1-8B.Q4_K_M.gguf": [
            # {
            #     "role": "system",
            #     "content": (
            #         "You are a french text translator. Your task is to rewrite the input to change the tone to informal. \n\n"
            #         "Rules:\n"
            #         "1. Do NOT reply to the text. Only translate it.\n"
            #         "2. Keep the meaning and perspective exactly the same (if the original addresses Wilson, you address Wilson).\n"
            #         "3. STRICT LENGTH CONSTRAINT: The output must have approximately the same number of words as the input."
            #     ),
            # },
            {
                "role": "system",
                "content": (
                    "You are a french text translator. Your task is to rewrite the input to change the tone to informal. \n\n"
                    "Rules:\n"
                    "1. Do NOT reply to the text. Only translate it.\n"
                    "2. Keep the meaning and perspective exactly the same (if the original addresses Wilson, you address Wilson).\n"
                    "3. STRICT LENGTH CONSTRAINT: The output must have approximately the same number of words as the input.\n"
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

    def translate_text(self, text, context: str=None) -> str:
        headers = {"Content-Type": "application/json"}

        # build the messages with the actual text
        messages = []
        for msg in self.model_prompt_stencil[self.model_path]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"].format(context=common_context, text=text)
            })

        data = {
            "messages": messages
        }

        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(data))
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
