import json
import requests

from .base import BaseTranslator

# ---------------------------------------------------------------------------
# Context template injected into every translation request.
# Now includes whole-video profile fields (genre, video_type, tone, etc.)
# so the LLM has much richer stylistic guidance without repeating analysis.
# ---------------------------------------------------------------------------
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
Speaker Emotion: {audio_speaker_emotion} (Ensure the tone reflects this emotion.)

Task: Translate the English text to French.
- The translation must be colloquial and match the genre/tone described above.
- STRICT LENGTH CONSTRAINT: The translation must match the duration/length of the source text perfectly.
- Do NOT include quotes, explanations, or notes.\
"""


class EnglishToFrenchTranslator(BaseTranslator):
    """
    Hits a llama.cpp server with a French-capable model.
    Context is now enriched with the whole-video VideoProfile (genre, video_type, etc.)
    alongside the per-segment visual context and speaker info.
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
                    "{context}\n\nTask: Translate this English movie dialogue into French for a {video_type} "
                    "({genre_examples}) — {maturity_rating}.\n"
                    "Rules for the translation:\n"
                    "- MUST be extremely crude, colloquial/informal French (argot/slang).\n"
                    "- ALWAYS use 'tu', 'te', 'toi'. NEVER use 'vous'. (e.g. use 't'es' instead of 'vous êtes').\n"
                    "- If there is swearing or aggression (e.g. 'fuck out', 'bitch'), translate it to heavy French slang "
                    "(e.g. 'casse-toi', 'putain', 'merde', 'connard', 'connerie').\n"
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
                    "You are an expert French dubbing translator for a {video_type} ({genre_examples}). "
                    "Your task is to translate English dialogue into highly colloquial, informal French. \n\n"
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

    def translate_text(self, segment) -> str:
        headers = {"Content-Type": "application/json"}

        speech_text = segment.get("text", "")

        # --- Speaker gender ---
        agc = segment.get("audio_gender_classification")
        speaker_gender = agc.get("label", "") if isinstance(agc, dict) else (agc if isinstance(agc, str) else "")

        # --- Speaker emotion ---
        aec = segment.get("audio_emotion_classification")
        speaker_emotion = aec.get("label", "") if isinstance(aec, dict) else (aec if isinstance(aec, str) else "")

        # --- Video profile fields (from the whole-video VideoProfile) ---
        vp = segment.get("video_profile") or {}
        # video_profile may be a dict (from JSON) or a VideoProfile Pydantic object
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
            audio_speaker_emotion=speaker_emotion,
        )

        # Build the messages, injecting all profile fields as well as context + text
        messages = []
        for msg in self.model_prompt_stencil[self.model_path]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"].format(
                    context=common_context,
                    text=speech_text,
                    video_type=video_type,
                    genre=genre,
                    genre_examples=genre_examples,
                    maturity_rating=maturity_rating,
                ),
            })

        data = {"messages": messages}

        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(data))
            print(data)
            response.raise_for_status()
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Translation error (FR): {e}")
            return ""


if __name__ == '__main__':
    """
    curl -s \
        --request POST \
        --url http://127.0.0.1:8080/v1/chat/completions \
        --header "Content-Type: application/json" \
        --data '{
            "messages": [
                {
                    "role": "user",
                    "content": "Translate the following text from English into French. Keep the tone as informal and comedy. Keep the word count strictly between 10 and 12.\\nEnglish: Mr. Wilson, you appear to have soiled yourself while on duty.\\nFrench:"
                }
            ]
        }'
    """
