import json
import requests
from tqdm import tqdm

def translate_text(text):
    url = 'http://localhost:8080/v1/chat/completions'
    headers = {'Content-Type': 'application/json'}
    data = {
        "messages": [
            {
                "role": "system",
                "content": f"Translate the text below to Hindi.: {text}"
            },
            {
                "role": "user",
                "content": f"{text}"
            }
        ]
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        res_json = response.json()
        return res_json['choices'][0]['message']['content']
    except Exception as e:
        print(f"Translation error: {e}")
        return ""

def translate_segments(json_path):
    # test translation server
    print("warm+test translation run")
    response = translate_text("if you see this, translation is working!")
    if (response == ""):
        print("Translation module isn't working as intended")
        return
    else:
        print(response)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # translation starts here
    for segment in tqdm(data.get("segments", [])):
        src_text = segment.get("text", "")
        translated = translate_text(src_text)
        segment["translation"] = translated

    # Save with new "translation" field per segment
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    translate_segments("media/deadpool-2025-12-18_15.27.22_extracted_dialog_diarize_result.json")
