import json
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import librosa
import numpy as np
import argparse

## Model 1
# Gender classification model setup
gender_model_name = "prithivMLmods/Common-Voice-Geneder-Detection"
gender_model = Wav2Vec2ForSequenceClassification.from_pretrained(gender_model_name)
gender_processor = Wav2Vec2FeatureExtractor.from_pretrained(gender_model_name)
gender_id2label = {
    "0": "female",
    "1": "male"
}

def classify_segment_audio(audio_clip, sample_rate):
    inputs = gender_processor(
        audio_clip,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = gender_model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    prediction = {gender_id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    main_label = gender_id2label[str(int(np.argmax(probs)))]
    return {"probs": prediction, "label": main_label}

## Model 2
# Emotion classification model setup
emotion_model_name = "prithivMLmods/Speech-Emotion-Classification"
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(emotion_model_name)
emotion_processor = Wav2Vec2FeatureExtractor.from_pretrained(emotion_model_name)
emotion_id2label = {
    "0": "Anger",
    "1": "Calm",
    "2": "Disgust",
    "3": "Fear",
    "4": "Happy",
    "5": "Neutral",
    "6": "Sad",
    "7": "Surprised"
}

def classify_audio_emotion_clip(audio_clip, sample_rate):
    inputs = emotion_processor(
        audio_clip,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = emotion_model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    prediction = {emotion_id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    main_label = emotion_id2label[str(int(np.argmax(probs)))]
    return {"probs": prediction, "label": main_label}

def process_audio_with_segments(audio_path, json_path, save_to=None):
    print(f"Loading audio: {audio_path}...")
    audio, sr = librosa.load(audio_path, sr=16000)
    
    print(f"Loading segments: {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    segments = data.get("segments", [])
    print(f"Processing {len(segments)} segments...")

    for idx, segment in enumerate(segments):
        seg_start = segment.get("start", None)
        seg_end = segment.get("end", None)
        
        if seg_start is not None and seg_end is not None:
            # Extract audio clip
            start_sample = int(round(sr * float(seg_start)))
            end_sample = int(round(sr * float(seg_end)))
            audio_clip = audio[start_sample:end_sample]
            
            # 1. Run Gender Classification
            gender_pred = classify_segment_audio(audio_clip, sr)
            gender_label = gender_pred["label"]
            gender_entry = {'label': gender_label, 'score': gender_pred["probs"][gender_label]}
            segment["audio_gender_classification"] = gender_entry
            
            # 2. Run Emotion Classification
            emotion_pred = classify_audio_emotion_clip(audio_clip, sr)
            emotion_label = emotion_pred["label"]
            emotion_entry = {'label': emotion_label, 'score': emotion_pred["probs"][emotion_label]}
            segment["audio_emotion_classification"] = emotion_entry

    out_path = save_to if save_to is not None else json_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Done! Results saved to: {out_path}")
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-task Audio Classifier")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio file (wav)")
    parser.add_argument("--json_path", type=str, help="Path to JSON with segments. If provided, runs classifiers on each segment.")
    parser.add_argument("--save_to", type=str, help="Path to save the output JSON. Defaults to overwriting json_path.")

    args = parser.parse_args()

    if args.json_path:
        # This runs both classifiers on every segment by default
        process_audio_with_segments(args.audio_path, args.json_path, save_to=args.save_to)
    else:
        print("[E] Need to provide audio and json path")