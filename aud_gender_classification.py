import json
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch
import librosa
import numpy as np

# Load model and processor
model_name = "prithivMLmods/Common-Voice-Geneder-Detection"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

# Label mapping
id2label = {
    "0": "female",
    "1": "male"
}

def classify_segment_audio(audio_clip, sample_rate):
    # Input: clip as numpy array, sample_rate (should already be 16k)
    inputs = processor(
        audio_clip,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }
    # Also return the main prediction
    main_label = id2label[str(int(np.argmax(probs)))]
    return {"probs": prediction, "label": main_label}

def process_audio_with_segments(audio_path, json_path, save_to=None):
    # Load and resample entire audio to 16kHz
    audio, sr = librosa.load(audio_path, sr=16000)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Classify each segment (segments list should be present)
    segments = data.get("segments", [])
    for segment in segments:
        seg_start = segment.get("start", None)
        seg_end = segment.get("end", None)
        if seg_start is not None and seg_end is not None:
            # Get sample indices for this segment
            start_sample = int(round(sr * float(seg_start)))
            end_sample = int(round(sr * float(seg_end)))
            audio_clip = audio[start_sample:end_sample]

            # Classify
            pred = classify_segment_audio(audio_clip, sr)

            # Save results in same json
            pred_label = pred["label"]
            entry = {'label': pred_label, 'score': pred["probs"][pred_label]}
            segment["audio_gender_classification"] = entry

    # Optionally save back to json (otherwise, just overwrite original)
    out_path = save_to if save_to is not None else json_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data

if __name__=='__main__':
    apath = "/home/prashant/Documents/dubpls/media/deadpool-2025-12-18_15.27.22_extracted_dialog.wav"
    jpath = "/home/prashant/Documents/dubpls/media/deadpool-2025-12-18_15.27.22_extracted_dialog_diarize_result.json"
    process_audio_with_segments(apath, jpath)
