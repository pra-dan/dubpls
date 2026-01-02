# dubpls
Non-realtime dubbing - high-fidelity, open-access, dubbing for each language pair.

# Techniques
1. Cascaded - STT - LLM - TTS (modular but loss of emotio, tone)
	e.g. Kokuro TTS+LLM+STT (Unmute - uses Mistral Small 24B for LLM) for EN-FR
	
	[TODO] We need to find a way where the info lost b/w STT-LLM is retained and then overlayed onto final result.
	- TTS like [this fine-tune of neutts-air](https://huggingface.co/jaeyong2/neutts-air-hi-preview) take in ref text+audio to extract speaker tone and then generates speech from test text. Wow the hindi results are good.
	Other, standard TTS include KyutaiTTS, Sesame and Nari Lab's Dia2.
	
	
2. End2End - Generally, streaming systems.

# Plan
A. Demixing: 
- De-mixes the speech audio into dialog + M&E i.e Music & Effects
- Choice: TIGER / Demucs v4

B. Speaker Diarization & Identification:
- maps each audio to new/registered user ids.
- Audio-only diarization: we use only audio to correctly label each audio to the speaker. This mainly helps with zero-shot voice cloning so that we don't clone the female audio using male's previous audio, etc. WhisperX helps with correct timestamp mapping (more on this below).
- Choice: PyAudio + WhisperX
- (optional) visual-audio diarization: This uses the video part to ensure that we only translate only for the foreground persons and not some BG noise. Choice: TalkNet, Dolphin.

C. Transcription & Translation:
- For multi-speaker convos, especially those lacking clear gap between consecutive speakers' speeches, we need to know exactly when a word was spoken (a.k.a Forced Alignment). This is done by WhisperX which uses VAD (for Hallucination Handling) and maps a timestamp to each word in the transcribed text.
- For translation, we use IndicTrans2 as it beats most general EN-HI translators.
- Isomorphic Translation: This ensures that the syllable count of the output text is close to that of input. (e.g 10% of input text). Choice: Any local LLM like quantized `Llama-3-8B-Instruct`. This can also be helpful in adding consistency checks e.g. Prompt: "Ensure the honorifics (Aap/Tum) remain consistent for Speaker A across this dialogue conversation."

D. TTS & Zero-shot voice & emotion cloning:
- To clone voice we prefer CosyVoice 3.0 over F5-TTS.
- We can extract emotion using Speech Emotion Recognition (SER). Choice: `wav2vec2-large-robust-12-ft-emotion-msp-dim`. This should classify emotion and add corresponding token to CosyVoice (`<|angry|>`).

E. (Optional) Synchronization:
- Aims for perfect audio-lip synchronization for audio-video inputs.
- Models like `IndexTTS-2` or `WSOLA` can modulate HI output audio to fit to EN speech length.

# Updates 2025
## Dec 17
The project was initially meant to be local-first, accuracy second - Imagine a single-click-installation VLC extension that doesn't require you to install any dependency and takes some time to generate the new dub audio with some delay. We can use a preprocessing / buffer window that gives the illusion of real-timeness at the cost of fidelity.

Fidelity: lets not underestimate it. A high-fidelity audio implies that the output resembles the input audio's meaning, emotions and naturalness. 

Rather, what if the user could download a file, just like a subtitile and plug-and-play with VLC? Much cooler, right? Moving away from the real-time thing frees opens the possibility of higher fidelity. That also means that we will now require a repository to contain the previously processed/dubbed files - similar to subdl or any other subtitle repo. So that some users can get lucky and save time by downloading the audio for what they need. I'll think on that later.

Only keeping the English to Hindi for start. The workflow would be: video/audio (Eng/Hin) -> dubpls -> audio (Hin/Eng). We just need one solution. 

So we'll only update the workflow for a language pair if there is a higher-fidelity possible - not for speed or multi-platform support.

## To explore
- Seed-TTS
- Qwen 2.5-Omni-7B
https://news.ycombinator.com/item?id=46264491#46279654

# Resources
- [StreamSpeech - only support for Fr, En, Es, De](https://github.com/ictnlp/StreamSpeech)
- [CosyVoice-3.0 TTS+zero-shot voice cloning](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) - works for Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian), 18+ Chinese dialects.
- [Other En-Ch models](https://github.com/FunAudioLLM/CosyVoice?tab=readme-ov-file#evaluation) This can also be used as reference for other languages in future.
- NVIDIA Nemotron (NIM) for cascaded system.
- [Separation of voice, music and effects from singel audio - really cool eg from movies](https://cslikai.cn/TIGER/)
- [Speaker Diarization/Separation with visual cues](https://huggingface.co/JusperLee/Dolphin)
- [TODO LLM-based TTS models](https://huggingface.co/blog/YatharthS/llm-tts-models)
- [TODO Making NeuTTS 200x realtime](https://huggingface.co/blog/YatharthS/making-neutts-200x-realtime)
- [Video-dubbing](https://huggingface.co/spaces/vuxuanhoan/video-dubbing)

----- 

# Setup



Get HF read-access token and add to env. From official WhisperX docs:
> To enable Speaker Diarization, include your Hugging Face access token (read) that you can generate from [Here](https://huggingface.co/settings/tokens) after the `--hf_token` argument and accept the user agreement for the following models: [Segmentation](https://huggingface.co/pyannote/segmentation-3.0) and [Speaker-Diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) ...
```
export HF_READ_TOKEN=hf_LxPrdDqc...
```

Install requirements
```console
conda create -n whisperx -y
conda activate whisperx
conda install pip -y
conda install python==3.11 -y

#Explicitly add lib path to [stay away from dependency issues](https://github.com/m-bain/whisperX/issues/902#issuecomment-2646634513):

export LD_LIBRARY_PATH=/PATH/TO/ENV/whisperx/lib/python3.11/site-packages/nvidia/cudnn/lib/ # use which python perhaps

git clone --recursive https://github.com/pra-dan/dubpls.git

pip install -r external/TIGER/requirements.txt
pip install -r requirements.txt
```

Download the translation model weights - using [mradermacher's quant of Sarvam](https://huggingface.co/mradermacher/sarvam-translate-GGUF/blob/main/sarvam-translate.Q3_K_M.gguf) here.

Launch Sarvam Translation server
```bash
sudo docker compose up # uses port 8080

# Hit the server using
# curl -s \
#     --request POST --url http://127.0.0.1:8080/v1/chat/completions \
#     --header "Content-Type: application/json" \
#     --data '{"messages": [ { "role": "system", "content": "Translate the text below to Hindi." }, { "role": "user", "content": "Mr. Wilson, you appear to have soiled yourself while on duty." }]}'
```
