# dubpls
Non-realtime dubbing - high-fidelity, open-access, dubbing for English (`En`) to non-English.
## Demo

Below is a quick demo of dubpls (work in progress :). Note how dubpls tries to preserve the original speaker's voice and emotion.
<table>
  <tr>
    <th width="50%">Original (Undubbed)</th>
    <th width="50%">Dubbed (French)</th>
  </tr>
  <tr>
    <td>
      <a href="assets/deadpool-2025-12-18_15.27.22.mp4">
        <img src="assets/thumbnail_original.png" alt="Original Video" width="100%">
      </a>
    </td>
    <td>
      <a href="assets/deadpool-2025-12-18_15.27.22_fr_feb3_1657.mp4">
        <img src="assets/thumbnail_dubbed.png" alt="Dubbed Video" width="100%">
      </a>
    </td>
  </tr>
</table>

Video Credit: Marvel Studios (Deadpool vs. Wolverine)

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
<!-- 
# Updates 2025
## Dec 17
The project was initially meant to be local-first, accuracy second - Imagine a single-click-installation VLC extension that doesn't require you to install any dependency and takes some time to generate the new dub audio with some delay. We can use a preprocessing / buffer window that gives the illusion of real-timeness at the cost of fidelity.

Fidelity: lets not underestimate it. A high-fidelity audio implies that the output resembles the input audio's meaning, emotions and naturalness. 

Rather, what if the user could download a file, just like a subtitile and plug-and-play with VLC? Much cooler, right? Moving away from the real-time thing frees opens the possibility of higher fidelity. That also means that we will now require a repository to contain the previously processed/dubbed files - similar to subdl or any other subtitle repo. So that some users can get lucky and save time by downloading the audio for what they need. I'll think on that later.

Only keeping the English to Hindi for start. The workflow would be: video/audio (Eng/Hin) -> dubpls -> audio (Hin/Eng). We just need one solution. 

So we'll only update the workflow for a language pair if there is a higher-fidelity possible - not for speed or multi-platform support.

# Updates 2026
## Jan 7
Pausing dev for HI as the only acceptable TTS is [Coqui_ai-XTTS-v3](/home/prashant/Documents/coqui-ai-TTS/run.py) and even the translation looks un-natural. So taking up FR as next language.

## Jan 14
The issue with segments (#8: loss of word and #3: no voice cloning) is not with ttsfrd but rather with the prompt (audio+text) length. #8 is resolved once the length increased. 

#3: Is this supposed to be scary | Estce cense faire peur? -> no cloning | pre, post procN checked
#8: Who are you? | Qui estu? -> too short prompt(text/audio)

The author [suggests 5 to 10s of prompt audio](https://github.com/FunAudioLLM/CosyVoice/issues/1070#issuecomment-2727273122). 

## Jan 22
The above issues were fixed by using the default prefix added to the prompt text in CosyVoice examples. Now, the only odd thing is the noticeable difference in the tone between reference/input audio and output. Next step would be adding some VLM for video scene understanding at low FPS and adding context to the common json. But deep-research suggests that I use difference XLMs for video and audio; the VLM tells us what scene it is while the ALM tells us whether the speaker is male/female and age, etc. Whether the speaker is yelling/laughing, is also derive-able from ALM.

VLM: 
- options: openbmb/MiniCPM-V-2_6-int4
- Sample prompt: "Analyze the key interaction in this frame. 1. Identify the gender of the speaker. 2. Describe the relationship between the speaker and listener (e.g., Intimate, Professional, Hostile). 3. Describe the speaker's emotion. Output in JSON format."
But how do we select the clip to use as input to VLM?

ALM:
- options: MiniCPM V2.6 (8B)
- output is a classification with score

System Prompt Template for Mistral-Nemo:

	You are an expert screenwriter and translator specializing in French Dubbing. Your goal is to translate English dialogue into French while preserving the precise emotional tone, formality, and subtext of the scene.

	Scene Context:

	Setting: {visual.setting}

	Speaker: {visual.speaker_gender} (Use appropriate gendered adjectives)

	Relationship: {visual.proximity} (Use 'Tu' for intimate/hostile, 'Vous' for professional/distant)

	Emotion: {audio.primary_emotion} / {visual.facial_expression}

	Task: Translate the dialogue "{text}". Constraint: The translation must match the lip movements as closely as possible (isochrony). Output: Provide ONLY the French translation.

(Optional) few-shot eg for 
	To fix the "robotic" output, we must leverage Few-Shot Prompting. We should include 3-5 examples in the prompt that demonstrate how to handle different tones.

	Example 1 (Angry/Informal): "Get out!" -> "Dégage!" (Not "Sortez").

	Example 2 (Polite/Formal): "Get out." -> "Veuillez sortir."

	Example 3 (Sad/Resigned): "I don't care." -> "C'est pas grave..." (Softened).

(Optional) Re-check
	Tone Consistency: Use a secondary LLM (e.g., GPT-4o-mini or a quantized 7B judge) to evaluate the output.

	Prompt: "Does the French phrase 'Dégage' match the context 'Angry man shouting'? Yes/No."


## Jan 28:
For the scene context extraction using VLM, this can be the pipeline:
- run scene detector and log start+end timestamps. Also save clips.
- iterate through each segment in json and get its TS.
- for each segment_ts, find at least 2 scenes before it (itself being the 3rd), OR the current scene upto first 7 seconds.

End pipeline would be 
... - STT(json) - scene_clipping + VLM + ALM - Translation (LLM) - ... 

-->


# Resources
- [StreamSpeech - only support for Fr, En, Es, De](https://github.com/ictnlp/StreamSpeech)
- [CosyVoice-3.0 TTS+zero-shot voice cloning](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) - works for Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian, 18+ Chinese dialects.
- [Other En-Ch models](https://github.com/FunAudioLLM/CosyVoice?tab=readme-ov-file#evaluation) This can also be used as reference for other languages in future.
- NVIDIA Nemotron (NIM) for cascaded system.
- [Separation of voice, music and effects from singel audio - really cool eg from movies](https://cslikai.cn/TIGER/)
- [Speaker Diarization/Separation with visual cues](https://huggingface.co/JusperLee/Dolphin)
- [TODO LLM-based TTS models](https://huggingface.co/blog/YatharthS/llm-tts-models)
- [TODO Making NeuTTS 200x realtime](https://huggingface.co/blog/YatharthS/making-neutts-200x-realtime)
- [Video-dubbing](https://huggingface.co/spaces/vuxuanhoan/video-dubbing)
- [NLLB demo for any2any language translation](https://huggingface.co/spaces/UNESCO/nllb)
- [Voice cloning using Coqui](https://coqui-tts.readthedocs.io/en/latest/vc.html)
- [Fine-tune indic TTS](https://snorbyte.com/blog/train-sota-multilingual-indic-tts)
- [this fine-tune of neutts-air](https://huggingface.co/jaeyong2/neutts-air-hi-preview)
- [Comparative study on Prosody](https://arxiv.org/pdf/2511.02104)
----- 

# Setup

## Tested Env:
```
driver: 580.95.05
PyTorch==2.8.0+cu128
```

a. Install WhisperX using [official instructions](https://github.com/m-bain/whisperX).
<!-- ```console
# env 1: whisperx
conda create -n whisperx -y
conda activate whisperx
conda install pip -y
conda install python==3.11 -y
``` -->
b. Clone repo and install dependencies
```
git clone --recursive https://github.com/pra-dan/dubpls.git
pip install -r external/TIGER/requirements.txt
pip install -r requirements.txt
```

c. Install CosyVoice using [official instructions](https://github.com/FunAudioLLM/CosyVoice) and [download weights and install ttsfrd](https://github.com/FunAudioLLM/CosyVoice?tab=readme-ov-file#model-download) for CosyVoice3.

<!-- # env 2: cosyvoice (For CosyVoice)
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
cd external/CosyVoice
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
apt-get install sox libsox-dev 

# Download weights for Cosyvoice. In a python console --- 
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='external/CosyVoice/CosyVoice2-0.5B')
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='external/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B')
snapshot_download('FunAudioLLM/CosyVoice-ttsfrd', local_dir='external/CosyVoice/pretrained_models/CosyVoice-ttsfrd') 

cd external/CosyVoice/pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```
-->

d. Download the translation model weights and move to "models" directory
| language | weights / quants |
|--|--|
| Hindi | [mradermacher's quant of Sarvam](https://huggingface.co/mradermacher/sarvam-translate-GGUF/blob/main/sarvam-translate.Q3_K_M.gguf) |
| French | [mradermacher's quant of TowerInstruct-Mistral-7B](https://huggingface.co/mradermacher/TowerInstruct-Mistral-7B-v0.2-GGUF?show_file_info=TowerInstruct-Mistral-7B-v0.2.Q6_K.gguf) |

e. Get HF read-access token and add to env. From official WhisperX docs:
> To enable Speaker Diarization, include your Hugging Face access token (read) that you can generate from [Here](https://huggingface.co/settings/tokens) after the `--hf_token` argument and accept the user agreement for the following models: [Segmentation](https://huggingface.co/pyannote/segmentation-3.0) and [Speaker-Diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) ...

Save the token to a `.env` file e.g.,
```txt
HF_READ_TOKEN=hf_LxPdD...
```

<!-- Launch Sarvam Translation server
```bash
sudo docker compose up # uses port 8080

# Hit the server using
# curl -s \
#     --request POST --url http://127.0.0.1:8080/v1/chat/completions \
#     --header "Content-Type: application/json" \
#     --data '{"messages": [ { "role": "system", "content": "Translate the text below to Hindi." }, { "role": "user", "content": "Mr. Wilson, you appear to have soiled yourself while on duty." }]}'
``` -->

# Run pipeline 
Choose language in the [config.yaml](config.yaml).
```
python3 main.py
```

# TODO
- Improve LLM/prompting for better tone determination.
- Add support for Hindi.
- Reduce multiple environments to one or independent services.

# Additional fixes
a. Explicitly add lib path to [stay away from dependency issues](https://github.com/m-bain/whisperX/issues/902#issuecomment-2646634513):
```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

b. If you use torch>2.6, whisperX will likely give [another issue](https://github.com/m-bain/whisperX/issues/1304#issuecomment-3599713003). The suggested solution is
```
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true 
```