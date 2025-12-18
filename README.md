# dubpls
Non-realtime dubbing - high-fidelity, open-access, dubbing for each language pair.

# Techniques
1. Cascaded - STT - LLM - TTS (modular but loss of emotio, tone)
	e.g. Kokuro TTS+LLM+STT (Unmute - uses Mistral Small 24B for LLM) for EN-FR
2. End2End - 

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
