# 🎤 HooperTTS

**HooperTTS is an AI narration engine that enhances scripts before generating expressive speech using Qwen3-TTS.**

Instead of sending raw text directly to a TTS model, HooperTTS analyzes the script, improves pacing, narration flow, semantic chunking, pronunciation, and style prompts to produce more natural voice generation.

Built on top of the official Qwen3-TTS models.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-v1.0--Beta-orange)
![Powered by](https://img.shields.io/badge/Powered%20by-Qwen3--TTS-red)


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/XADITYAM/HooperTTS/blob/main/notebooks/HooperTTS_Qwen.ipynb
)

## Features

- 🎙️ Voice cloning using Qwen3-TTS
- ✨ Script optimization before generation
- 🧠 Narration planning
- 🎭 Multiple narration profiles
- 🔊 Pronunciation engine
- ✂️ Semantic chunking
- 📊 Benchmark & evaluation framework
- 💻 CLI interface
- 🌐 Gradio web interface
- ☁️ Google Colab support

-           Script (.txt)
                 │
                 ▼
        HooperTTS Engine
        ├───────────────┐
        │ Script Optimizer
        │ Pronunciation
        │ Narration Planner
        │ Semantic Chunking
        │ Profiles
        └───────────────┘
                 │
                 ▼
        Official Qwen3-TTS
                 │
                 ▼
          Generated Speech

 ## Acknowledgements

HooperTTS is built on top of the excellent work by:

- Qwen Team - Qwen3-TTS
- NeuralFalcon - Google Colab inspiration

HooperTTS extends the workflow with narration optimization, semantic chunking, pronunciation enhancement, and expressive prompting.

## Support the Project

If you found HooperTTS useful, consider giving the repository a ⭐.
