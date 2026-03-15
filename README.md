# AutoEIT Transcription Pipeline
Open-source transcription pipeline for Spanish Elicited Imitation Task (EIT) learner speech.

## Overview
This repository provides a reproducible pipeline for converting learner Spanish audio responses into sentence-level transcriptions suitable for linguistic analysis and downstream scoring. The system is designed for non-native speech, where generic ASR systems often fail due to accent transfer, hesitations, partial repetitions, and other learner-specific phenomena.

## Key Features
- **Audio Preprocessing:** Normalization and SNR checks for noisy learner recordings.
- **Robust Transcription:** Specialized decoding for learner-specific speech patterns.
- **Post-processing:** Preserves learner disfluencies while correcting machine artifacts.
- **Multi-format Export:** Batch results in Excel, CSV, and JSON.
- **Evaluation:** Quantitative assessment using Word Error Rate (WER) and Character Error Rate (CER).

## Repository Structure
```text
autoeit-transcription/
├── data/
│   ├── raw/           # Raw learner audio
│   ├── processed/     # Normalized/Segmented audio
│   └── metadata/      # EIT prompts and scoring rubrics
├── src/
│   ├── io/            # Data ingestion/export
│   ├── audio/         # Preprocessing & segmentation
│   ├── asr/           # ASR logic and model wrappers
│   ├── postprocess/   # Transcript cleanup
│   ├── eval/          # Metric calculation
│   └── cli.py         # Main entry point
├── configs/           # Pipeline configurations
├── notebooks/         # Analysis and demos
└── tests/             # Quality assurance
```

## Installation
### Requirements
- Python 3.10+
- FFmpeg (for audio processing)
- PyTorch (for ASR)

### Setup
```bash
git clone https://github.com/HumanAI/autoeit-transcription.git
cd autoeit-transcription
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
### 1. Data Preparation
Place input audio and workbook files under `data/raw/`.

### 2. Run Transcription
```bash
python -m src.cli --audio-dir data/raw --prompt-xlsx data/metadata/prompts.xlsx --output-xlsx output/results.xlsx
```

## Contribution
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License
Copyright (c) 2026 HumanAI Project. Distributed under the MIT License.
