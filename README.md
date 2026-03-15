# Task I: Audio-to-Text Transcription

## Goal
Convert each provided Spanish EIT audio file into a 30-row participant transcription sheet without grammar correction.

## Inputs
- Audio directory: [input/audio](/Users/fallofpheonix/Project/Human AI/AutoEIT/task1/input/audio)
- Prompt workbook: [input/workbooks/AutoEIT Sample Audio for Transcribing.xlsx](/Users/fallofpheonix/Project/Human AI/AutoEIT/task1/input/workbooks/AutoEIT%20Sample%20Audio%20for%20Transcribing.xlsx)

## Command
```bash
python -m task1.src.run \
  --audio-dir task1/input/audio \
  --prompt-xlsx "task1/input/workbooks/AutoEIT Sample Audio for Transcribing.xlsx" \
  --output-xlsx task1/output/AutoEIT_Task1_Transcriptions.xlsx
```

## First-Run Model Note
- The first ASR run downloads the selected `faster-whisper` model from Hugging Face.
- The pipeline disables the Xet transfer path automatically because it can stall in desktop environments.
- If model fetch still stalls, prefetch a local model directory and pass it through `--model-size`.
- Direct prefetch command:
```bash
python scripts/prefetch_faster_whisper_model.py \
  --model tiny \
  --output-dir task1/models/faster-whisper-tiny
```
- Then run Task I with:
```bash
python -m task1.src.run \
  --audio-dir task1/input/audio \
  --prompt-xlsx "task1/input/workbooks/AutoEIT Sample Audio for Transcribing.xlsx" \
  --output-xlsx task1/output/AutoEIT_Task1_Transcriptions.xlsx \
  --model-size task1/models/faster-whisper-tiny
```
- A small debug run can be forced with `--sheet 38010-2A --model-size tiny` before attempting all 4 files.
- `tiny` is useful for debugging the pipeline path only; it is not expected to produce submission-quality transcriptions on this dataset.

## Output Contract
- Preserves the original participant sheets from the provided workbook.
- Populates `Transcription`, `Normalized transcription`, and `Notes` columns.
- Adds `AutoEIT_Task1_Summary` with flat participant-level rows.
- Fails if any participant cannot be aligned to exactly 30 utterances.

## Current Status
- Code path, staged assets, tests, notebook, and local-model prefetch path are present.
- A full debug workbook has been generated at `task1/output/AutoEIT_Task1_Transcriptions_tiny.xlsx` using a locally prefetched `tiny` model.
- The pipeline is runtime-valid, but `tiny` output quality is not submission-grade for this dataset. A stronger local model remains the next quality step.

## Notebook
- Notebook: [notebooks/task1_transcription.ipynb](/Users/fallofpheonix/Project/Human AI/AutoEIT/task1/notebooks/task1_transcription.ipynb)
- PDF: [output/AutoEIT_Task1_Notebook.pdf](/Users/fallofpheonix/Project/Human AI/AutoEIT/task1/output/AutoEIT_Task1_Notebook.pdf)
