"""CLI entrypoint for Task I."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.asr.pipeline import EXPECTED_SENTENCE_COUNT, run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoEIT Task I transcription pipeline")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--prompt-xlsx", type=Path, required=True)
    parser.add_argument("--output-xlsx", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_SENTENCE_COUNT)
    parser.add_argument("--model-size", default="large-v3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--language", default="es")
    parser.add_argument("--sheet", default=None, help="Process a single participant sheet.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    transcripts = run_batch(
        audio_dir=args.audio_dir,
        prompt_workbook=args.prompt_xlsx,
        output_workbook=args.output_xlsx,
        expected_count=args.expected_count,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
        only_sheet=args.sheet,
    )
    print(f"Processed {len(transcripts)} participant sheet(s).")
    print(f"Output workbook: {args.output_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
