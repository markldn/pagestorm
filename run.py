#!/usr/bin/env python3
"""Drive pagestorm staged book generation against a local llama-server (GGUF).

This reuses pagestorm's own orchestrator / parsing / validation unchanged; only
the generation backend (pagestorm.vllm_runtime) has been swapped for a
llama-server client. Point it at the local bundle dir (config + tokenizer) and
make sure serve.sh is running.

Example:
    python3 run.py --prompt "Thriller in Zurich" --output-directory out/zurich
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the local (patched) pagestorm package wins over any installed copy.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pagestorm.artifacts import write_generation_outputs, write_partial_generation_outputs
from pagestorm.bundle import load_story_bundle
from pagestorm.models import GenerationRun, StageOutput, StoryBundle
from pagestorm.orchestrator import generate_full_book, validate_run

DEFAULT_BUNDLE_PATH = str(Path(__file__).resolve().parent / "models" / "pagestorm-research-preview-14b-full-book")


def _build_stage_output_callback(*, bundle: StoryBundle, prompt: str, output_directory: str | None):
    completed: list[StageOutput] = []

    def handle(stage_output: StageOutput) -> None:
        completed.append(stage_output)
        # Visible stage separator on stderr so streamed text stays readable.
        print(f"\n\n===== completed stage: {stage_output.role} =====\n", file=sys.stderr, flush=True)
        if output_directory is not None:
            write_partial_generation_outputs(output_directory, bundle.protocol, prompt, completed)

    return handle


def _print_chunk(text_chunk: str) -> None:
    print(text_chunk, file=sys.stderr, end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--bundle-path", default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--output-directory", default=None)
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Override context window (default: profile's 262144). "
                             "Lower it to fit VRAM, e.g. 32768.")
    parser.add_argument("--validate", action="store_true",
                        help="Run the strict validation pass (fails on regex mismatch).")
    args = parser.parse_args(argv)

    bundle = load_story_bundle(profile_name="14b", bundle_path=args.bundle_path)
    callback = _build_stage_output_callback(
        bundle=bundle, prompt=args.prompt, output_directory=args.output_directory
    )
    generation_function = validate_run if args.validate else generate_full_book
    run: GenerationRun = generation_function(
        bundle,
        prompt=args.prompt,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.0,  # ignored by the llama-server backend
        max_model_len=args.max_model_len,
        stage_text_chunk_callback=_print_chunk,
        stage_output_callback=callback,
    )
    if args.output_directory is not None:
        write_generation_outputs(args.output_directory, run)
    print(json.dumps({
        "profile": run.profile,
        "model": run.model,
        "validation_success": run.validation_success,
        "failed_stage_role": run.failed_stage_role,
        "later_chapter_count": run.later_chapter_count,
        "output_directory": None if args.output_directory is None else str(Path(args.output_directory).resolve()),
    }, indent=2))
    return 0 if run.validation_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
