from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import write_generation_outputs, write_partial_generation_outputs
from .bundle import load_story_bundle
from .models import GenerationRun, StageOutput, StageOutputCallback, StoryBundle
from .orchestrator import generate_full_book, validate_run
from .profiles import get_profile_by_repo_id


def _load_prompt(args: argparse.Namespace) -> str:
    return args.prompt


def _load_generation_bundle(args: argparse.Namespace) -> StoryBundle:
    profile = get_profile_by_repo_id(args.repo_id)
    return load_story_bundle(profile_name=profile.name, repo_id=args.repo_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone vLLM interface for the Pagestorm story models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="Run the staged generation flow and write artifacts.")
    generate_parser.add_argument("--repo-id", required=True, help="Known Pagestorm Hugging Face model repo ID.")
    generate_parser.add_argument("--prompt", required=True, help="Inline prompt text.")
    generate_parser.add_argument(
        "--output-directory",
        default=None,
        help="Optional directory for artifact.txt and the generated-title book text file.",
    )
    generate_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    generate_parser.add_argument("--gpu-memory-utilization", type=float, default=0.96)
    generate_parser.add_argument("--max-model-len", type=int, default=None)

    validate_parser = subparsers.add_parser("validate", help="Run staged generation and fail on regex mismatches.")
    validate_parser.add_argument("--repo-id", required=True, help="Known Pagestorm Hugging Face model repo ID.")
    validate_parser.add_argument("--prompt", required=True, help="Inline prompt text.")
    validate_parser.add_argument(
        "--output-directory",
        default=None,
        help="Optional directory for artifact.txt and the generated-title book text file.",
    )
    validate_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    validate_parser.add_argument("--gpu-memory-utilization", type=float, default=0.96)
    validate_parser.add_argument("--max-model-len", type=int, default=None)

    return parser


def _print_run_summary(output_directory: str | None, generation_run: GenerationRun) -> None:
    print(
        json.dumps(
            {
                "profile": generation_run.profile,
                "model": generation_run.model,
                "validation_success": generation_run.validation_success,
                "failed_stage_role": generation_run.failed_stage_role,
                "later_chapter_count": generation_run.later_chapter_count,
                "output_directory": None if output_directory is None else str(Path(output_directory).resolve()),
            },
            indent=2,
        )
    )


def _print_stage_separator(stage_output: StageOutput) -> None:
    trailing_newline_count = len(stage_output.text) - len(stage_output.text.rstrip("\n"))
    missing_newline_count = max(0, 2 - trailing_newline_count)
    if missing_newline_count:
        print("\n" * missing_newline_count, file=sys.stderr, end="", flush=True)


def _print_stage_text_chunk(text_chunk: str) -> None:
    print(text_chunk, file=sys.stderr, end="", flush=True)


def _build_stage_output_callback(
    *,
    bundle: StoryBundle,
    prompt: str,
    output_directory: str | None,
) -> StageOutputCallback:
    completed_stage_outputs: list[StageOutput] = []

    def handle_stage_output(stage_output: StageOutput) -> None:
        completed_stage_outputs.append(stage_output)
        _print_stage_separator(stage_output)
        if output_directory is not None:
            write_partial_generation_outputs(
                output_directory,
                bundle.protocol,
                prompt,
                completed_stage_outputs,
            )

    return handle_stage_output


def _run_generate_like_command(args: argparse.Namespace, *, validate_only: bool) -> int:
    bundle = _load_generation_bundle(args)
    prompt = _load_prompt(args)
    generation_function = validate_run if validate_only else generate_full_book
    stage_output_callback = _build_stage_output_callback(
        bundle=bundle,
        prompt=prompt,
        output_directory=args.output_directory,
    )
    generation_run = generation_function(
        bundle,
        prompt=prompt,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        stage_text_chunk_callback=_print_stage_text_chunk,
        stage_output_callback=stage_output_callback,
    )
    if args.output_directory is not None:
        write_generation_outputs(args.output_directory, generation_run)
    _print_run_summary(args.output_directory, generation_run)
    return 0 if generation_run.validation_success else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        return _run_generate_like_command(args, validate_only=False)
    if args.command == "validate":
        return _run_generate_like_command(args, validate_only=True)

    parser.error(f"Unsupported command: {args.command}")
    return 2
