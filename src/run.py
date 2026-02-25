#!/usr/bin/env python3
"""
CLI entry point for data generation, formatting, splitting, and base-model testing.

Run from project root:
  python -m src.run test-base-model --task extraction --runner openai
  python -m src.run generate-data --task extraction --runner openai --limit 100
  python -m src.run format-data
  python -m src.run split-data
  python -m src.run pipeline --task extraction --runner openai --limit 100
"""

import argparse
import os
import sys

# Ensure src is on path when run as python -m src.run from project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from controllers import DataController
from data import format_data_for_finetuning, prepare_rawdata, split_data
from evaluation import LocalRunner, OpenAIRunner, run_task
from evaluation.eval_base_local import eval_base_model
from models.enums import ModelEnum
from models.shcemes import ExtractNewsDetails, TranslationStory
from utils.prompt_template import (
    create_details_extraction_prompt,
    translation_messagges_prompt,
)


def _get_processed_dir():
    from controllers import BaseController
    return BaseController().processed_data_dir


def _get_runner(runner: str, model_name: str | None = None):
    if runner == "openai":
        return OpenAIRunner(model_name or ModelEnum.OPENAI_MODEL.value)
    if runner == "local":
        return LocalRunner(model_name or ModelEnum.BASE_MODEL_QWEN.value)
    raise ValueError(f"Unknown runner: {runner}. Use 'openai' or 'local'.")


def _get_task_config(task: str):
    if task == "extraction":
        return (
            create_details_extraction_prompt,
            ExtractNewsDetails,
            "Extract the story details into a JSON.",
        )
    if task == "translation":
        return (
            translation_messagges_prompt,
            TranslationStory,
            f"Translate the following story to {ModelEnum.TARGET_LANG.value} and output JSON according to the schema.",
        )
    raise ValueError(f"Unknown task: {task}. Use 'extraction' or 'translation'.")


def cmd_test_base_model(args):
    """Run base or teacher model on one or a few stories and print output."""
    dc = DataController()
    if args.limit and args.limit > 1:
        raw_data = dc.load_raw_data()[: args.limit]
        stories = [{"content": r.get("content", r.get("story", ""))} for r in raw_data]
    else:
        story_text = dc.load_example_story()
        stories = [{"content": story_text}]

    build_messages_fn, schema_cls, _ = _get_task_config(args.task)
    runner = _get_runner(args.runner, args.model)
    target_lang = ModelEnum.TARGET_LANG.value if args.task == "translation" else None

    for i, story in enumerate(stories):
        text = story["content"].strip()
        if not text:
            continue
        if args.limit and args.limit > 1:
            print(f"--- Example {i + 1} ---")
        kwargs = {"text": text}
        if target_lang is not None:
            kwargs["target_lang"] = target_lang
        messages = build_messages_fn(schema_cls, **kwargs)
        if args.runner == "local":
            response = eval_base_model(
                args.model or ModelEnum.BASE_MODEL_QWEN.value, messages
            )
        else:
            response, _ = run_task(
                runner=runner,
                build_messages_fn=build_messages_fn,
                schema_cls=schema_cls,
                **kwargs,
            )
        print(response)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(response)
            print(f"Saved to {args.output}")


def cmd_generate_data(args):
    """Generate SFT data using teacher (OpenAI or local) and save to JSONL."""
    dc = DataController()
    raw_data = dc.load_raw_data()
    if args.limit and args.limit > 0:
        raw_data = raw_data[: args.limit]

    output_file = args.output or os.path.join(_get_processed_dir(), "prepared_data.jsonl")
    if os.path.exists(output_file) and not args.overwrite:
        print(f"Output exists: {output_file}. Use --overwrite to replace.")
        return

    build_messages_fn, schema_cls, _ = _get_task_config(args.task)
    runner = _get_runner(args.runner, args.model)
    target_lang = ModelEnum.TARGET_LANG.value if args.task == "translation" else None

    if args.overwrite and os.path.exists(output_file):
        os.remove(output_file)

    prepare_rawdata(
        target_lang=target_lang,
        runner=runner,
        build_messages_fn=build_messages_fn,
        schema_cls=schema_cls,
        raw_data=raw_data,
        output_file=output_file,
    )
    count = sum(1 for _ in open(output_file, encoding="utf-8") if _.strip())
    print(f"Generated {count} examples -> {output_file}")


def cmd_format_data(args):
    """Format prepared JSONL into instruction/input/output structure. Returns in-memory list."""
    input_path = args.input or os.path.join(_get_processed_dir(), "prepared_data.jsonl")
    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}. Run generate-data first.")
        return
    data = format_data_for_finetuning(input_path)
    print(f"Formatted {len(data)} examples from {input_path}")
    if args.output:
        import json
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
        print(f"Wrote formatted data to {args.output}")
    return data


def cmd_split_data(args):
    """Split formatted data into train.json and val.json."""
    processed_dir = _get_processed_dir()
    input_path = args.input or os.path.join(processed_dir, "prepared_data.jsonl")
    if not os.path.exists(input_path):
        print(f"Input not found: {input_path}. Run generate-data and format-data first.")
        return
    data = format_data_for_finetuning(input_path)
    if not data:
        print("No data to split.")
        return
    split_data(data)
    train_path = os.path.join(processed_dir, "train.json")
    val_path = os.path.join(processed_dir, "val.json")
    print(f"Split {len(data)} examples -> {train_path}, {val_path}")


def cmd_pipeline(args):
    """Run generate-data -> format-data -> split-data in sequence."""
    processed_dir = _get_processed_dir()
    output_file = os.path.join(processed_dir, "prepared_data.jsonl")

    # Generate
    gen_args = argparse.Namespace(
        task=args.task,
        runner=args.runner,
        model=args.model,
        limit=args.limit,
        output=output_file,
        overwrite=args.overwrite,
    )
    cmd_generate_data(gen_args)

    # Format + split (split_data reads from formatted list; we format from prepared_data.jsonl)
    data = format_data_for_finetuning(output_file)
    if not data:
        print("No data to split after generation.")
        return
    split_data(data)
    train_path = os.path.join(processed_dir, "train.json")
    val_path = os.path.join(processed_dir, "val.json")
    print(f"Pipeline done. Train: {train_path}, Val: {val_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LoRA finetuning: test base model, generate data, format, split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # test-base-model
    p_test = subparsers.add_parser("test-base-model", help="Run base/teacher model on one or few stories")
    p_test.add_argument("--task", choices=["extraction", "translation"], default="extraction")
    p_test.add_argument("--runner", choices=["openai", "local"], default="openai")
    p_test.add_argument("--model", type=str, default=None, help="Model name (default from ModelEnum)")
    p_test.add_argument("--limit", type=int, default=None, help="Number of stories (default: 1 from example)")
    p_test.add_argument("--output", type=str, default=None, help="Save response to file")

    # generate-data
    p_gen = subparsers.add_parser("generate-data", help="Generate SFT data with teacher, write JSONL")
    p_gen.add_argument("--task", choices=["extraction", "translation"], default="extraction")
    p_gen.add_argument("--runner", choices=["openai", "local"], default="openai")
    p_gen.add_argument("--model", type=str, default=None)
    p_gen.add_argument("--limit", type=int, default=None, help="Max number of stories (default: all)")
    p_gen.add_argument("--output", type=str, default=None, help="Output JSONL path")
    p_gen.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")

    # format-data
    p_fmt = subparsers.add_parser("format-data", help="Format prepared JSONL for finetuning")
    p_fmt.add_argument("--input", type=str, default=None, help="Prepared JSONL path")
    p_fmt.add_argument("--output", type=str, default=None, help="Optional: write formatted JSON")

    # split-data
    p_split = subparsers.add_parser("split-data", help="Split formatted data into train.json and val.json")
    p_split.add_argument("--input", type=str, default=None, help="Prepared JSONL (will be formatted then split)")

    # pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Run generate-data then format and split")
    p_pipe.add_argument("--task", choices=["extraction", "translation"], default="extraction")
    p_pipe.add_argument("--runner", choices=["openai", "local"], default="openai")
    p_pipe.add_argument("--model", type=str, default=None)
    p_pipe.add_argument("--limit", type=int, default=None)
    p_pipe.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.command == "test-base-model":
        cmd_test_base_model(args)
    elif args.command == "generate-data":
        cmd_generate_data(args)
    elif args.command == "format-data":
        cmd_format_data(args)
    elif args.command == "split-data":
        cmd_split_data(args)
    elif args.command == "pipeline":
        cmd_pipeline(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
