#!/usr/bin/env python3
"""
prepare_data.py

Pulls an instruction-style dataset from Hugging Face and converts it into
the chat-message JSONL format mlx-lm expects for LoRA fine-tuning:

    {"messages": [{"role": "user", "content": "..."},
                   {"role": "assistant", "content": "..."}]}

Works with any dataset as long as you tell it which columns hold the
user prompt and the assistant response (or a pre-built "messages" column).

Usage examples
--------------
# Dolly-15k: has "instruction"/"context"/"response" columns
python prepare_data.py \
    --dataset databricks/databricks-dolly-15k \
    --prompt-field instruction \
    --response-field response \
    --limit 800 \
    --output-dir ./data

# A dataset that already stores a "messages" column (e.g. chat-format sets)
python prepare_data.py \
    --dataset HuggingFaceH4/no_robots \
    --messages-field messages \
    --limit 800 \
    --output-dir ./data
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("prepare_data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="HF dataset repo id, e.g. databricks/databricks-dolly-15k")
    p.add_argument("--split", default="train", help="Dataset split to load (default: train)")
    p.add_argument("--config-name", default=None, help="HF dataset config/subset name, if the dataset has one")

    p.add_argument("--messages-field", default=None,
                    help="Column already containing a list of {role, content} messages")
    p.add_argument("--prompt-field", default=None, help="Column holding the user prompt text")
    p.add_argument("--response-field", default=None, help="Column holding the assistant response text")
    p.add_argument("--context-field", default=None,
                    help="Optional column with extra context to prepend to the prompt (e.g. Dolly's 'context')")
    p.add_argument("--system-prompt", default=None,
                    help="Optional fixed system message to prepend to every example")

    p.add_argument("--limit", type=int, default=None, help="Cap the number of examples (recommended while iterating)")
    p.add_argument("--val-split", type=float, default=0.1, help="Fraction held out for validation (default 0.1)")
    p.add_argument("--seed", type=int, default=42, help="Shuffle seed for the train/val split")
    p.add_argument("--output-dir", default="./data", help="Where to write train.jsonl / valid.jsonl")
    p.add_argument("--min-chars", type=int, default=1, help="Drop examples shorter than this (either field)")
    return p.parse_args()


def build_example(ex: dict, args: argparse.Namespace) -> dict | None:
    """Return a {'messages': [...]} dict, or None if this row should be skipped."""
    if args.messages_field:
        messages = ex.get(args.messages_field)
        if not messages or not isinstance(messages, list):
            return None
        # Trust the dataset's own roles/content, just make sure shape is right.
        cleaned = [{"role": m["role"], "content": m["content"]} for m in messages if "role" in m and "content" in m]
        if len(cleaned) < 2:
            return None
        if args.system_prompt and cleaned[0]["role"] != "system":
            cleaned.insert(0, {"role": "system", "content": args.system_prompt})
        return {"messages": cleaned}

    if not args.prompt_field or not args.response_field:
        raise ValueError(
            "Provide either --messages-field, or both --prompt-field and --response-field."
        )

    prompt = (ex.get(args.prompt_field) or "").strip()
    response = (ex.get(args.response_field) or "").strip()
    if len(prompt) < args.min_chars or len(response) < args.min_chars:
        return None

    if args.context_field:
        context = (ex.get(args.context_field) or "").strip()
        if context:
            prompt = f"{prompt}\n\nContext:\n{context}"

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})
    return {"messages": messages}


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Missing dependency. Install with: pip install datasets")
        sys.exit(1)

    log.info("Loading %s (split=%s)...", args.dataset, args.split)
    ds = load_dataset(args.dataset, args.config_name, split=args.split) if args.config_name \
        else load_dataset(args.dataset, split=args.split)

    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    log.info("Loaded %d raw examples. Columns: %s", len(ds), ds.column_names)

    examples = []
    skipped = 0
    for ex in ds:
        built = build_example(ex, args)
        if built is None:
            skipped += 1
            continue
        examples.append(built)

    if not examples:
        log.error("No usable examples produced. Check --prompt-field/--response-field/--messages-field "
                   "against the dataset's actual columns: %s", ds.column_names)
        sys.exit(1)

    log.info("Built %d examples (%d skipped as empty/invalid).", len(examples), skipped)

    random.seed(args.seed)
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_split)) if len(examples) > 1 else 0
    valid, train = examples[:n_val], examples[n_val:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "valid.jsonl", valid)

    log.info("Wrote %d train examples -> %s", len(train), out_dir / "train.jsonl")
    log.info("Wrote %d valid examples -> %s", len(valid), out_dir / "valid.jsonl")
    log.info("Done. Sample example:\n%s", json.dumps(train[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# python prepare_data.py \
#     --dataset HuggingFaceH4/no_robots \
#     --messages-field messages \
#     --limit 200 \
#     --output-dir ./data