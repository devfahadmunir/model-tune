#!/usr/bin/env python3
"""
test_model.py

Generates a response with the base model and with the LoRA-adapted model
side by side, so you can actually see what the fine-tune changed.

Usage:
    python test_model.py --config config/lora_config.yaml --prompt "Explain LoRA in one sentence."
    python test_model.py --config config/lora_config.yaml --prompts-file my_test_prompts.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("test_model")


def load_config(config_path: str) -> dict:
    with Path(config_path).open() as f:
        return yaml.safe_load(f)


def get_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts_file:
        lines = [line.strip() for line in Path(args.prompts_file).read_text().splitlines()]
        return [line for line in lines if line]
    if args.prompt:
        return [args.prompt]
    return [
        "Explain the difference between LoRA and full fine-tuning in two sentences.",
        "Write a short, polite email declining a meeting invite.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the lora config (reads model/adapter_path from it)")
    parser.add_argument("--prompt", default=None, help="A single prompt to test")
    parser.add_argument("--prompts-file", default=None, help="A text file with one prompt per line")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--skip-base", action="store_true", help="Only run the fine-tuned model (faster)")
    args = parser.parse_args()

    try:
        from mlx_lm import load, generate
    except ImportError:
        log.error("mlx-lm not installed. Run: pip install mlx-lm")
        sys.exit(1)

    cfg = load_config(args.config)
    model_name = cfg["model"]
    adapter_path = cfg.get("adapter_path", "./adapters")

    if not Path(adapter_path).exists():
        log.error("Adapter path %s does not exist yet. Run train.py first.", adapter_path)
        sys.exit(1)

    prompts = get_prompts(args)

    if not args.skip_base:
        log.info("Loading base model: %s", model_name)
        base_model, base_tokenizer = load(model_name)

    log.info("Loading fine-tuned model: %s + adapter %s", model_name, adapter_path)
    ft_model, ft_tokenizer = load(model_name, adapter_path=adapter_path)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'=' * 70}\nPROMPT {i}: {prompt}\n{'=' * 70}")

        chat = [{"role": "user", "content": prompt}]

        if not args.skip_base:
            base_prompt = base_tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            base_out = generate(base_model, base_tokenizer, prompt=base_prompt, max_tokens=args.max_tokens, verbose=False)
            print(f"\n--- BASE MODEL ---\n{base_out}")

        ft_prompt = ft_tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        ft_out = generate(ft_model, ft_tokenizer, prompt=ft_prompt, max_tokens=args.max_tokens, verbose=False)
        print(f"\n--- FINE-TUNED MODEL ---\n{ft_out}")


if __name__ == "__main__":
    main()
