#!/usr/bin/env python3
"""
train.py

Thin, professional wrapper around `mlx_lm.lora`. Does the pre-flight checks
people usually forget (Apple Silicon? mlx installed? data present? HF auth
needed?) and prints a clear, timestamped log instead of hoping the raw
mlx_lm CLI output explains itself.

Usage:
    python train.py --config config/lora_config.yaml
    python train.py --config config/lora_config.yaml --dry-run   # just checks, no training
"""

from __future__ import annotations

import argparse
import logging
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("train")


def check_apple_silicon() -> None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        log.error("MLX requires Apple Silicon macOS. Detected: %s / %s", platform.system(), platform.machine())
        sys.exit(1)
    log.info("Platform OK: macOS on %s", platform.machine())


def check_mlx_available() -> None:
    try:
        import mlx.core as mx  # noqa: F401
        import mlx_lm  # noqa: F401
    except ImportError:
        log.error("mlx / mlx-lm not installed. Run: pip install mlx-lm")
        sys.exit(1)
    log.info("mlx-lm import OK.")


def check_data_dir(data_dir: str) -> None:
    train_file = Path(data_dir) / "train.jsonl"
    valid_file = Path(data_dir) / "valid.jsonl"
    missing = [str(p) for p in (train_file, valid_file) if not p.exists()]
    if missing:
        log.error("Missing data file(s): %s. Run prepare_data.py first.", ", ".join(missing))
        sys.exit(1)
    n_train = sum(1 for _ in train_file.open())
    n_valid = sum(1 for _ in valid_file.open())
    log.info("Data OK: %d train / %d valid examples in %s", n_train, n_valid, data_dir)


def check_mlx_lm_cli() -> None:
    if shutil.which("mlx_lm.lora") is None:
        log.warning("`mlx_lm.lora` not found on PATH; will invoke it via `python -m mlx_lm.lora` instead.")


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        log.error("Config file not found: %s", path)
        sys.exit(1)
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the mlx_lm.lora YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Run pre-flight checks only, skip training")
    args = parser.parse_args()

    log.info("=== Pre-flight checks ===")
    check_apple_silicon()
    check_mlx_available()
    cfg = load_config(args.config)
    check_data_dir(cfg.get("data", "./data"))
    check_mlx_lm_cli()

    adapter_path = Path(cfg.get("adapter_path", "./adapters"))
    adapter_path.mkdir(parents=True, exist_ok=True)

    log.info("=== Config summary ===")
    for key in ("model", "iters", "batch_size", "learning_rate", "fine_tune_type", "num_layers"):
        log.info("  %-15s %s", key + ":", cfg.get(key))

    if args.dry_run:
        log.info("Dry run complete — all checks passed. Remove --dry-run to start training.")
        return

    # Build the full command BEFORE logging or running it, so what gets printed
    # always matches what actually executes.
    cmd = [sys.executable, "-m", "mlx_lm", "lora", "--config", args.config]

    resume_file = adapter_path / "adapters.safetensors"
    if resume_file.exists():
        log.info("Found existing checkpoint: %s — resuming from it.", resume_file)
        cmd += ["--resume-adapter-file", str(resume_file)]
    else:
        log.info("No checkpoint found — starting fresh.")

    log.info("=== Starting mlx_lm lora ===")
    log.info("Running: %s", " ".join(cmd))

    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.error("Training failed with exit code %d.", result.returncode)
        sys.exit(result.returncode)

    log.info("Training complete. Adapter saved to: %s", adapter_path)
    log.info("Next: python test_model.py --config %s --prompt \"your test prompt\"", args.config)


if __name__ == "__main__":
    main()

# python train.py --config ./lora_config.yaml
#command to fuse the model and adapter into a single model for inference
# mlx_lm.fuse --model mlx-community/Llama-3.2-3B-Instruct-4bit --adapter-path ./adapters --save-path ./fused-model

# run this server
# mlx_lm.server --model ./fused-model --port 8080

