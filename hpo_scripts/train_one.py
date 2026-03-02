#!/usr/bin/env python3
"""
hpo_scripts/train_one.py

Run ONE training job with ONE configuration and return ONE scalar validation metric.

What it does:
- Creates/uses a trial work_dir
- Loads the trial config.yaml
- Fixes relative dataset.dataset_dir (relative to the config file location)
- Forces output_dir/checkpoint_dir into the trial folder (so trials don't overwrite)
- (Preflight) clamps days_per_batch to 1 to avoid sampling empty days
- Runs the real NEJM training pipeline (BrainToTextDecoder_Trainer.train())
- Extracts ONE metric value and writes it to metrics.json (Optuna reads this)
- Writes stdout.txt, stderr.txt, and (if failure) error.txt into the trial folder
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import yaml

# NEJM_ROOT: use env var if set (e.g. on cluster), otherwise auto-detect relative to this script
NEJM_ROOT = Path(os.environ.get("NEJM_ROOT", str(Path(__file__).parent.parent / "nejm-brain-to-text"))).resolve()
NEJM_MODEL_TRAINING_DIR = (NEJM_ROOT / "model_training").resolve()


# ----------------- logging -----------------
def setup_logging(work_dir: Path) -> logging.Logger:
    """
    Create a logger that writes:
      - INFO+ to stdout.txt
      - ERROR+ to stderr.txt
    We also keep a console handler (optional) if you want, but usually trial logs go to files.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_one")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # don't double-log if root logger configured elsewhere

    # Clear existing handlers if script reused in same python proc (rare but safe)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stdout_path = work_dir / "stdout.txt"
    stderr_path = work_dir / "stderr.txt"

    h_out = logging.FileHandler(stdout_path, mode="w")
    h_out.setLevel(logging.INFO)
    h_out.setFormatter(fmt)

    h_err = logging.FileHandler(stderr_path, mode="w")
    h_err.setLevel(logging.ERROR)
    h_err.setFormatter(fmt)

    logger.addHandler(h_out)
    logger.addHandler(h_err)

    return logger


# ----------------- config helpers -----------------
def resolve_dataset_dir(cfg: Dict[str, Any], config_path: Path) -> None:
    """
    If cfg['dataset']['dataset_dir'] is relative, resolve it relative to the config file directory.
    Mutates cfg in-place.
    """
    if "dataset" not in cfg or not isinstance(cfg["dataset"], dict):
        raise KeyError("Config missing 'dataset' section (dict).")

    ds = cfg["dataset"]
    if "dataset_dir" not in ds:
        raise KeyError("Config missing dataset.dataset_dir")

    ds_dir = ds["dataset_dir"]
    if isinstance(ds_dir, str):
        p = Path(ds_dir)
        if not p.is_absolute() and not ds_dir.startswith("~"):
            ds["dataset_dir"] = str((config_path.parent / p).resolve())


def force_trial_output_dirs(cfg: Dict[str, Any], work_dir: Path) -> None:
    """
    Force all outputs for this trial to stay inside work_dir/out.
    Mutates cfg in-place.
    """
    out_dir = (work_dir / "out").resolve()
    ckpt_dir = (out_dir / "checkpoint").resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg["output_dir"] = str(out_dir)
    cfg["checkpoint_dir"] = str(ckpt_dir)


def preflight_avoid_empty_days(cfg: Dict[str, Any]) -> None:
    """
    Preflight: clamp dataset.days_per_batch to 1 to avoid sampling empty sessions/days.
    Mutates cfg in-place.
    """
    ds = cfg.get("dataset")
    if not isinstance(ds, dict):
        return
    cur = ds.get("days_per_batch", 4)
    try:
        cur_i = int(cur)
    except Exception:
        cur_i = 4
    ds["days_per_batch"] = min(cur_i, 1)


# ----------------- NEJM import -----------------
def import_trainer(logger: logging.Logger):
    """
    Import the NEJM trainer reliably:
      - add NEJM_ROOT and NEJM_MODEL_TRAINING_DIR to sys.path
      - chdir to NEJM_ROOT to satisfy code that assumes relative paths
    """
    if not NEJM_MODEL_TRAINING_DIR.exists():
        raise FileNotFoundError(f"NEJM model_training dir not found at {NEJM_MODEL_TRAINING_DIR}")

    for p in [NEJM_ROOT, NEJM_MODEL_TRAINING_DIR]:
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)

    os.chdir(str(NEJM_ROOT))
    logger.info(f"Changed working directory to: {NEJM_ROOT}")

    from rnn_trainer import BrainToTextDecoder_Trainer  # type: ignore
    return BrainToTextDecoder_Trainer


def run_training_and_get_val_metric(cfg: Dict[str, Any], metric_name: str, logger: logging.Logger) -> float:
    """
    Runs real NEJM training for ONE trial.
    Returns ONE scalar metric (metric_name) from the returned metrics dict.
    """
    try:
        from omegaconf import OmegaConf
    except ImportError as e:
        raise ImportError("omegaconf not found in this environment. Install it in b2txt25.") from e

    BrainToTextDecoder_Trainer = import_trainer(logger)

    args = OmegaConf.create(cfg)
    trainer = BrainToTextDecoder_Trainer(args)

    logger.info("Starting trainer.train() ...")
    metrics = trainer.train()
    logger.info("trainer.train() returned.")

    if not isinstance(metrics, dict):
        raise TypeError(f"trainer.train() returned {type(metrics)} (expected dict). Value: {metrics}")

    if metric_name not in metrics:
        raise KeyError(f"Metric '{metric_name}' not in returned metrics. Keys: {list(metrics.keys())}")

    val = metrics[metric_name]
    if isinstance(val, list):
        val = min(val)
    try:
        return float(val)
    except Exception as e:
        raise TypeError(f"Metric '{metric_name}' value not float-convertible: {val} ({type(val)})") from e


# ----------------- main -----------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to trial config.yaml")
    ap.add_argument("--work_dir", required=True, help="Trial output directory")
    ap.add_argument(
        "--metric_name",
        default="val_score",
        help="Metric key to extract from trainer.train() result dict",
    )
    args = ap.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    error_path = work_dir / "error.txt"
    metrics_path = work_dir / "metrics.json"

    logger = setup_logging(work_dir)

    logger.info("=== train_one.py starting ===")
    logger.info(f"python:        {sys.executable}")
    logger.info(f"python_version:{sys.version.splitlines()[0]}")
    logger.info(f"config_path:   {config_path}")
    logger.info(f"work_dir:      {work_dir}")
    logger.info(f"metric_name:   {args.metric_name}")
    logger.info(f"NEJM_ROOT:     {NEJM_ROOT}")
    logger.info(f"PYTHONPATH:    {os.environ.get('PYTHONPATH', '')}")
    logger.info(f"initial_cwd:   {Path.cwd()}")

    try:
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with config_path.open("r") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            raise TypeError(f"Loaded config is not a dict. Got: {type(cfg)}")

        # Fix paths + preflight + force isolation
        resolve_dataset_dir(cfg, config_path)
        preflight_avoid_empty_days(cfg)
        force_trial_output_dirs(cfg, work_dir)

        # Debug prints
        ds = cfg.get("dataset", {})
        logger.info(f"dataset_dir:    {ds.get('dataset_dir') if isinstance(ds, dict) else 'N/A'}")
        logger.info(f"days_per_batch: {ds.get('days_per_batch') if isinstance(ds, dict) else 'N/A'}")
        logger.info(f"output_dir:     {cfg.get('output_dir')}")
        logger.info(f"checkpoint_dir: {cfg.get('checkpoint_dir')}")

        # Run training
        val = run_training_and_get_val_metric(cfg, args.metric_name, logger)

        # Write metrics.json (Optuna reads this)
        metrics_obj = {args.metric_name: float(val)}
        metrics_path.write_text(json.dumps(metrics_obj, indent=2))

        logger.info("=== train_one.py finished OK ===")
        logger.info(json.dumps(metrics_obj))

        return 0

    except Exception:
        tb = traceback.format_exc()
        # Write a clean traceback file that you can open directly
        work_dir.mkdir(parents=True, exist_ok=True)
        error_path.write_text(tb)

        # Log as ERROR so it lands in stderr.txt
        logger.error("=== train_one.py FAILED ===")
        logger.error(tb)

        return 1


if __name__ == "__main__":
    raise SystemExit(main())