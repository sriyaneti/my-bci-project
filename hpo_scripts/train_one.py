# Run ONE training job with ONE configuration and return ONE val metric

# creates an output dir for the trial (outdir)
# copies the config (hyperparameters) for a specific trial in outdir
# saves output of stdout.txt - output from training scripts
# saves output of stderr.txt - outputs of potential error --> DELETE THIS??
# extracts the validation metric
    # saves this to metrics.json
    # this is MOST IMPORTANT bc Optuna reads this

#!/usr/bin/env python3

# hpo_scripts/train_one.py
import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml


def run_training_and_get_val_metric(cfg: dict, work_dir: Path) -> float:
    """
    TODO: Replace the code BELOW with your real training call.
    It MUST return one scalar validation metric (higher is better by default).
    """
    # Example stub (replace):
    #   from src.train import train_main
    #   val_metric = train_main(cfg, output_dir=work_dir)
    #   return float(val_metric)

    # ----- STUB ONLY -----
    time.sleep(0.5)
    # pretend metric depends on a hyperparameter
    n_units = cfg["model"]["n_units"]
    dropout = cfg["model"]["rnn_dropout"]
    val_metric = 1.0 / (1.0 + abs(n_units - 768) / 768 + abs(dropout - 0.4))
    return float(val_metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config.yaml for this trial")
    ap.add_argument("--work_dir", required=True, help="Trial output directory (writes metrics.json here)")
    ap.add_argument("--metric_name", default="val_score", help="Key name written into metrics.json")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # (Optional) make runs reproducible if you pass seed in cfg
    # seed = cfg.get("seed", 0)
    # set_seed(seed)

    try:
        val_metric = run_training_and_get_val_metric(cfg, work_dir)
    except Exception as e:
        # If training fails, write a metrics.json so Optuna can record the failure.
        # You can also raise to mark trial as failed; Optuna will handle it.
        err_path = work_dir / "error.txt"
        err_path.write_text(f"{type(e).__name__}: {e}\n")
        raise

    metrics = {
        args.metric_name: float(val_metric),
    }
    (work_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Print something helpful (goes into stdout.txt in your subprocess wrapper)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()