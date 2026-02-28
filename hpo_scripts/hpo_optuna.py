# hpo_scripts/hpo_optuna.py
import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import optuna
import yaml


def deep_update(base: dict, upd: dict) -> dict:
    """Recursively updates nested dicts."""
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

"""
This function defines the search space of which specific hyperparameters you are
trying to optimize. It returns a nested dict that will be merged into base.yaml
"""
def sample_hparams(trial: optuna.Trial) -> dict:
    # Capacity / regularization
    n_units = trial.suggest_int("n_units", 384, 1024, step=128)
    rnn_dropout = trial.suggest_float("rnn_dropout", 0.0, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # LR schedule knobs (you use cosine with lr_max/lr_min)
    lr_max = trial.suggest_float("lr_max", 5e-4, 8e-3, log=True)
    lr_min_frac = trial.suggest_float("lr_min_frac", 0.01, 0.2)
    lr_min = lr_max * lr_min_frac

    return {
        "model": {
            "n_units": n_units,
            "rnn_dropout": rnn_dropout,
        },
        "weight_decay": weight_decay,
        "lr_max": lr_max,
        "lr_min": lr_min,

        # Keep day-specific LR schedule aligned (optional but usually sensible)
        "lr_max_day": lr_max,
        "lr_min_day": lr_min,
    }


def objective(trial: optuna.Trial, base_cfg: dict, args) -> float:
    # Create trial directory
    trial_name = f"trial_{trial.number:05d}"
    trial_dir = Path(args.run_dir) / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Sample hyperparameters
    hparams_patch = sample_hparams(trial)

    # Merge with base config
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # cheap deep copy
    cfg = deep_update(cfg, hparams_patch)

    # Make outputs unique per trial to avoid overwriting between trials
    cfg["output_dir"] = str(trial_dir / "model_out")
    cfg["checkpoint_dir"] = str(trial_dir / "checkpoints")

    # Safety: don't accidentally load weights from some prior run
    cfg["init_from_checkpoint"] = False
    cfg["init_checkpoint_path"] = None

    # Optional: set per-trial seed
    if args.seed is not None:
        cfg["seed"] = int(args.seed) + int(trial.number)

    # Write config.yaml
    cfg_path = trial_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Build command to run train_one.py
    cmd = [
        args.python,
        str(Path(args.train_one).resolve()),
        "--config", str(cfg_path),
        "--work_dir", str(trial_dir),
        "--metric_name", args.metric_name,
    ]

    # Save commands.json
    commands = {"cmd": cmd}
    (trial_dir / "commands.json").write_text(json.dumps(commands, indent=2))

    # Run subprocess, capture stdout/stderr to stdout.txt
    stdout_path = trial_dir / "stdout.txt"
    with open(stdout_path, "w") as f:
        proc = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=str(Path(args.project_root).resolve()),
            env=os.environ.copy(),
            text=True,
        )

    if proc.returncode != 0:
        raise RuntimeError(f"train_one.py failed (returncode={proc.returncode}). See {stdout_path}")

    # Read metrics.json
    metrics_path = trial_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found (train_one.py must write it).")

    metrics = json.loads(metrics_path.read_text())
    if args.metric_name not in metrics:
        raise KeyError(f"'{args.metric_name}' not found in metrics.json. Keys: {list(metrics.keys())}")

    return float(metrics[args.metric_name])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=".", help="Path to my_bci_project")
    ap.add_argument("--base_config", required=True, help="Path to configs/base.yaml")
    ap.add_argument("--run_dir", default="hpo_runs/run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--train_one", default="hpo_scripts/train_one.py")
    ap.add_argument("--python", default="python", help="Python executable to use")
    ap.add_argument("--metric_name", default="val_score")
    ap.add_argument("--direction", choices=["maximize", "minimize"], default="minimize")
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--study_name", default="bci_hpo")
    ap.add_argument("--storage", default=None, help="Optuna storage URL (e.g. sqlite:///.../study.db)")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load base config
    base_cfg = yaml.safe_load(Path(args.base_config).read_text())

    # Default SQLite storage inside run_dir so you can resume
    storage = args.storage or f"sqlite:///{(run_dir / 'optuna_study.db').as_posix()}"

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction=args.direction,
        load_if_exists=True,
    )

    study.optimize(lambda t: objective(t, base_cfg, args), n_trials=args.n_trials)

    # Write a small summary
    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
    }
    (run_dir / "best.json").write_text(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()