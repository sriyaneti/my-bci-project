# hpo_scripts/hpo_optuna.py
import argparse
import json
import os
import subprocess
from datetime import datetime

import optuna
import yaml


def mkdir_p(directory):
    """Equivalent to mkdir -p for Python 3.2"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def deep_update(base, upd):
    """Recursively updates nested dicts."""
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def sample_hparams(trial):
    """
    This function defines the search space of which specific hyperparameters you are
    trying to optimize. It returns a nested dict that will be merged into base.yaml
    """
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


def objective(trial, base_cfg, args):
    # Create trial directory
    trial_name = f"trial_{trial.number:05d}"
    trial_dir = os.path.join(args.run_dir, trial_name)
    mkdir_p(trial_dir)

    # Sample hyperparameters
    hparams_patch = sample_hparams(trial)

    # Merge with base config
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # cheap deep copy
    cfg = deep_update(cfg, hparams_patch)

    # Make outputs unique per trial to avoid overwriting between trials
    cfg["output_dir"] = os.path.join(trial_dir, "model_out")
    cfg["checkpoint_dir"] = os.path.join(trial_dir, "checkpoints")

    # Safety: don't accidentally load weights from some prior run
    cfg["init_from_checkpoint"] = False
    cfg["init_checkpoint_path"] = None

    # Optional: set per-trial seed
    if args.seed is not None:
        cfg["seed"] = int(args.seed) + int(trial.number)

    # Write config.yaml
    cfg_path = os.path.join(trial_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        f.write(yaml.safe_dump(cfg, sort_keys=False))

    # Build command to run train_one.py
    cmd = [
        args.python,
        os.path.abspath(args.train_one),
        "--config", cfg_path,
        "--work_dir", trial_dir,
        "--metric_name", args.metric_name,
    ]

    # Save commands.json
    commands = {"cmd": cmd}
    commands_path = os.path.join(trial_dir, "commands.json")
    with open(commands_path, "w") as f:
        f.write(json.dumps(commands, indent=2))

    # Run subprocess, capture stdout/stderr to stdout.txt
    stdout_path = os.path.join(trial_dir, "stdout.txt")
    with open(stdout_path, "w") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=os.path.abspath(args.project_root),
            env=os.environ.copy(),
        )
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"train_one.py failed (returncode={proc.returncode}). See {stdout_path}")

    # Read metrics.json
    metrics_path = os.path.join(trial_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"{metrics_path} not found (train_one.py must write it).")

    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
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
    ap.add_argument("--metric_name", default="val_PERs")
    ap.add_argument("--direction", choices=["maximize", "minimize"], default="minimize")
    ap.add_argument("--n_trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--study_name", default="bci_hpo")
    ap.add_argument("--storage", default=None, help="Optuna storage URL (e.g. sqlite:///.../study.db)")
    args = ap.parse_args()

    project_root = os.path.abspath(args.project_root)
    run_dir = os.path.abspath(args.run_dir)
    mkdir_p(run_dir)

    # Load base config
    with open(args.base_config, "r") as f:
        base_cfg = yaml.safe_load(f)

    # Resolve dataset_dir to absolute path now, relative to the base config location.
    # If we leave it relative, it will be resolved against the trial directory later
    # (wrong location) instead of the original config directory.
    base_config_dir = os.path.dirname(os.path.abspath(args.base_config))
    ds = base_cfg.get("dataset", {})
    if isinstance(ds, dict) and isinstance(ds.get("dataset_dir"), str):
        ds_dir = ds["dataset_dir"]
        if not os.path.isabs(ds_dir):
            ds["dataset_dir"] = os.path.normpath(os.path.join(base_config_dir, ds_dir))

    # Default SQLite storage inside run_dir so you can resume
    storage = args.storage or "sqlite:///" + os.path.join(run_dir, 'optuna_study.db')

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
    best_path = os.path.join(run_dir, "best.json")
    with open(best_path, "w") as f:
        f.write(json.dumps(best, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
