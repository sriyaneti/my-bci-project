#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON=${PYTHON:-python}

RUN_DIR="${PROJECT_ROOT}/hpo_runs/run_$(date +%Y%m%d_%H%M%S)"

$PYTHON "${PROJECT_ROOT}/hpo_scripts/hpo_optuna.py" \
  --project_root "$PROJECT_ROOT" \
   --base_config "${PROJECT_ROOT}/../nejm-brain-to-text/model-training/rnn_args.yaml" \
  --run_dir "$RUN_DIR" \
  --n_trials 50 \
  --direction maximize \
  --metric_name val_score