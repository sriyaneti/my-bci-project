#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Override NEJM_ROOT if it is not a sibling of the project root (e.g. on the cluster)
export NEJM_ROOT="${NEJM_ROOT:-${PROJECT_ROOT}/../nejm-brain-to-text}"

PYTHON=${PYTHON:-python}

RUN_DIR="${PROJECT_ROOT}/hpo_runs/run_$(date +%Y%m%d_%H%M%S)"

$PYTHON "${PROJECT_ROOT}/hpo_scripts/hpo_optuna.py" \
  --project_root "$PROJECT_ROOT" \
   --base_config "${PROJECT_ROOT}/../nejm-brain-to-text/model_training/rnn_args.yaml" \
  --run_dir "$RUN_DIR" \
  --n_trials 50 \
  --direction minimize \
  --metric_name val_PERs