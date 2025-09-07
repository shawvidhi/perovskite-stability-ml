#!/usr/bin/env bash
set -euo pipefail

python scripts/make_synthetic_dataset.py --n 2500 --seed 42

python -m perostab.train --config configs/model/random_forest.yaml
python -m perostab.train --config configs/model/logreg.yaml

python -m perostab.evaluate --model-path models/rf.joblib
python -m perostab.explain --model-path models/rf.joblib --n-samples 500

echo "Done: data, models, evaluation, and SHAP figures are generated."

