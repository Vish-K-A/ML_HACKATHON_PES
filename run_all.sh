#!/usr/bin/env bash
# Reproduce every result in results/ from the raw data in data/{train,student_test}
set -e
cd "$(dirname "$0")"
python scripts/01_eda.py
python scripts/02_build_dataset.py
python scripts/03_train_models.py
python scripts/04_bayes_opt.py LightGBM 40
python scripts/05_hmm_experiments.py LightGBM
