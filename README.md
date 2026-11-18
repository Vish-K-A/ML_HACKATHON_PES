# PathSnap: Distributed Map-Matching & Route Reconstruction

**Course**: UE24CS352A — Machine Learning (PES University, Department of CSE, RR Campus)  
**Project**: End-to-End Probabilistic & Sequential Map-Matching System (`Bayesian-Tuned LightGBM + Topological HMM/Viterbi Decoder`)

---

## 1. Repository Structure

The repository is organized into clearly separated, modular packages satisfying all deliverables in the Technical Report & Submission Guidelines:

```text
ML_HACKATHON_PES/
├── data_processing/             # Raw data reading, inspection, WGS-84 metric projection & EDA
│   ├── __init__.py
│   ├── preprocess.py            # WGS-84 tangent-plane projection, multi-scale kinematics, label alignment
│   └── eda.py                   # Dataset inventory, missing-value, gap & road-topology inspection
├── feature_engineering/         # Candidate generation & 27 spatio-temporal/topological features
│   ├── __init__.py
│   └── features.py              # cKDTree candidate retrieval (K=8) & 27-feature extraction pipeline
├── model_training/              # Multi-family ML experimentation (6 models across 6 families)
│   ├── __init__.py
│   └── train_and_compare.py     # 5-Fold GroupKFold CV & Student-Test evaluation across 6 models
├── model_evaluation/            # Candidate-level, observation-level & sequence-level metrics
│   ├── __init__.py
│   └── metrics.py               # Top-1 Accuracy, Macro F1, Log Loss, Route Continuity, Sequence F1
├── hyperparameter_optimisation/ # Bayesian Hyperparameter Optimisation (Optuna TPE)
│   ├── __init__.py
│   └── bayesian_opt.py          # TPE search record, untuned vs. tuned comparison & final model export
├── hmm_viterbi/                 # Sequential route reconstruction via HMM & Viterbi decoding
│   ├── __init__.py
│   └── viterbi_decoder.py       # Topological transition matrix, gap handling, Viterbi & BFS bridging
├── visualisations/              # Automated plotting & interactive Folium map generation
│   ├── __init__.py
│   └── plots_and_maps.py        # EDA figures, model comparison charts, confusion matrices, Folium HTML maps
├── api/                         # Laptop 1: FastAPI Backend & Server-Side Evaluation
│   ├── __init__.py
│   └── main.py                  # Endpoints (/health, /trips, /predict/{trip_id}, /predict)
├── ui/                          # Laptop 2: Evaluator Web Interface
│   ├── __init__.py
│   └── app.py                   # Interactive trip selector, metric dashboard & Folium map viewer
├── deployment/                  # Two-laptop startup scripts & instructions
│   ├── start_laptop1_api.bat    # Starts FastAPI backend on Laptop 1 (0.0.0.0:8000)
│   └── start_laptop2_ui.bat     # Starts Evaluator UI on Laptop 2 (0.0.0.0:8501)
├── models/                      # Saved final model artifact & configuration
│   └── final_model.joblib       # Bayesian-tuned LightGBM classifier + HMM/Viterbi configuration
├── supporting_results/          # CSV experiment logs, JSON summaries, figures & Folium HTML maps
│   ├── dataset_inventory.csv
│   ├── partition_summary.csv
│   ├── data_quality_summary.json
│   ├── model_comparison_table.csv
│   ├── model_comparison_detailed.json
│   ├── bayesian_optimisation_trials.csv
│   ├── bayesian_optimisation_summary.json
│   ├── feature_importances.csv
│   ├── per_trip_student_test_results.csv
│   ├── figures/                 # High-resolution PNG charts & map views for the Technical Report
│   └── maps/                    # Interactive Folium HTML maps for all 10 student_test trips
├── run_pipeline.py              # Single-command reproducible pipeline (EDA -> Train -> Opt -> HMM -> Maps)
└── requirements.txt             # Python dependencies
```

---

## 2. Installation

Ensure Python 3.10+ is installed, then install the required packages:

```bash
pip install -r requirements.txt
```

---

## 3. Reproducing All Experiments & Final Model Training

To run the complete pipeline from scratch (EDA, candidate generation, 27-feature extraction, training and comparing all 6 models across 6 algorithm families, Bayesian hyperparameter optimisation via Optuna TPE, HMM + Viterbi route reconstruction, and generating all plots and Folium maps):

```bash
python run_pipeline.py
```

This saves:
- The single final production model artifact to `models/final_model.joblib`
- All CSV/JSON experimental records to `supporting_results/`
- All comparison figures to `supporting_results/figures/`
- All interactive Folium HTML maps to `supporting_results/maps/`

---

## 4. Two-Laptop Deployment Instructions (Final Evaluation)

### Laptop 1 — Model & FastAPI Backend (`api/main.py`)
1. Find Laptop 1's local Wi-Fi/LAN IP address using `ipconfig` (e.g., `192.168.1.42`).
2. Start the FastAPI server on port `8000`:
   ```bash
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   *(Or double-click `deployment/start_laptop1_api.bat`)*
3. If the evaluator provides a separate folder for the 10 reserved evaluation trips, place it at `Student folder/evaluator_test/` (or set `set PATHSNAP_DATA_DIR=C:\path\to\folder`). The API automatically discovers evaluation trips without exposing ground-truth `.route` labels to the prediction pipeline or UI.
4. Keep the terminal window visible on Laptop 1 to demonstrate live inference logs on `/trips` and `/predict/{trip_id}`.

### Laptop 2 — Evaluator User Interface (`ui/app.py`)
1. Start the UI server on port `8501`:
   ```bash
   python -m uvicorn ui.app:ui_app --host 0.0.0.0 --port 8501
   ```
   *(Or double-click `deployment/start_laptop2_ui.bat`)*
2. Open `http://127.0.0.1:8501` in a browser on Laptop 2.
3. In the top bar, set **Laptop 1 API URL** to `http://<Laptop1_IP>:8000` (defaults to `http://127.0.0.1:8000` when testing on a single machine) and click **Connect & Refresh Trips**.
4. Select any of the 10 evaluation trips to trigger live inference on Laptop 1 and view the reconstructed route, interactive Folium map, and evaluation metrics.
