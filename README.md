# PathSnap: Map-Matching & Route Reconstruction

PathSnap matches noisy GPS observations to a road network and reconstructs the
driven route. A candidate classifier scores the nearby road segments (arcs) for every GPS point;
an HMM / Viterbi decoder then turns those scores into a connected route using
the road-network topology. A FastAPI backend (Laptop 1) runs the full pipeline
for a selected trip and a browser UI (Laptop 2) shows the result.

```
GPS track ──► clean + UTM projection ──► candidate arcs (≤60 m, ≤12) ──► 26 features
          ──► LightGBM P(candidate = true arc) ──► HMM / Viterbi ──► route ──► metrics (server side)
```

## Repository layout

| Path | Contents |
|---|---|
| `pathsnap/` | Library: `io` (raw readers), `geo` (UTM, geometry), `network` (road graph, candidate search, bounded Dijkstra), `labels` (route → per-point labels), `features`, `hmm` (Viterbi + route fill), `metrics`, `pipeline` (inference), `viz` (Folium) |
| `scripts/01_eda.py` | Dataset inventory, data-quality checks, EDA plots, Folium maps |
| `scripts/02_build_dataset.py` | Candidate generation, feature tables, label alignment, candidate radius/K sensitivity |
| `scripts/03_train_models.py` | 9 models from 7 families, 5-fold GroupKFold CV, student-test comparison, plots |
| `scripts/04_bayes_opt.py` | Optuna (TPE) Bayesian optimisation of the selected model, tuned vs untuned |
| `scripts/05_hmm_experiments.py` | Decoder tuning, classifier vs HMM, gap robustness, maps, final artifact export |
| `scripts/06_build_report.py` | Builds `report/PathSnap_Technical_Report.pdf` from `report/report.md` + results CSVs |
| `api/main.py` | FastAPI backend |
| `ui/index.html` | Evaluator UI (Leaflet), single static file |
| `models/` | `pathsnap_final.joblib` (model + feature list + HMM parameters), tuned model, config JSON |
| `results/` | All experiment outputs: CSVs, logs, plots, Optuna study (SQLite + CSV), Folium maps |
| `report/` | Report source, generated PDF, screenshots |

## Installation

Python 3.11+ (developed on 3.13, Windows 11).

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The raw data is supplied by the organisers and is not committed. Place it as:

```
data/
  train/<trip_id>/<trip_id>.track|.nodes|.arcs|.route      (80 trips)
  student_test/<trip_id>/...                               (10 trips)
  evaluator_test/<trip_id>/...                             (10 trips, final evaluation only)
```

`PATHSNAP_DATA` can point to a different data folder.

## Reproducing the experiments

```bash
python scripts/01_eda.py                     # ~2 min
python scripts/02_build_dataset.py           # ~2 min
python scripts/03_train_models.py            # ~75 min (all 9 models, 5-fold CV)
python scripts/04_bayes_opt.py LightGBM 40   # ~45 min
python scripts/05_hmm_experiments.py LightGBM  # ~15 min, writes models/pathsnap_final.joblib
python scripts/06_build_report.py            # report PDF
```
or `bash run_all.sh`. All scripts use seed 42. Only `train` and
`student_test` are read by these scripts; `evaluator_test` is never used for
training, feature selection, model selection or decoder tuning.

## Deployment (two laptops)

**Laptop 1: API.** Needs the repo, `models/pathsnap_final.joblib` and the
`data/` folder (the evaluation trips with their reference routes, kept on the server).

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000      # or run_api.bat
```
The API serves `data/evaluator_test` if that folder exists, otherwise
`data/student_test` (override with `PATHSNAP_EVAL_PARTITION`). Each request is
logged in the console and in `results/api_requests.log`. Find Laptop 1's IP with
`ipconfig` and allow port 8000 through the Windows firewall if asked.

**Laptop 2: UI.** Only `ui/index.html` is needed.

```bash
cd ui && python -m http.server 5500                              # or run_ui.bat
```
Open `http://localhost:5500`, enter `http://<laptop-1-ip>:8000` and press
**Connect**. The ten evaluation trips are listed; selecting one sends only the
trip ID to the API, which runs the whole pipeline and returns the predicted route
and server-side metrics. The reference route never leaves the API. For a single machine test, open
`http://localhost:8000/` (the API also serves the UI).

### API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | model name, HMM parameters, evaluation partition |
| GET | `/trips` | evaluation trip IDs, point counts, durations |
| POST | `/predict` | body `{"trip_id": "00000009"}` → GPS points, predicted route, classifier-only route, metrics, timings |

## Method summary

* **Labels.** `.route` is one ordered arc list per trip; a monotone DP alignment
  assigns each GPS point a route arc (order-preserving, handles loops).
* **Candidates.** Arcs within 60 m (max 12) via a KD-tree on densified arcs;
  fallback to the nearest arcs when nothing is within range.
* **Features (26).** Distance and its rank / gap / ratio to the nearest
  candidate, Gaussian share, projection position, arc length, heading difference
  (forward/reverse) with speed reliability, two-way flag, distances of the ±3
  neighbouring observations to the arc, node degrees, time gaps.
* **Models compared.** Logistic Regression, Gaussian NB, KNN, Decision Tree,
  Random Forest, Extra Trees, HistGradientBoosting, LightGBM, MLP.
* **Selection.** Primary metric: observation-level Top-1 accuracy; also macro
  F1, log loss (HMM emissions need calibrated probabilities), continuity, speed.
* **HMM.** States = candidates; emission = normalised classifier probability^w;
  transition = exp(−|route distance − GPS distance| / β) using directed shortest
  paths; chain breaks on gaps > 120 s or unreachable steps.

Detailed results are in `report/PathSnap_Technical_Report.pdf` and `results/`.
