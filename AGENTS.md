# PathSnap model implementation notes

## Goal and time limit

Build one lightweight candidate-ranking model with Extra Trees. Prioritize a correct end-to-end baseline and reproducible outputs over HMM decoding or broad hyperparameter search. Never run training without the user's direct instruction; give the user the command and let them start it locally.

## Data layout

Raw data is stored outside this Git repository at `../data/{train,test}/<trip_id>/`. A trip directory contains matching `.track`, `.nodes`, `.arcs`, and `.route` files. `.track` has longitude, latitude, timestamp; `.nodes` has longitude, latitude; `.arcs` contains endpoint node IDs, with the row index serving as arc ID; `.route` is an ordered list of arc IDs.

Use only `../data/train` for training and grouped validation. The current `../data/test` contains 10 trips; treat it as evaluator data and do not inspect its labels or use it for tuning unless official instructions confirm it is a development partition.

## Data and label rules

- Parse whitespace-delimited numeric files without headers.
- Validate all arc endpoints and route arc IDs before building features.
- GPS and route lengths can differ. Never join them by row index. The first implementation estimates per-observation labels by nearest route geometry with nondecreasing route order. Record and inspect alignment distances. Treat poor alignment as a blocker for interpreting classifier metrics.
- Convert coordinates to a local metric CRS before measuring distances or headings. Do not compute Euclidean distances in longitude/latitude degrees.
- Candidate generation must be bounded with a spatial index and capped number of candidates per observation.
- Report candidate recall/miss rate. If the reference arc is absent, the classifier cannot predict it.

## Model and validation

- Each training row represents one GPS observation and one candidate arc.
- Target is 1 for the estimated reference arc and 0 otherwise.
- Use a trip-grouped holdout; do not split candidate rows randomly.
- Keep one deterministic seed and conservative CPU use (`n_jobs=2`). Start with Extra Trees defaults appropriate for a fast baseline; do not launch concurrent searches.
- Evaluate observation-level top-1 accuracy, macro F1, candidate recall, and route-alignment distance summaries. Separate candidate-row and observation-level metrics.
- Save model, metadata, split IDs, and metrics to `artifacts/` and `experiments/`; raw data stays outside Git.

## Safe execution sequence

1. Inspect parser and alignment behavior on one or two training trips.
2. Build candidate tables from the training partition only.
3. Review alignment distances and candidate recall before fitting.
4. Fit the grouped-holdout model only after the user explicitly runs the provided command.
5. Defer HMM/Viterbi, Bayesian optimization, UI/API work, and evaluator-test scoring until the basic result is stable and authorized by the rules.

## Dependencies

Install from `requirements.txt` in a project virtual environment. Do not install packages or execute training as part of file edits unless the user asks.
