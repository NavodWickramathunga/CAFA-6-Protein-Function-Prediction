# CAFA-6 Kaggle Starter

Structure for protein function prediction (GO terms) using the competition data.

## Setup
1) Install Python 3.10+.
2) Install deps: `pip install -r requirements.txt` (or create a conda env first).
3) Ensure Kaggle API creds exist at `%USERPROFILE%\.kaggle\kaggle.json` (download from Kaggle > Account > Create New Token).
4) Run data download (once): `python -m src.data.download --dest data/raw`.

## Layout
- `configs/` base paths and run params.
- `src/data/download.py` fetch competition files via Kaggle API.
- `src/data/prepare.py` parsing helpers for GO, FASTA, labels, taxonomy.
- `src/eval/metrics.py` weighted precision/recall/F-max utilities.
- `requirements.txt` main dependencies.

## Next
- Add training scripts under `src/` (e.g., `train.py`).
- Add notebooks under `02_notebook/` for EDA.
- Implement model baselines (k-NN over embeddings, sequence classifier, ontology smoothing).
