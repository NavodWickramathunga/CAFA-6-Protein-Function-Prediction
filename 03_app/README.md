# CAFA-6 Baseline Application

A simple Streamlit web app that serves the TF-IDF k-mer + cosine-similarity baseline for protein function prediction.

## Prerequisites
- Python 3.10+
- Artifacts exported from the notebook to `05_model/artifacts`:
  - `vectorizer.pkl`
  - `X_train.npz`
  - `train_ids.json`
  - `train_term_map.json`

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r 05_model/requirements.txt
```

## Run the app
```bash
streamlit run 03_app/app.py
```

Paste a sequence or upload a FASTA. Download predictions as a TSV compatible with the competition format.
