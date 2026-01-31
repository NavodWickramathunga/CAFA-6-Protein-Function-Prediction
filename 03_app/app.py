import json
from pathlib import Path

import streamlit as st
import numpy as np
import joblib
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CAFA-6 Function Prediction", layout="wide")
st.title("Protein Function Prediction (CAFA-6 Baseline)")

# Resolve artifacts path: sibling folder 05_model/artifacts
ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "05_model" / "artifacts"

@st.cache_resource
def load_artifacts():
    required = [
        "vectorizer.pkl",
        "X_train.npz",
        "train_ids.json",
        "train_term_map.json",
    ]
    missing = [name for name in required if not (ART / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing artifacts in 05_model/artifacts: "
            + ", ".join(missing)
        )

    vectorizer = joblib.load(ART / "vectorizer.pkl")
    X_train = sparse.load_npz(ART / "X_train.npz")
    train_ids = json.loads((ART / "train_ids.json").read_text())
    train_term_map = json.loads((ART / "train_term_map.json").read_text())
    return vectorizer, X_train, train_ids, train_term_map

try:
    vectorizer, X_train, train_ids, train_term_map = load_artifacts()
except FileNotFoundError as e:
    st.error(str(e))
    st.info(
        "Run the notebook cells that export artifacts to 05_model/artifacts, "
        "then restart the app. Expected files: vectorizer.pkl, X_train.npz, "
        "train_ids.json, train_term_map.json."
    )
    st.stop()

st.sidebar.header("Settings")
K = st.sidebar.slider("Nearest neighbours (K)", min_value=1, max_value=20, value=5)
MAX_TERMS = st.sidebar.slider("Max GO terms per protein", min_value=50, max_value=2000, value=500, step=50)

st.markdown("""
Paste a protein sequence (single line), or upload a FASTA file.
Predictions use TF-IDF k-mers + cosine similarity to labeled training proteins, with
neighbour voting and score normalization.
""")

def kmers(sequence: str, k: int = 3) -> str:
    sequence = sequence.strip().replace("\n", "")
    return " ".join(sequence[i:i+k] for i in range(len(sequence) - k + 1)) if len(sequence) >= k else ""

# FASTA parsing (simple)
def read_fasta_text(text: str):
    seqs = {}
    current_id = None
    parts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                seqs[current_id] = "".join(parts)
            header = line[1:]
            token = header.split()[0]
            current_id = token.split("|")[1] if "|" in token else token
            parts = []
        else:
            parts.append(line)
    if current_id is not None:
        seqs[current_id] = "".join(parts)
    return seqs

col1, col2 = st.columns(2)
with col1:
    input_seq = st.text_area("Paste a single protein sequence", height=150, placeholder="MKTAYIAKQRQISFVKSHFSRQDILD...")
with col2:
    uploaded = st.file_uploader("Or upload FASTA", type=["fasta", "fa", "txt"])    

sequences = {}
if uploaded is not None:
    try:
        text = uploaded.read().decode("utf-8", errors="ignore")
        sequences.update(read_fasta_text(text))
    except Exception as e:
        st.error(f"Failed to parse FASTA: {e}")

if input_seq.strip():
    sequences["input_sequence"] = input_seq.strip().replace("\n", "")

if sequences:
    st.success(f"Loaded {len(sequences)} sequence(s)")
    rows = []

    # Precompute cosine similarity to training set for each input sequence
    for pid, seq in sequences.items():
        text = kmers(seq, k=3)
        if not text:
            st.warning(f"Sequence '{pid}' is too short for k-mers.")
            continue
        x = vectorizer.transform([text])
        sims = cosine_similarity(x, X_train).ravel()
        top_idx = sims.argsort()[-K:][::-1]
        top_sim = sims[top_idx]

        score_map = {}
        for idx, sim in zip(top_idx, top_sim):
            train_pid = train_ids[idx]
            for go in train_term_map.get(train_pid, []):
                score_map[go] = max(score_map.get(go, 0.0), float(sim))

        if not score_map:
            st.warning(f"No neighbour terms found for '{pid}'.")
            continue

        # Normalize and limit to MAX_TERMS
        max_score = max(score_map.values())
        terms_sorted = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:MAX_TERMS]
        for go, sc in terms_sorted:
            prob = sc / max_score if max_score > 0 else 0.0
            prob = max(0.001, min(1.0, prob))
            rows.append([pid, go, round(prob, 3)])

    # Display results
    if rows:
        st.subheader("Predictions")
        import pandas as pd
        df = pd.DataFrame(rows, columns=["protein_id", "go_id", "score"])        
        st.dataframe(df, use_container_width=True, height=400)

        # Download as TSV
        tsv = df.to_csv(sep="\t", index=False, header=False)
        st.download_button("Download TSV", tsv, file_name="submission.tsv", mime="text/tab-separated-values")
else:
    st.info("Provide a sequence or upload a FASTA to get predictions.")
