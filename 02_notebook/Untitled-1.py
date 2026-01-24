python - <<'PY'
from pathlib import Path
from src.data.prepare import load_sequences, load_terms, load_taxonomy
root = Path('../cafa-6-protein-function-prediction/Train')
seqs = load_sequences(root/'train_sequences.fasta')
terms = load_terms(root/'train_terms.tsv')
tax = load_taxonomy(root/'train_taxonomy.tsv')
print('sequences:', len(seqs))
print('terms rows:', len(terms))
print('taxonomy rows:', len(tax))
PY