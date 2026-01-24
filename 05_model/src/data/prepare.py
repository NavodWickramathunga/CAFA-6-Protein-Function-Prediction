"""Parsing utilities for CAFA-6 data."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd
import networkx as nx
import obonet
from Bio import SeqIO

SubOntology = Dict[str, Set[str]]


def load_go(obo_path: Path) -> nx.MultiDiGraph:
    """Load GO graph from OBO file."""
    return obonet.read_obo(str(obo_path))


def get_ancestors(go_graph: nx.MultiDiGraph, term: str) -> Set[str]:
    """Return ancestors including self for a GO term."""
    ancestors = nx.ancestors(go_graph, term)
    ancestors.add(term)
    return ancestors


def load_terms(terms_path: Path) -> pd.DataFrame:
    """Load train_terms.tsv (columns: protein, term, namespace)."""
    df = pd.read_csv(terms_path, sep="\t", header=None, names=["protein_id", "term_id", "namespace"])
    return df


def load_taxonomy(tax_path: Path) -> pd.DataFrame:
    return pd.read_csv(tax_path, sep="\t", header=None, names=["protein_id", "taxon_id"])


def load_sequences(fasta_path: Path) -> Dict[str, str]:
    """Return dict of UniProt accession -> sequence."""
    records = SeqIO.parse(str(fasta_path), "fasta")
    seqs = {}
    for rec in records:
        # Header pattern: sp|ACC|... or tr|ACC|...
        acc = rec.id.split("|")[1] if "|" in rec.id else rec.id
        seqs[acc] = str(rec.seq)
    return seqs


def load_ia_weights(ia_path: Path) -> pd.Series:
    """Load IA.tsv (term_id, ia_weight)."""
    df = pd.read_csv(ia_path, sep="\t", header=None, names=["term_id", "ia"])
    return pd.Series(df.ia.values, index=df.term_id.values)


def propagate_labels(df_terms: pd.DataFrame, go_graph: nx.MultiDiGraph) -> pd.DataFrame:
    """Propagate each term to its ancestors; returns expanded rows."""
    rows: List[Tuple[str, str, str]] = []
    for protein, term, ns in df_terms.itertuples(index=False):
        for anc in get_ancestors(go_graph, term):
            rows.append((protein, anc, ns))
    return pd.DataFrame(rows, columns=["protein_id", "term_id", "namespace"])


def split_by_namespace(df_terms: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    grouped = {}
    for ns in ["BPO", "CCO", "MFO"]:
        grouped[ns] = df_terms[df_terms["namespace"] == ns].copy()
    return grouped
