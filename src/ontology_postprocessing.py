"""
Gene Ontology DAG propagation and post-processing.

After predicting GO term scores, we must propagate predictions upward
through the GO hierarchy: if a child term is predicted, all ancestor
terms must also be predicted (true path rule).

Also computes information content (IC) per term for Smin evaluation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import obonet
import networkx as nx

logger = logging.getLogger(__name__)

# Root nodes of the three GO sub-ontologies
GO_ROOTS = {
    "BPO": "GO:0008150",
    "MFO": "GO:0003674",
    "CCO": "GO:0005575",
}


# ---------------------------------------------------------------------------
# OBO loading
# ---------------------------------------------------------------------------

def load_go_dag(obo_path: str | Path) -> nx.MultiDiGraph:
    """
    Load the Gene Ontology OBO file and return a directed graph.

    Edges in obonet point from child → parent (is_a / part_of).
    """
    graph = obonet.read_obo(obo_path)
    logger.info(f"GO DAG loaded: {graph.number_of_nodes():,} terms, {graph.number_of_edges():,} edges")
    return graph


def get_ancestors(graph: nx.MultiDiGraph, term: str) -> set[str]:
    """Return all ancestor terms (transitive closure) of a GO term."""
    try:
        return set(nx.descendants(graph, term))  # obonet: descendants = ancestors in biology
    except nx.NetworkXError:
        return set()


def build_ancestor_map(graph: nx.MultiDiGraph, terms: list[str]) -> dict[str, set[str]]:
    """Pre-compute ancestor sets for all terms in the given list."""
    ancestor_map: dict[str, set[str]] = {}
    for t in terms:
        ancestor_map[t] = get_ancestors(graph, t)
    return ancestor_map


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def propagate_predictions(
    scores: np.ndarray,
    go_terms: list[str],
    ancestor_map: dict[str, set[str]],
    method: str = "max",
) -> np.ndarray:
    """
    Propagate scores upward through the GO hierarchy.

    For each protein, for each predicted term, ensure all ancestor terms
    receive at least the same score (true path rule).

    Args:
        scores: (n_proteins, n_terms) float32 array of prediction scores.
        go_terms: list of GO terms corresponding to score columns.
        ancestor_map: {term: set_of_ancestor_terms} pre-computed.
        method: 'max' (propagate max score) or 'sum' (add parent scores).

    Returns:
        Propagated scores array of the same shape.
    """
    term_to_idx = {t: i for i, t in enumerate(go_terms)}
    propagated = scores.copy()

    for j, term in enumerate(go_terms):
        for ancestor in ancestor_map.get(term, set()):
            if ancestor not in term_to_idx:
                continue
            a_idx = term_to_idx[ancestor]
            if method == "max":
                propagated[:, a_idx] = np.maximum(propagated[:, a_idx], scores[:, j])
            elif method == "sum":
                propagated[:, a_idx] += scores[:, j]

    if method == "sum":
        # Normalise to [0, 1]
        propagated = propagated / (propagated.max(axis=1, keepdims=True) + 1e-9)

    return propagated


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return binary prediction matrix by applying a scalar threshold."""
    return (scores >= threshold).astype(np.int8)


# ---------------------------------------------------------------------------
# Information Content
# ---------------------------------------------------------------------------

def compute_ic(
    annotations_df,
    go_terms: list[str],
    graph: nx.MultiDiGraph | None = None,
    ancestor_map: dict[str, set[str]] | None = None,
) -> np.ndarray:
    """
    Compute information content (IC) for each GO term.

    IC(t) = -log2( freq(t) / n_proteins )
    where freq(t) = number of proteins annotated to t or any descendant.

    Args:
        annotations_df: DataFrame with columns [protein_id, go_term].
        go_terms: list of terms for which to compute IC.
        graph: GO DAG (optional, used for propagation-aware IC).
        ancestor_map: pre-computed ancestor map (optional).

    Returns:
        ic_array: (n_terms,) float32 array.
    """
    prot_terms: dict[str, set[str]] = defaultdict(set)
    for _, row in annotations_df.iterrows():
        prot_terms[row["protein_id"]].add(row["go_term"])

    n_proteins = len(prot_terms)
    if n_proteins == 0:
        return np.zeros(len(go_terms), dtype=np.float32)

    # Count proteins per term (frequency)
    term_freq: dict[str, int] = defaultdict(int)
    for pid, terms in prot_terms.items():
        for t in terms:
            term_freq[t] += 1

    ic = np.zeros(len(go_terms), dtype=np.float32)
    for i, term in enumerate(go_terms):
        freq = term_freq.get(term, 0)
        if freq > 0:
            ic[i] = -np.log2(freq / n_proteins)

    return ic


# ---------------------------------------------------------------------------
# Full post-processing pipeline
# ---------------------------------------------------------------------------

def postprocess_predictions(
    raw_scores: np.ndarray,
    go_terms: list[str],
    obo_path: str | Path | None = None,
    graph: nx.MultiDiGraph | None = None,
    threshold: float = 0.3,
    propagate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply ontology propagation and thresholding to raw prediction scores.

    Args:
        raw_scores: (n_proteins, n_terms) logit or probability scores.
        go_terms: GO term list matching score columns.
        obo_path: path to go.obo (required if graph is None).
        graph: pre-loaded GO DAG.
        threshold: binarisation threshold.
        propagate: whether to apply true-path-rule propagation.

    Returns:
        (propagated_scores, binary_predictions)
    """
    if propagate:
        if graph is None:
            if obo_path is None:
                raise ValueError("Either obo_path or graph must be provided for propagation.")
            graph = load_go_dag(obo_path)
        ancestor_map = build_ancestor_map(graph, go_terms)
        scores = propagate_predictions(raw_scores, go_terms, ancestor_map)
    else:
        scores = raw_scores.copy()

    binary = apply_threshold(scores, threshold)
    return scores, binary


# ---------------------------------------------------------------------------
# CAFA submission format
# ---------------------------------------------------------------------------

def format_cafa_submission(
    protein_ids: list[str],
    go_terms: list[str],
    scores: np.ndarray,
    threshold: float = 0.01,
) -> str:
    """
    Format predictions in the CAFA6 Kaggle submission format.

    Matches sample_submission.tsv: protein_id  go_term  score  (tab-separated, no header)
    Only rows with score >= threshold are emitted.
    """
    lines: list[str] = []
    for i, pid in enumerate(protein_ids):
        for j, term in enumerate(go_terms):
            s = float(scores[i, j])
            if s >= threshold:
                lines.append(f"{pid}\t{term}\t{s:.3f}")
    return "\n".join(lines)
