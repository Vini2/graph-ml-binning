"""Shared utilities for the graph-ML contig binning notebooks.

The notebooks intentionally keep the model-specific cells small. This module
does the SPAdes graph loading, RepBin-style contig feature construction,
unsupervised embedding training, clustering, and evaluation.
"""

from __future__ import annotations

import itertools
import os
import random
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import numpy as np
import pandas as pd
from Bio import SeqIO
from agtools.assemblers import spades
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment


DNA = "ACGT"


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def canonical_tetranucleotides() -> list[str]:
    """Return the 136 reverse-complement-collapsed 4-mer feature names."""
    kmers = ("".join(kmer) for kmer in itertools.product(DNA, repeat=4))
    return sorted({min(kmer, reverse_complement(kmer)) for kmer in kmers})


TETRA_FEATURES = canonical_tetranucleotides()
TETRA_INDEX = {kmer: idx for idx, kmer in enumerate(TETRA_FEATURES)}


def parse_coverage(contig_id: str) -> float:
    match = re.search(r"_cov_([0-9.]+)", contig_id)
    if not match:
        raise ValueError(f"Could not parse coverage from contig id: {contig_id}")
    return float(match.group(1))


def parse_length(contig_id: str) -> int:
    match = re.search(r"_length_([0-9]+)", contig_id)
    if not match:
        raise ValueError(f"Could not parse length from contig id: {contig_id}")
    return int(match.group(1))


def tetranucleotide_frequencies(sequence: str) -> np.ndarray:
    sequence = sequence.upper()
    counts = np.zeros(len(TETRA_FEATURES), dtype=np.float32)
    total = 0
    for i in range(max(0, len(sequence) - 3)):
        kmer = sequence[i : i + 4]
        if set(kmer) <= set(DNA):
            counts[TETRA_INDEX[min(kmer, reverse_complement(kmer))]] += 1.0
            total += 1
    if total:
        counts /= total
    return counts


def zscore_column(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    std = values.std()
    if std == 0:
        return np.zeros_like(values, dtype=np.float32)
    return (values - values.mean()) / std


def short_contig_name(contig_id: str) -> str:
    match = re.search(r"NODE_\d+", contig_id)
    if not match:
        raise ValueError(f"Could not parse short contig name from: {contig_id}")
    return match.group(0)


def load_marker_sets(marker_file: str | Path, contig_names: list[str]) -> tuple[list[list[int]], np.ndarray, int]:
    """Parse RepBin-style marker sets and derive cannot-link pairs plus k.

    Each line in ``contigs.fasta.markers`` is a marker-gene family. Contigs in
    the same line should usually come from different bins, so RepBin samples
    pairwise constraints from these lines and pushes their embeddings apart.
    The number of genomes/bins is estimated from the typical marker-set size.
    """
    short_to_idx = {short_contig_name(name): idx for idx, name in enumerate(contig_names)}
    marker_sets = []
    with Path(marker_file).open() as handle:
        for line in handle:
            if ":" not in line:
                continue
            _, values = line.strip().split(":", 1)
            ids = []
            for value in values.split(","):
                short_name = value.strip()
                if short_name in short_to_idx:
                    ids.append(short_to_idx[short_name])
            ids = sorted(set(ids))
            if len(ids) >= 2:
                marker_sets.append(ids)

    pairs = []
    for marker_set in marker_sets:
        for left, right in combinations(marker_set, 2):
            pairs.append((left, right))
            pairs.append((right, left))
    marker_pairs = np.array(pairs, dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)
    marker_cluster_count = max((len(marker_set) for marker_set in marker_sets), default=1)
    return marker_sets, marker_pairs, marker_cluster_count


def load_spades_dataset(data_dir: str | Path = "../tests/data", labeled_only: bool = True):
    """Load the example SPAdes graph and construct contig feature matrices."""
    data_dir = Path(data_dir)
    graph_file = data_dir / "assembly_graph_with_scaffolds.gfa"
    contigs_file = data_dir / "contigs.fasta"
    contig_paths_file = data_dir / "contigs.paths"
    ground_truth_file = data_dir / "ground_truth.csv"
    marker_file = data_dir / "contigs.fasta.markers"

    cg = spades.get_contig_graph(str(graph_file), str(contigs_file), str(contig_paths_file))
    adjacency = np.asarray(cg.get_adjacency_matrix().data, dtype=np.float32)

    sequences = {record.id: str(record.seq) for record in SeqIO.parse(contigs_file, "fasta")}
    tetra = np.vstack([tetranucleotide_frequencies(sequences[name]) for name in cg.contig_names])
    coverage = np.array([parse_coverage(name) for name in cg.contig_names], dtype=np.float32)
    lengths = np.array([parse_length(name) for name in cg.contig_names], dtype=np.int64)
    log_coverage = np.log1p(coverage)
    coverage_feature = zscore_column(log_coverage)[:, None]
    sequence_features = np.hstack([tetra, coverage_feature]).astype(np.float32)

    raw_labels = {}
    with ground_truth_file.open() as handle:
        for line in handle:
            contig, label = line.strip().split(",", 1)
            raw_labels[contig] = label
    if labeled_only:
        keep_idx = [idx for idx, name in enumerate(cg.contig_names) if name in raw_labels]
        contig_names = [cg.contig_names[idx] for idx in keep_idx]
        adjacency = adjacency[np.ix_(keep_idx, keep_idx)]
        sequence_features = sequence_features[keep_idx]
        coverage = coverage[keep_idx]
        lengths = lengths[keep_idx]
    else:
        missing = [name for name in cg.contig_names if name not in raw_labels]
        if missing:
            raise ValueError(
                f"{len(missing)} contigs are missing from {ground_truth_file}. "
                "Use labeled_only=True for example-dataset evaluation."
            )
        contig_names = cg.contig_names

    adjacency_features = adjacency.astype(np.float32)
    features = np.hstack([adjacency_features, sequence_features]).astype(np.float32)
    feature_names = (
        [f"adjacency_to_node_{idx}" for idx in range(adjacency.shape[0])]
        + TETRA_FEATURES
        + ["log1p_coverage_zscore"]
    )
    marker_sets, marker_pairs, marker_cluster_count = load_marker_sets(marker_file, contig_names)

    label_names = sorted(set(raw_labels.values()))
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    labels = np.array([label_to_id[raw_labels[name]] for name in contig_names], dtype=np.int64)

    return {
        "contig_graph": cg,
        "contig_names": contig_names,
        "adjacency": adjacency,
        "features": features,
        "adjacency_features": adjacency_features,
        "sequence_features": sequence_features,
        "coverage": coverage,
        "lengths": lengths,
        "marker_sets": marker_sets,
        "marker_pairs": marker_pairs,
        "marker_cluster_count": marker_cluster_count,
        "labels": labels,
        "label_names": label_names,
        "feature_names": feature_names,
    }


def normalized_adjacency(adjacency: np.ndarray, mode: str = "symmetric") -> np.ndarray:
    matrix = adjacency.astype(np.float32).copy()
    matrix += np.eye(matrix.shape[0], dtype=np.float32)
    degree = matrix.sum(axis=1)
    degree[degree == 0] = 1.0
    if mode == "mean":
        return matrix / degree[:, None]
    inv_sqrt = 1.0 / np.sqrt(degree)
    return inv_sqrt[:, None] * matrix * inv_sqrt[None, :]


def set_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def evaluate_predictions(labels: np.ndarray, predictions: np.ndarray, label_names=None) -> dict:
    precision, recall, f1 = cluster_precision_recall_f1(labels, predictions)
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ari": adjusted_rand_score(labels, predictions),
        "nmi": normalized_mutual_info_score(labels, predictions),
        "n_bins_predicted": int(len(set(predictions.tolist()))),
        "n_true_bins": int(len(set(labels.tolist()))),
    }
    metrics["aligned_macro_f1"] = aligned_macro_f1(labels, predictions)
    return metrics


def cluster_precision_recall_f1(labels: np.ndarray, clusters: np.ndarray) -> tuple[float, float, float]:
    """RepBin-style cluster precision, recall, and F1 from a contingency table.

    Precision sums each predicted bin's dominant true-label count and divides by
    total binned contigs. Recall sums each true label's best matching predicted
    bin count and divides by all labelled contigs.
    """
    true_ids = np.unique(labels)
    cluster_ids = np.unique(clusters)
    contingency = np.zeros((len(cluster_ids), len(true_ids)), dtype=np.int64)
    for row, cluster_id in enumerate(cluster_ids):
        for col, true_id in enumerate(true_ids):
            contingency[row, col] = np.sum((clusters == cluster_id) & (labels == true_id))

    total = contingency.sum()
    if total == 0:
        return 0.0, 0.0, 0.0
    precision = contingency.max(axis=1).sum() / total
    recall = contingency.max(axis=0).sum() / total
    f1 = 0.0 if precision == 0.0 or recall == 0.0 else 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def aligned_macro_f1(labels: np.ndarray, clusters: np.ndarray) -> float:
    """Macro-F1 after optimally mapping arbitrary cluster IDs to true label IDs."""
    true_ids = np.unique(labels)
    cluster_ids = np.unique(clusters)
    contingency = np.zeros((len(cluster_ids), len(true_ids)), dtype=np.int64)
    for row, cluster_id in enumerate(cluster_ids):
        for col, true_id in enumerate(true_ids):
            contingency[row, col] = np.sum((clusters == cluster_id) & (labels == true_id))
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {cluster_ids[row]: true_ids[col] for row, col in zip(row_ind, col_ind)}
    fallback = true_ids[np.argmax(np.bincount(labels))]
    aligned = np.array([mapping.get(cluster, fallback) for cluster in clusters], dtype=np.int64)
    return f1_score(labels, aligned, average="macro")


def print_dataset_summary(data: dict):
    print(f"contigs: {len(data['contig_names'])}")
    print(f"edges: {int(data['adjacency'].sum() // 2)}")
    print(
        f"features: {data['features'].shape[1]} "
        f"({data['adjacency_features'].shape[1]} adjacency + {len(TETRA_FEATURES)} TNF + coverage)"
    )
    print(f"marker sets: {len(data['marker_sets'])}")
    print(f"marker-derived k: {data['marker_cluster_count']}")
    counts = Counter(data["labels"])
    for label_id, count in sorted(counts.items()):
        print(f"{data['label_names'][label_id]}: {count}")


def train_torch_graph_autoencoder(
    model,
    features: np.ndarray,
    adjacency: np.ndarray,
    marker_pairs: np.ndarray | None = None,
    *,
    epochs=400,
    learning_rate=0.01,
    weight_decay=5e-4,
    patience=40,
    constraint_weight=0.25,
):
    """Train an unsupervised graph autoencoder and return node embeddings.

    Labels are deliberately not used here. The model learns contig embeddings
    that reconstruct the SPAdes contig graph; clustering and evaluation happen
    after training, mirroring RepBin's embed-then-cluster workflow.
    """
    import torch
    import torch.nn.functional as F

    x = torch.tensor(features, dtype=torch.float32)
    adj = torch.tensor(adjacency, dtype=torch.float32)
    pairs = None
    if marker_pairs is not None and len(marker_pairs) > 0:
        pairs = torch.tensor(marker_pairs, dtype=torch.long)
    target = torch.tensor(adjacency, dtype=torch.float32)
    target = torch.maximum(target, torch.eye(target.shape[0]))
    positives = target.sum()
    negatives = target.numel() - positives
    pos_weight = negatives / positives.clamp_min(1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = None
    best_loss = float("inf")
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        embeddings = model(x, adj)
        logits = embeddings @ embeddings.T
        reconstruction_loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        if pairs is not None:
            left = embeddings[pairs[:, 0]]
            right = embeddings[pairs[:, 1]]
            constraint_loss = torch.exp(-F.pairwise_distance(left, right, p=2)).mean()
            loss = reconstruction_loss + constraint_weight * constraint_loss
        else:
            constraint_loss = torch.tensor(0.0)
            loss = reconstruction_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings = model(x, adj)
            logits = embeddings @ embeddings.T
            reconstruction_loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            if pairs is not None:
                left = embeddings[pairs[:, 0]]
                right = embeddings[pairs[:, 1]]
                constraint_loss = torch.exp(-F.pairwise_distance(left, right, p=2)).mean()
                current_loss = (reconstruction_loss + constraint_weight * constraint_loss).item()
                current_constraint_loss = constraint_loss.item()
            else:
                current_loss = reconstruction_loss.item()
                current_constraint_loss = 0.0

        history.append(
            {
                "epoch": epoch,
                "loss": current_loss,
                "reconstruction_loss": float(reconstruction_loss.item()),
                "marker_constraint_loss": current_constraint_loss,
            }
        )
        if current_loss < best_loss - 1e-5:
            best_loss = current_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        embeddings = model(x, adj).cpu().numpy()
    return embeddings, history


def run_kmeans(embeddings: np.ndarray, labels: np.ndarray, seed=7, n_clusters: int | None = None):
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    n_clusters = max(1, min(int(n_clusters), len(labels)))
    predictions = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(embeddings)
    return predictions, evaluate_predictions(labels, predictions)


METRIC_COLUMNS = ["precision", "recall", "f1", "ari", "nmi", "aligned_macro_f1"]
ABLATION_ORDER = [
    "original assembly graph",
    "random edge removal",
    "random edge addition",
    "coverage-similarity edges added",
    "short contigs removed",
]


def run_repeated_experiment(run_once, n_runs=10, seed=7) -> pd.DataFrame:
    """Run a notebook's embedding/clustering experiment repeatedly."""
    rows = []
    for run_idx in range(n_runs):
        run_seed = seed + run_idx
        metrics = run_once(run_seed)
        row = {"run": run_idx + 1, "seed": run_seed}
        row.update(metrics)
        rows.append(row)
        metric_text = ", ".join(f"{metric}={row[metric]:.4f}" for metric in ["precision", "recall", "f1", "ari", "nmi"])
        print(f"run {run_idx + 1:02d} seed={run_seed}: {metric_text}")
    return pd.DataFrame(rows)


def summarize_metric_table(results: pd.DataFrame, metric_columns: list[str] | None = None) -> pd.DataFrame:
    """Return min/max/mean/std for selected metrics."""
    if metric_columns is None:
        metric_columns = [metric for metric in METRIC_COLUMNS if metric in results.columns]
    summary = results[metric_columns].agg(["min", "max", "mean", "std"]).T
    return summary[["min", "max", "mean", "std"]]


def plot_metric_bars(summary: pd.DataFrame, title: str = "Metrics over 10 runs"):
    """Plot mean metric values with standard-deviation error bars."""
    import matplotlib.pyplot as plt

    ax = summary["mean"].plot(
        kind="bar",
        yerr=summary["std"],
        capsize=4,
        figsize=(9, 4),
        color="#4C78A8",
        edgecolor="#2F3A4A",
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return ax


def remap_marker_sets(marker_sets: list[list[int]], keep_idx: np.ndarray | list[int], n_nodes: int) -> tuple[list[list[int]], np.ndarray, int]:
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    old_to_new = {int(old): new for new, old in enumerate(keep_idx)}
    remapped_sets = []
    for marker_set in marker_sets:
        remapped = sorted({old_to_new[idx] for idx in marker_set if idx in old_to_new})
        if len(remapped) >= 2:
            remapped_sets.append(remapped)
    pairs = []
    for marker_set in remapped_sets:
        for left, right in combinations(marker_set, 2):
            pairs.append((left, right))
            pairs.append((right, left))
    marker_pairs = np.array(pairs, dtype=np.int64) if pairs else np.empty((0, 2), dtype=np.int64)
    marker_cluster_count = max((len(marker_set) for marker_set in remapped_sets), default=max(1, min(n_nodes, 1)))
    return remapped_sets, marker_pairs, marker_cluster_count


def rebuild_features_for_graph(data: dict, adjacency: np.ndarray, keep_idx: np.ndarray | list[int] | None = None) -> dict:
    """Create a dataset view after graph corruption or node filtering."""
    if keep_idx is None:
        keep_idx = np.arange(len(data["contig_names"]))
    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    adjacency = adjacency.astype(np.float32)
    sequence_features = data["sequence_features"][keep_idx].astype(np.float32)
    features = np.hstack([adjacency, sequence_features]).astype(np.float32)
    marker_sets, marker_pairs, marker_cluster_count = remap_marker_sets(data["marker_sets"], keep_idx, adjacency.shape[0])
    view = {
        **data,
        "contig_names": [data["contig_names"][idx] for idx in keep_idx],
        "adjacency": adjacency,
        "features": features,
        "adjacency_features": adjacency,
        "sequence_features": sequence_features,
        "coverage": data["coverage"][keep_idx],
        "lengths": data["lengths"][keep_idx],
        "labels": data["labels"][keep_idx],
        "marker_sets": marker_sets,
        "marker_pairs": marker_pairs,
        "marker_cluster_count": marker_cluster_count,
        "feature_names": [f"adjacency_to_node_{idx}" for idx in range(adjacency.shape[0])] + TETRA_FEATURES + ["log1p_coverage_zscore"],
    }
    return view


def remove_random_edges(adjacency: np.ndarray, fraction=0.15, seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = adjacency.astype(np.float32).copy()
    edges = np.transpose(np.nonzero(np.triu(matrix, k=1)))
    if len(edges) == 0:
        return matrix
    n_remove = int(round(len(edges) * fraction))
    if n_remove == 0:
        return matrix
    chosen = rng.choice(len(edges), size=n_remove, replace=False)
    for i, j in edges[chosen]:
        matrix[i, j] = 0.0
        matrix[j, i] = 0.0
    return matrix


def add_random_edges(adjacency: np.ndarray, fraction=0.15, seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = adjacency.astype(np.float32).copy()
    existing = int(np.triu(matrix, k=1).sum())
    n_add = max(1, int(round(existing * fraction)))
    candidates = np.transpose(np.nonzero(np.triu(1 - matrix - np.eye(matrix.shape[0]), k=1)))
    if len(candidates) == 0:
        return matrix
    chosen = rng.choice(len(candidates), size=min(n_add, len(candidates)), replace=False)
    for i, j in candidates[chosen]:
        matrix[i, j] = 1.0
        matrix[j, i] = 1.0
    return matrix


def add_coverage_similarity_edges(adjacency: np.ndarray, coverage: np.ndarray, top_k=2) -> np.ndarray:
    matrix = adjacency.astype(np.float32).copy()
    log_cov = np.log1p(coverage.astype(np.float32))
    for i in range(matrix.shape[0]):
        candidates = np.where((matrix[i] == 0) & (np.arange(matrix.shape[0]) != i))[0]
        if len(candidates) == 0:
            continue
        distances = np.abs(log_cov[candidates] - log_cov[i])
        nearest = candidates[np.argsort(distances)[:top_k]]
        matrix[i, nearest] = 1.0
        matrix[nearest, i] = 1.0
    return matrix


def remove_short_contigs(data: dict, quantile=0.10) -> dict:
    threshold = np.quantile(data["lengths"], quantile)
    keep_idx = np.where(data["lengths"] > threshold)[0]
    if len(keep_idx) == 0:
        keep_idx = np.arange(len(data["lengths"]))
    adjacency = data["adjacency"][np.ix_(keep_idx, keep_idx)]
    return rebuild_features_for_graph(data, adjacency, keep_idx)


def make_graph_ablation_datasets(
    data: dict,
    seed=7,
    edge_fraction=0.15,
    coverage_top_k=2,
    short_contig_quantile=0.10,
) -> dict[str, dict]:
    """Return original and corrupted graph views for robustness studies."""
    original = rebuild_features_for_graph(data, data["adjacency"])
    removed = rebuild_features_for_graph(data, remove_random_edges(data["adjacency"], edge_fraction, seed))
    added = rebuild_features_for_graph(data, add_random_edges(data["adjacency"], edge_fraction, seed))
    cov_added = rebuild_features_for_graph(data, add_coverage_similarity_edges(data["adjacency"], data["coverage"], coverage_top_k))
    no_short = remove_short_contigs(data, short_contig_quantile)
    return {
        ABLATION_ORDER[0]: original,
        ABLATION_ORDER[1]: removed,
        ABLATION_ORDER[2]: added,
        ABLATION_ORDER[3]: cov_added,
        ABLATION_ORDER[4]: no_short,
    }


def run_repeated_experiment_on_data(run_once, ablation_data: dict, n_runs=10, seed=7) -> pd.DataFrame:
    rows = []
    for run_idx in range(n_runs):
        run_seed = seed + run_idx
        metrics = run_once(run_seed, ablation_data)
        row = {"run": run_idx + 1, "seed": run_seed}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def run_graph_ablation_study(data: dict, run_once, n_runs=10, seed=7) -> tuple[pd.DataFrame, pd.DataFrame]:
    variants = make_graph_ablation_datasets(data, seed=seed)
    all_results = []
    for ablation_name, ablation_data in variants.items():
        print(f"ablation: {ablation_name}")
        results = run_repeated_experiment_on_data(run_once, ablation_data, n_runs=n_runs, seed=seed)
        results.insert(0, "ablation", ablation_name)
        all_results.append(results)
    combined = pd.concat(all_results, ignore_index=True)
    combined["ablation"] = pd.Categorical(combined["ablation"], categories=ABLATION_ORDER, ordered=True)
    metric_columns = [metric for metric in METRIC_COLUMNS if metric in combined.columns]
    summary = (
        combined.groupby("ablation", observed=False)[metric_columns]
        .agg(["min", "max", "mean", "std"])
        .reindex(ABLATION_ORDER)
    )
    return combined, summary


def plot_ablation_metric_bars(
    ablation_summary: pd.DataFrame,
    metrics: list[str] | None = None,
    title: str = "Graph ablation metrics",
):
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = ["precision", "recall", "f1", "ari", "nmi"]
    means = pd.DataFrame({metric: ablation_summary[(metric, "mean")] for metric in metrics})
    stds = pd.DataFrame({metric: ablation_summary[(metric, "std")] for metric in metrics})
    ax = means.plot(kind="bar", yerr=stds, capsize=3, figsize=(12, 5), width=0.82)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return ax


def plot_assembly_graph_bins(
    data: dict,
    bins: np.ndarray | None = None,
    title: str = "Assembly graph contigs coloured by predicted bin",
    bbox=(900, 700),
):
    """Plot contigs in the assembly graph with igraph, coloured by predicted bin."""
    import igraph as ig

    adjacency = data["adjacency"]
    if bins is None:
        bins = data["labels"]
    bins = np.asarray(bins)

    graph = ig.Graph.Adjacency((adjacency > 0).tolist(), mode="undirected")
    graph.simplify(multiple=True, loops=True)

    unique_bins = sorted(np.unique(bins).tolist())
    palette = ig.RainbowPalette(n=max(1, len(unique_bins)))
    colour_map = {bin_id: palette.get(idx) for idx, bin_id in enumerate(unique_bins)}
    graph.vs["color"] = [colour_map[bin_id] for bin_id in bins]
    graph.vs["size"] = 5
    graph.vs["label"] = ["" for _ in range(graph.vcount())]
    graph.es["color"] = "rgba(110,110,110,0.25)"
    graph.es["width"] = 0.5

    layout = graph.layout_fruchterman_reingold()
    try:
        plot = ig.plot(
            graph,
            layout=layout,
            bbox=bbox,
            margin=35,
            vertex_color=graph.vs["color"],
            vertex_size=graph.vs["size"],
            vertex_label=graph.vs["label"],
            edge_color=graph.es["color"],
            edge_width=graph.es["width"],
        )
    except AttributeError as exc:
        raise ImportError(
            "igraph plotting requires pycairo or cairocffi. "
            "Create/update the conda environment from environment.yml, which includes pycairo."
        ) from exc
    print(title)
    counts = Counter(bins.tolist())
    for bin_id in unique_bins:
        print(f"bin {bin_id}: {counts[bin_id]} contigs")
    return plot


def ppmi_deepwalk_embeddings(adjacency: np.ndarray, dimensions=32, walk_length=20, walks_per_node=12, window_size=5, seed=7):
    rng = np.random.default_rng(seed)
    n_nodes = adjacency.shape[0]
    neighbors = [np.flatnonzero(adjacency[i]).tolist() for i in range(n_nodes)]
    cooc = np.zeros((n_nodes, n_nodes), dtype=np.float32)

    for start in range(n_nodes):
        for _ in range(walks_per_node):
            walk = [start]
            current = start
            for _ in range(walk_length - 1):
                if not neighbors[current]:
                    break
                current = rng.choice(neighbors[current])
                walk.append(int(current))
            for center_pos, center in enumerate(walk):
                left = max(0, center_pos - window_size)
                right = min(len(walk), center_pos + window_size + 1)
                for context_pos in range(left, right):
                    if context_pos != center_pos:
                        cooc[center, walk[context_pos]] += 1.0

    total = cooc.sum()
    if total == 0:
        return np.zeros((n_nodes, dimensions), dtype=np.float32)
    row_sum = cooc.sum(axis=1, keepdims=True)
    col_sum = cooc.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    with np.errstate(divide="ignore", invalid="ignore"):
        ppmi = np.maximum(np.log((cooc * total) / np.maximum(expected, 1e-12)), 0.0)
    ppmi[~np.isfinite(ppmi)] = 0.0
    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
    dims = min(dimensions, u.shape[1])
    embeddings = u[:, :dims] * np.sqrt(s[:dims])
    if dims < dimensions:
        embeddings = np.pad(embeddings, ((0, 0), (0, dimensions - dims)))
    return embeddings.astype(np.float32)
