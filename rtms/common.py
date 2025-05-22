"""Utility functions shared by RTMS analysis scripts."""

from __future__ import annotations

import os
import csv
import random
from pathlib import Path
from typing import Iterable, List, Set

import networkx as nx
import numpy as np


def remove_prefix_suffix(name: str) -> str:
    """Return ``name`` without the ROICorrelation prefix and ``.txt`` suffix."""
    if name.startswith("ROICorrelation_FisherZ_"):
        name = name[len("ROICorrelation_FisherZ_") :]
    return name[:-4] if name.endswith(".txt") else name


def matrix_preprocess(matrix: np.ndarray) -> np.ndarray:
    """Zero the diagonal of ``matrix`` and return it."""
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)
    return matrix


def matrix_abs(matrix: np.ndarray) -> np.ndarray:
    """Return absolute values of ``matrix``."""
    return np.abs(matrix)


def group_average(path: str) -> np.ndarray:
    """Compute the group average matrix within ``path``.

    All matrices in ``path`` are loaded, preprocessed, summed, and then
    averaged. The returned matrix has its diagonal set to zero and contains
    only positive values.
    """
    matrices: List[np.ndarray] = []
    for fname in os.listdir(path):
        fpath = Path(path) / fname
        if fpath.suffix != ".txt":
            continue
        data = np.loadtxt(fpath)
        matrices.append(matrix_abs(matrix_preprocess(np.array(data))))

    if not matrices:
        raise ValueError(f"No matrix files found in {path}")

    avg = sum(matrices) / len(matrices)
    return avg


def generate_network_with_threshold(matrix: np.ndarray, threshold: float) -> tuple[nx.Graph, np.ndarray, float]:
    """Build a graph with edges above ``threshold`` using absolute weights."""
    matrix = matrix_abs(matrix_preprocess(matrix))
    n = matrix.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    thresholded = np.zeros_like(matrix)

    for i in range(n):
        for j in range(i + 1, n):
            weight = matrix[i, j]
            if weight > threshold:
                graph.add_edge(i, j, weight=weight)
                thresholded[i, j] = thresholded[j, i] = weight
    return graph, thresholded, threshold


def greedy_mds(graph: nx.Graph, times: int = 100, seed: int | None = None) -> Set[int]:
    """Return an approximate minimum dominating set using a randomized greedy search.

    Parameters
    ----------
    graph : nx.Graph
        The graph on which to compute the dominating set.
    times : int, optional
        How many greedy trials to perform. Defaults to ``100``.
    seed : int | None, optional
        Random seed for reproducibility. Defaults to ``None``.
    """

    rng = random.Random(seed)
    best_set: Set[int] | None = None

    for _ in range(times):
        undominated: set[int] = set(graph.nodes)
        dom_set: set[int] = set()

        # Greedy stage
        while undominated:
            gains = {
                v: len(set(graph.neighbors(v)).union({v}) & undominated)
                for v in graph.nodes
                if v not in dom_set
            }
            max_gain = max(gains.values())
            candidates = [v for v, g in gains.items() if g == max_gain]
            v_star = rng.choice(candidates)

            dom_set.add(v_star)
            undominated -= set(graph.neighbors(v_star)).union({v_star})

        # Backward pruning to ensure minimality
        for v in list(dom_set):
            if all(
                any(u in dom_set and u != v for u in graph.neighbors(w))
                or (w in dom_set and w != v)
                for w in set(graph.neighbors(v)).union({v})
            ):
                dom_set.remove(v)

        if best_set is None or len(dom_set) < len(best_set):
            best_set = dom_set

    return best_set if best_set is not None else set()


def dominating_frequency(all_sets: Iterable[Iterable[int]], graph: nx.Graph) -> dict[int, float]:
    """Calculate node dominating frequency across ``all_sets``."""
    count = {n: 0 for n in graph.nodes}
    all_sets = list(all_sets)
    for ds in all_sets:
        for node in ds:
            count[node] += 1
    total = len(all_sets)
    return {n: c / total for n, c in count.items()}


def read_sets(path: str) -> List[Set[int]]:
    """Load dominating sets from ``path``."""
    sets: List[Set[int]] = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("Set "):
                sets.append(set(eval(line.split(": ")[1])))
    return sets


def write_sets(sets: Iterable[Iterable[int]], path: str) -> None:
    """Write dominating ``sets`` to ``path``."""
    with open(path, "w") as f:
        for idx, ds in enumerate(sets, start=1):
            f.write(f"Set {idx}: {set(ds)}\n")


def save_frequency(freq: dict[int, float], path: str) -> None:
    """Save frequency dictionary to ``path`` as CSV."""
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Node", "Frequency"])
        for node, val in freq.items():
            writer.writerow([node, val])

