"""Utility functions shared by RTMS analysis scripts."""

from __future__ import annotations

import os
import csv
from pathlib import Path
from typing import Iterable, List, Set
from ast import literal_eval
from collections import Counter

import networkx as nx
import numpy as np


def remove_prefix_suffix(name: str) -> str:
    """Return ``name`` without the ROICorrelation prefix and ``.txt`` suffix."""
    if name.startswith("ROICorrelation_FisherZ_"):
        name = name[len("ROICorrelation_FisherZ_") :]
    return name[:-4] if name.endswith(".txt") else name


def matrix_preprocess(matrix: np.ndarray) -> np.ndarray:
    """Return a copy of ``matrix`` with zeroed diagonal."""
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)
    return matrix


def matrix_abs(matrix: np.ndarray) -> np.ndarray:
    """Return the absolute value of ``matrix`` in a new array."""
    return np.abs(matrix)


def group_average(path: str) -> np.ndarray:
    """Return the average connectivity matrix computed from ``path``."""

    files = [Path(path) / f for f in os.listdir(path) if f.endswith(".txt")]
    if not files:
        raise ValueError(f"No matrix files found in {path}")

    total = None
    for fpath in files:
        data = np.loadtxt(fpath)
        np.fill_diagonal(data, 0)
        data = np.abs(data)
        if total is None:
            total = data
        else:
            total += data

    return total / len(files)


def generate_network_with_threshold(matrix: np.ndarray, threshold: float) -> tuple[nx.Graph, np.ndarray, float]:
    """Build a graph using edges with absolute weight above ``threshold``."""

    matrix = np.abs(matrix_preprocess(matrix))
    mask = matrix > threshold
    np.fill_diagonal(mask, False)

    rows, cols = np.where(np.triu(mask, 1))
    weights = matrix[rows, cols]

    graph = nx.Graph()
    graph.add_nodes_from(range(matrix.shape[0]))
    graph.add_weighted_edges_from(zip(rows, cols, weights))

    thresholded = np.zeros_like(matrix)
    thresholded[rows, cols] = weights
    thresholded[cols, rows] = weights

    return graph, thresholded, threshold


def greedy_minimum_dominating_set(graph: nx.Graph, times: int) -> List[Set[int]]:
    """Return candidate minimum dominating sets using a greedy approach."""

    neighbor_map = {n: set(graph.neighbors(n)) | {n} for n in graph}
    nodes = list(graph.nodes)
    best_sets: List[Set[int]] = []
    best_size = None

    for _ in range(times):
        remaining = set(nodes)
        dom_set: Set[int] = set()
        while remaining:
            node = remaining.pop()
            dom_set.add(node)
            remaining.difference_update(neighbor_map[node])

        if best_size is None or len(dom_set) < best_size:
            best_sets = [dom_set]
            best_size = len(dom_set)
        elif len(dom_set) == best_size and dom_set not in best_sets:
            best_sets.append(dom_set)

    return best_sets


def dominating_frequency(all_sets: Iterable[Iterable[int]], graph: nx.Graph) -> dict[int, float]:
    """Calculate node dominating frequency across ``all_sets``."""

    counter: Counter[int] = Counter()
    total = 0
    for ds in all_sets:
        counter.update(ds)
        total += 1

    return {n: counter[n] / total for n in graph.nodes}


def read_sets(path: str) -> List[Set[int]]:
    """Load dominating sets from ``path``."""
    sets: List[Set[int]] = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("Set "):
                sets.append(set(literal_eval(line.split(": ", 1)[1])))
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

