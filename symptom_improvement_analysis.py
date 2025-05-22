"""Rewritten symptom improvement analysis.
This script consolidates the logic previously in `症状改善相关.ipynb`.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from power264_brain_area import (
    brain_area_264,
    brain_area_subgraph_mapping,
    subgraph_ids,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def build_brain_area_to_subgraph() -> Dict[int, int]:
    """Return mapping from brain area index to subgraph index."""
    mapping = {}
    for idx, name in brain_area_264.items():
        mapping[idx] = brain_area_subgraph_mapping.get(name, 13)
    return mapping


def merge_freq_ac(
    freq_path: Path, ac_path: Path, output_path: Path
) -> pd.DataFrame:
    """Merge frequency difference and controllability difference files."""
    freq_df = pd.read_csv(freq_path)
    ac_df = pd.read_csv(ac_path)

    freq_df = freq_df.rename(columns={c: f"Freq_{c}" for c in freq_df.columns if c != "id"})
    ac_df = ac_df.rename(columns={c: f"AC_{c}" for c in ac_df.columns if c != "id"})

    merged = pd.merge(freq_df, ac_df, on="id", how="inner")
    merged.to_csv(output_path, index=False)
    return merged


def _reshape_by_subgraph(
    df: pd.DataFrame, mapping: Dict[int, int], prefix: str
) -> pd.DataFrame:
    """Reshape node level data to subgraph level."""
    long_df = df.melt(id_vars="id", var_name="brain", value_name=prefix)
    long_df["brain"] = long_df["brain"].astype(int)
    long_df["subgraph"] = long_df["brain"].map(mapping)

    sub = long_df.groupby(["id", "subgraph"])[prefix].mean().unstack("subgraph")
    sub.columns = [f"{prefix}_sg{c}" for c in sub.columns]
    return sub.reset_index()


def merge_node_subgraph(
    node_path: Path, subgraph_path: Path, output_path: Path
) -> pd.DataFrame:
    """Merge node-level and subgraph-level metrics."""
    node_df = pd.read_csv(node_path)
    sub_df = pd.read_csv(subgraph_path)
    sub_df = sub_df[sub_df["id"].isin(node_df["id"])]
    merged = pd.merge(node_df, sub_df, on="id")
    merged.to_csv(output_path, index=False)
    return merged


def compute_score_diff(score_path: Path, out_path: Path) -> pd.DataFrame:
    """Compute score differences between pre and post treatment."""
    df = pd.read_csv(score_path)
    score_cols = [
        "HAMA_total_score",
        "HAMD17_total_score",
        "BPRS_total_score",
    ]
    pre = df[df["scan"] == 1].set_index("id")[score_cols]
    post = df[df["scan"] == 3].set_index("id")[score_cols]
    delta = post - pre
    delta.columns = [f"{c}_diff" for c in delta.columns]
    delta.reset_index().to_csv(out_path, index=False)
    return delta.reset_index()


def merge_features_scores(
    feature_path: Path, score_path: Path, out_path: Path
) -> pd.DataFrame:
    """Merge features with score differences."""
    feat_df = pd.read_csv(feature_path)
    score_df = pd.read_csv(score_path)
    score_df = score_df[score_df["id"].isin(feat_df["id"])]
    merged = pd.merge(feat_df, score_df, on="id")
    merged.to_csv(out_path, index=False)
    return merged


def correlation_analysis(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    score_cols: Iterable[str],
    corr_path: Path,
    spearman_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Pearson and Spearman correlations."""
    score_cols = list(score_cols)
    features = df[feature_cols]
    targets = df[score_cols]

    corr = features.corrwith(targets[score_cols[0]]).to_frame("HAMA_corr")
    corr["HAMD17_corr"] = features.corrwith(targets[score_cols[1]])
    corr["BPRS_corr"] = features.corrwith(targets[score_cols[2]])
    corr.to_csv(corr_path)

    results: List[List[float]] = []
    for col in feature_cols:
        r1, p1 = spearmanr(features[col], targets[score_cols[0]])
        r2, p2 = spearmanr(features[col], targets[score_cols[1]])
        r3, p3 = spearmanr(features[col], targets[score_cols[2]])
        results.append([col, r1, p1, r2, p2, r3, p3])

    spearman_df = pd.DataFrame(
        results,
        columns=[
            "feature",
            "HAMA_spearman_corr",
            "HAMA_p",
            "HAMD17_spearman_corr",
            "HAMD17_p",
            "BPRS_spearman_corr",
            "BPRS_p",
        ],
    )
    spearman_df.to_csv(spearman_path, index=False)
    return corr, spearman_df


# ---------------------------------------------------------------------------
# Example pipeline (paths should be adapted to actual data locations)
# ---------------------------------------------------------------------------

def main() -> None:
    mapping = build_brain_area_to_subgraph()

    freq_diff = Path("frequency_diff.csv")
    ac_diff = Path("Average_Controllability_diff.csv")
    merged_node = Path("merged_diff.csv")

    merge_freq_ac(freq_diff, ac_diff, merged_node)

    node_df = pd.read_csv(merged_node)
    freq_sub = _reshape_by_subgraph(node_df[["id"] + [c for c in node_df.columns if c.startswith("Freq_")]], mapping, "Freq")
    ac_sub = _reshape_by_subgraph(node_df[["id"] + [c for c in node_df.columns if c.startswith("AC_")]], mapping, "AC")

    subgraph_df = pd.merge(freq_sub, ac_sub, on="id")
    subgraph_df.to_csv("subgraph_diff.csv", index=False)

    merge_node_subgraph("node_diff.csv", "subgraph_diff.csv", "freq_ac(node+subgraph).csv")

    compute_score_diff("rtms_scores.csv", "score_diff.csv")

    merge_features_scores(
        "freq_ac(node+subgraph).csv",
        "score_diff.csv",
        "freq_ac_score_diff.csv",
    )

    df = pd.read_csv("freq_ac_score_diff.csv")
    feature_cols = [c for c in df.columns if c not in ["id", "HAMA_total_score_diff", "HAMD17_total_score_diff", "BPRS_total_score_diff"]]
    score_cols = [
        "HAMA_total_score_diff",
        "HAMD17_total_score_diff",
        "BPRS_total_score_diff",
    ]

    correlation_analysis(
        df,
        feature_cols,
        score_cols,
        Path("score_correlation.csv"),
        Path("score_spearman.csv"),
    )


if __name__ == "__main__":
    main()
