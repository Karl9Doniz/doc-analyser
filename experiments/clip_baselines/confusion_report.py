#!/usr/bin/env python
"""Compute confusion matrices for CLIP doc/non-doc scores at multiple thresholds."""

import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

DOC_CLASSES = {"docs", "email", "form", "letter", "memo", "news", "report", "resume", "scientific"}
NON_DOC_CLASSES = {"adve", "note", "handwritten"}


def compute_metrics(df: pd.DataFrame, threshold: float) -> dict:
    preds = (df["doc_minus_non_doc"] >= threshold).astype(int)
    actuals = df["is_document"].astype(int)

    tp = int(((preds == 1) & (actuals == 1)).sum())
    fp = int(((preds == 1) & (actuals == 0)).sum())
    tn = int(((preds == 0) & (actuals == 0)).sum())
    fn = int(((preds == 0) & (actuals == 1)).sum())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
    }


def plot_distribution(df: pd.DataFrame, threshold: float, output_path: Path) -> None:
    plt.figure()
    plt.hist(
        [df[df["is_document"] == 1]["doc_minus_non_doc"], df[df["is_document"] == 0]["doc_minus_non_doc"]],
        bins=20,
        label=["Document", "Non-document"],
        alpha=0.7,
    )
    plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold {threshold}")
    plt.title(f"doc_minus distribution @ threshold {threshold}")
    plt.xlabel("doc_minus_non_doc")
    plt.ylabel("Count")
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_confusion_heatmap(df: pd.DataFrame, threshold: float, output_path: Path) -> None:
    preds = (df["doc_minus_non_doc"] >= threshold).astype(int)
    actuals = df["is_document"].astype(int)
    cm = confusion_matrix(actuals, preds, labels=[1, 0])
    plt.figure()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Doc", "Pred Non-Doc"],
        yticklabels=["Actual Doc", "Actual Non-Doc"],
    )
    plt.title(f"Confusion Matrix @ threshold {threshold}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Confusion analysis for CLIP doc/non-doc scores")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.4, 0.5, 0.6])
    parser.add_argument("--summary-json", type=Path, default=Path("experiments/clip_baselines/out/clip_confusions.json"))
    parser.add_argument("--plot-dir", type=Path, default=Path("experiments/clip_baselines/out/plots"))
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df["doc_minus_non_doc"] = df["doc_minus_non_doc"].astype(float)
    df["is_document"] = df["dataset_class"].str.lower().map(lambda c: 1 if c in DOC_CLASSES else 0)

    results = []
    for thr in args.thresholds:
        metrics = compute_metrics(df, thr)
        results.append(metrics)
        hist_path = args.plot_dir / f"doc_minus_distribution_thr_{thr:.1f}.png"
        plot_distribution(df, thr, hist_path)
        heat_path = args.plot_dir / f"confusion_matrix_thr_{thr:.1f}.png"
        plot_confusion_heatmap(df, thr, heat_path)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(results, indent=2))

    print("Confusion metrics saved to", args.summary_json)
    for entry in results:
        print(entry)


if __name__ == "__main__":
    main()
