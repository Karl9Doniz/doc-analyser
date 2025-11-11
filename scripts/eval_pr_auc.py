#!/usr/bin/env python
"""Compute Precision-Recall curve and PR-AUC from CSV results."""

import argparse
from pathlib import Path

import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PR curve / PR-AUC from classifier CSV output")
    parser.add_argument("input_csv", type=Path, help="CSV with columns path,label,is_document,score")
    parser.add_argument("--plot", type=Path, default=Path("out/evals/pr_curve.png"), help="Where to save the PR curve plot")
    parser.add_argument("--summary", type=Path, default=Path("out/evals/pr_summary.json"), help="Where to save summary JSON")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    y_true = df["is_document"].astype(int)
    y_score = df["score"].astype(float)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    # plot
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.step(recall, precision, where="post", label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.05])
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.savefig(args.plot)
    plt.close()

    summary = {
        "samples": len(df),
        "positive_count": int(y_true.sum()),
        "negative_count": int((len(df) - y_true.sum())),
        "pr_auc": pr_auc,
        "input_csv": str(args.input_csv),
        "plot_path": str(args.plot),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2))

    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Plot saved to {args.plot}")
    print(f"Summary saved to {args.summary}")


if __name__ == "__main__":
    main()
