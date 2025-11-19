from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# Make the src/ directory importable so tools.py can locate chart_extractor, etc.
ROOT_DIR = Path(__file__).parent.resolve()
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from tools import DocumentScorer, ImageLoader, ImagePreprocessor, OCRExtractor  # noqa: E402

from experiments.clip_baselines.clip_zero_shot import (
    ALLOWED_SUFFIXES,
    EXCLUDED_FOLDERS,
    load_prompts,
)
from experiments.clip_baselines.clip_zero_shot import iter_images as clip_iter_images  # noqa: E402
from experiments.clip_baselines.clip_zero_shot import infer_ground_truth  # noqa: E402
import open_clip  # noqa: E402


def gather_samples(dataset_root: Path, sample_per_class: int | None) -> List[Tuple[str, Path]]:
    """Collect (folder, path) pairs matching the CLIP experiment rules."""
    dataset_root = dataset_root.expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    samples = list(clip_iter_images(dataset_root, sample_per_class))
    if not samples:
        raise RuntimeError(f"No images found under {dataset_root} with suffixes {sorted(ALLOWED_SUFFIXES)}")
    return samples


def run_clip_scoring(
    samples: Sequence[Tuple[str, Path]],
    prompts_path: Path,
    model_name: str,
    pretrained: str,
) -> pd.DataFrame:
    """Replicate clip_zero_shot scoring for the provided sample set."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = open_clip.create_model_from_pretrained(model_name, pretrained)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(model_name)

    candidate_labels, reverse_map = load_prompts(prompts_path)
    with torch.no_grad():
        text_tokens = tokenizer(candidate_labels)
        text_emb = model.encode_text(text_tokens.to(device))
        text_emb = F.normalize(text_emb, dim=-1)

    rows: List[Dict[str, object]] = []
    start = time.time()
    for folder, image_path in samples:
        image = preprocess(Image.open(image_path).convert("RGB"))
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            image_emb = model.encode_image(image)
            image_emb = F.normalize(image_emb, dim=-1)
            logits = (100.0 * image_emb @ text_emb.T).squeeze(0)
            probs = logits.softmax(dim=-1).cpu()

        prompt_scores = {label: float(probs[idx]) for idx, label in enumerate(candidate_labels)}
        group_scores: Dict[str, float] = {}
        for prompt, score in prompt_scores.items():
            group = reverse_map[prompt]
            group_scores[group] = max(group_scores.get(group, 0.0), score)

        top_idx = int(probs.argmax())
        top_label = candidate_labels[top_idx]
        top_group = reverse_map[top_label]
        rows.append(
            {
                "path": str(image_path),
                "dataset_class": folder,
                "is_document": infer_ground_truth(folder),
                "clip_top_label": top_label,
                "clip_top_group": top_group,
                "clip_top_score": float(probs[top_idx]),
                "clip_score_document": group_scores.get("document", 0.0),
                "clip_score_non_document": group_scores.get("non_document", 0.0),
                "clip_doc_minus_non_doc": group_scores.get("document", 0.0)
                - group_scores.get("non_document", 0.0),
            }
        )
    elapsed = time.time() - start
    print(f"[CLIP] Processed {len(rows)} samples in {elapsed:.1f}s using {model_name}/{pretrained} on {device}")
    return pd.DataFrame(rows)


def run_heuristic_scoring(samples: Sequence[Tuple[str, Path]]) -> pd.DataFrame:
    """Run ImageLoader -> Preprocessor -> OCR -> DocumentScorer for each sample."""
    rows: List[Dict[str, object]] = []
    start = time.time()
    for folder, image_path in samples:
        image_str = str(image_path)

        load_result = ImageLoader.execute(image_str)
        if not load_result.success:
            print(f"[Heuristic] Skipping {image_str}: {load_result.error}")
            continue

        preprocess_result = ImagePreprocessor.execute(image_str, method="adaptive")
        ocr_input = image_str
        if preprocess_result.success:
            ocr_input = preprocess_result.data.get("output_path", image_str)

        ocr_result = OCRExtractor.execute(ocr_input)
        if not ocr_result.success:
            text_data = {"char_count": 0, "word_count": 0, "has_meaningful_content": False}
        else:
            text_data = ocr_result.data

        score_result = DocumentScorer.execute(image_str, text_data)
        if not score_result.success:
            print(f"[Heuristic] Scoring failed for {image_str}: {score_result.error}")
            continue

        rows.append(
            {
                "path": image_str,
                "dataset_class": folder,
                "is_document": infer_ground_truth(folder),
                "heuristic_char_density": score_result.data.get("char_density"),
                "heuristic_content_score": score_result.data.get("content_score"),
                "heuristic_aspect_score": score_result.data.get("aspect_score"),
                "heuristic_final_score": score_result.data.get("final_score"),
            }
        )
    elapsed = time.time() - start
    print(f"[Heuristic] Processed {len(rows)} samples in {elapsed:.1f}s")
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame, score_col: str, threshold: float) -> Dict[str, object]:
    """Return confusion counts, precision/recall, and per-class stats."""
    df = df.dropna(subset=["is_document"]).copy()
    if df.empty:
        raise RuntimeError("No labeled samples available for metric computation.")

    preds = (df[score_col] >= threshold).astype(int)
    actuals = df["is_document"].astype(int)

    tp = int(((preds == 1) & (actuals == 1)).sum())
    fp = int(((preds == 1) & (actuals == 0)).sum())
    tn = int(((preds == 0) & (actuals == 0)).sum())
    fn = int(((preds == 0) & (actuals == 1)).sum())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    accuracy = (tp + tn) / max(1, len(df))

    per_class_accuracy: Dict[str, float] = {}
    per_class_score: Dict[str, float] = {}
    for cls, group in df.groupby("dataset_class"):
        class_preds = (group[score_col] >= threshold).astype(int)
        class_actuals = group["is_document"].astype(int)
        per_class_accuracy[cls] = float((class_preds == class_actuals).mean())
        per_class_score[cls] = float(group[score_col].mean())

    return {
        "samples": int(len(df)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "per_class_accuracy": per_class_accuracy,
        "per_class_score": per_class_score,
    }


def plot_per_class_accuracy(
    heuristic_acc: Dict[str, float],
    clip_acc: Dict[str, float],
    output_path: Path,
) -> None:
    """Save a bar chart comparing per-class accuracy for both methods."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib is not installed; skipping per-class accuracy plot.")
        return

    classes = sorted(set(heuristic_acc) | set(clip_acc))
    if not classes:
        print("No per-class accuracy data available; skipping plot.")
        return

    x = range(len(classes))
    heuristic_vals = [heuristic_acc.get(cls, 0.0) for cls in classes]
    clip_vals = [clip_acc.get(cls, 0.0) for cls in classes]

    width = 0.4
    plt.figure(figsize=(max(8, len(classes) * 0.7), 5))
    plt.bar([i - width / 2 for i in x], heuristic_vals, width=width, label="Heuristic", color="#4C72B0")
    plt.bar([i + width / 2 for i in x], clip_vals, width=width, label="CLIP", color="#DD8452")
    plt.xticks(list(x), classes, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Per-class document detection accuracy")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved comparison plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CLIP doc scoring vs heuristic DocumentScorer.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root folder of labeled dataset.")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("experiments/clip_baselines/prompts/doc_vs_non_doc.json"),
        help="Prompt file to use for CLIP zero-shot evaluation.",
    )
    parser.add_argument("--model", default="ViT-L-14", help="open_clip model name.")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained identifier.")
    parser.add_argument("--sample-per-class", type=int, default=None, help="Optional cap per dataset folder.")
    parser.add_argument("--clip-threshold", type=float, default=0.4, help="Threshold on doc_minus score for CLIP.")
    parser.add_argument(
        "--heuristic-threshold",
        type=float,
        default=0.5,
        help="Threshold on DocumentScorer final_score.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/clip_baselines/out/competition"),
        help="Directory to store CSVs, metrics, and plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = gather_samples(args.dataset_root, args.sample_per_class)
    print(f"Collected {len(samples)} images across {len(set(folder for folder, _ in samples))} folders.")

    clip_rows_start = time.time()
    clip_df = run_clip_scoring(samples, args.prompts, args.model, args.pretrained)
    clip_elapsed = time.time() - clip_rows_start

    heuristic_rows_start = time.time()
    heuristic_df = run_heuristic_scoring(samples)
    heuristic_elapsed = time.time() - heuristic_rows_start

    combined = clip_df.merge(
        heuristic_df,
        on=["path", "dataset_class", "is_document"],
        how="inner",
        validate="one_to_one",
    )
    if combined.empty:
        raise RuntimeError("No overlapping samples between CLIP and heuristic runs.")

    clip_metrics = compute_metrics(combined, "clip_doc_minus_non_doc", args.clip_threshold)
    heuristic_metrics = compute_metrics(combined, "heuristic_final_score", args.heuristic_threshold)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_csv = output_dir / "clip_scores.csv"
    heuristic_csv = output_dir / "heuristic_scores.csv"
    combined_csv = output_dir / "combined_scores.csv"
    clip_df.to_csv(clip_csv, index=False)
    heuristic_df.to_csv(heuristic_csv, index=False)
    combined.to_csv(combined_csv, index=False)

    summary = {
        "clip": {**clip_metrics, "runtime_seconds": clip_elapsed},
        "heuristic": {**heuristic_metrics, "runtime_seconds": heuristic_elapsed},
        "settings": {
            "dataset_root": str(args.dataset_root),
            "prompts": str(args.prompts),
            "model": args.model,
            "pretrained": args.pretrained,
            "clip_threshold": args.clip_threshold,
            "heuristic_threshold": args.heuristic_threshold,
            "sample_per_class": args.sample_per_class,
        },
        "artifacts": {
            "clip_csv": str(clip_csv),
            "heuristic_csv": str(heuristic_csv),
            "combined_csv": str(combined_csv),
        },
    }
    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary metrics to {summary_path}")

    plot_path = output_dir / "per_class_accuracy.png"
    plot_per_class_accuracy(
        heuristic_metrics["per_class_accuracy"],
        clip_metrics["per_class_accuracy"],
        plot_path,
    )

    print("\n--- Results ---")
    for label, metrics in (("CLIP", clip_metrics), ("Heuristic", heuristic_metrics)):
        print(
            f"{label}: accuracy={metrics['accuracy']:.3f} precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} (tp={metrics['tp']} fp={metrics['fp']} tn={metrics['tn']} fn={metrics['fn']})"
        )


if __name__ == "__main__":
    main()
