from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import open_clip

DOCUMENT_FOLDERS = {
    "docs",
    "email",
    "form",
    "letter",
    "memo",
    "news",
    "report",
    "resume",
    "scientific",
}

NON_DOCUMENT_FOLDERS = {
    "adve",
    "note",
    "handwritten",
}

EXCLUDED_FOLDERS = {
    "docs",
    "handwritten",
}
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_prompts(path: Path) -> Tuple[List[str], Dict[str, str]]:
    config = json.loads(path.read_text())
    labels: List[str] = []
    reverse: Dict[str, str] = {}
    for group, prompts in config.items():
        for prompt in prompts:
            labels.append(prompt)
            reverse[prompt] = group
    return labels, reverse


def infer_ground_truth(folder_name: str) -> int | None:
    name = folder_name.lower()
    if name in DOCUMENT_FOLDERS:
        return 1
    if name in NON_DOCUMENT_FOLDERS:
        return 0
    return None


def iter_images(dataset_root: Path, sample_per_class: int | None = None):
    for label_dir in sorted(dataset_root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if label.lower() in EXCLUDED_FOLDERS:
            continue
        files = [p for p in sorted(label_dir.iterdir()) if p.suffix.lower() in ALLOWED_SUFFIXES]
        if sample_per_class is not None:
            files = files[:sample_per_class]
        for path in files:
            yield label, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CLIP zero-shot doc/non-doc baseline using open_clip")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model", default="ViT-L-14", help="open_clip model name")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained identifier")
    parser.add_argument("--sample-per-class", type=int, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = open_clip.create_model_from_pretrained(args.model, args.pretrained)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(args.model)

    candidate_labels, reverse_map = load_prompts(args.prompts)
    with torch.no_grad():
        text_tokens = tokenizer(candidate_labels)
        text_emb = model.encode_text(text_tokens.to(device))
        text_emb = F.normalize(text_emb, dim=-1)

    rows = []
    for folder, image_path in iter_images(args.dataset_root, args.sample_per_class):
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
        row = {
            "path": str(image_path),
            "dataset_class": folder,
            "is_document": infer_ground_truth(folder),
            "clip_top_label": top_label,
            "clip_top_group": top_group,
            "clip_top_score": float(probs[top_idx]),
            "score_document": group_scores.get("document", 0.0),
            "score_non_document": group_scores.get("non_document", 0.0),
            "doc_minus_non_doc": group_scores.get("document", 0.0) - group_scores.get("non_document", 0.0),
        }
        rows.append(row)
        print(f"{image_path}: doc={row['score_document']:.3f} non-doc={row['score_non_document']:.3f}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "dataset_class",
                "is_document",
                "clip_top_label",
                "clip_top_group",
                "clip_top_score",
                "score_document",
                "score_non_document",
                "doc_minus_non_doc",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
