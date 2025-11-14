import argparse
import csv
from pathlib import Path
from typing import Dict, List, Any

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.tools import ImagePreprocessor, OCRExtractor, DocumentScorer


def is_document_label(label: str) -> int:
    doc_like = {
        "report",
        "news",
        "letter",
        "memo",
        "form",
        "resume",
        "scientific",
        "email",
    }
    non_doc = {"adve", "note"}
    if label.lower() in non_doc:
        return 0
    return int(label.lower() in doc_like)


def build_manifest(dataset_root: Path, sample_per_class: int | None = None) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for label_dir in sorted(dataset_root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        files = [p for p in sorted(label_dir.iterdir()) if p.is_file()]
        if sample_per_class is not None:
            files = files[:sample_per_class]
        for file_path in files:
            entries.append({
                "path": str(file_path),
                "label": label,
                "is_document": is_document_label(label),
            })
    return entries


def run_pipeline(image_path: str) -> float:
    preprocess_result = ImagePreprocessor.execute(image_path, method="original")
    image_to_use = preprocess_result.data.get("input_path", image_path)

    ocr_result = OCRExtractor.execute(image_to_use)
    score_result = DocumentScorer.execute(
        image_path=image_to_use,
        text_data=ocr_result.data if ocr_result.success else {},
    )
    if score_result.success:
        return float(score_result.data.get("final_score", 0.0))
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DocumentScorer confidence on a dataset")
    parser.add_argument("dataset_root", type=Path, help="Root directory with category subfolders")
    parser.add_argument("output_csv", type=Path, help="CSV file to write per-sample predictions")
    parser.add_argument("--sample-per-class", type=int, default=None, help="Optional maximum samples per class")
    args = parser.parse_args()

    manifest = build_manifest(args.dataset_root, args.sample_per_class)
    if not manifest:
        raise SystemExit("No samples found under the dataset root")

    rows = []
    for entry in manifest:
        score = run_pipeline(entry["path"])
        rows.append({**entry, "score": score})
        print(f"{entry['path']}: score={score:.3f}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "is_document", "score"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
