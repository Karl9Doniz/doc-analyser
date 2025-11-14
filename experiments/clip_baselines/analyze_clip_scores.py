import argparse
from pathlib import Path
import json
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CLIP zero-shot CSV results")
    parser.add_argument("input_csv", type=Path, help="CSV produced by clip_zero_shot.py")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save JSON summary")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df["score_document"] = df["score_document"].astype(float)
    df["score_non_document"] = df["score_non_document"].astype(float)
    df["doc_minus_non_doc"] = df["doc_minus_non_doc"].astype(float)

    doc_classes = {"docs", "email", "form", "letter", "memo", "news", "report", "resume", "scientific"}
    non_doc_classes = {"adve", "note", "handwritten"}

    df["is_document"] = df["dataset_class"].str.lower().map(lambda c: 1 if c in doc_classes else (0 if c in non_doc_classes else None))

    summary = {
        "samples": len(df),
        "doc_avg": df[df["dataset_class"].str.lower().isin(doc_classes)]["score_document"].mean(),
        "non_doc_avg": df[df["dataset_class"].str.lower().isin(non_doc_classes)]["score_document"].mean(),
        "doc_minus_doc_avg": df[df["dataset_class"].str.lower().isin(doc_classes)]["doc_minus_non_doc"].mean(),
        "doc_minus_non_doc_avg": df[df["dataset_class"].str.lower().isin(non_doc_classes)]["doc_minus_non_doc"].mean(),
    }

    per_class = df.groupby("dataset_class")["doc_minus_non_doc"].agg(["mean", "median", "count"]).reset_index()
    summary["per_class"] = per_class.to_dict(orient="records")

    print("Total samples:", summary["samples"])
    print("Avg CLIP document score for doc folders:", summary["doc_avg"])
    print("Avg CLIP document score for non-doc folders:", summary["non_doc_avg"])
    print("Mean doc_minus for doc folders:", summary["doc_minus_doc_avg"])
    print("Mean doc_minus for non-doc folders:", summary["doc_minus_non_doc_avg"])
    print("\nPer-class doc_minus stats:")
    print(per_class)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
