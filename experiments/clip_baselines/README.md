# CLIP Baselines

This folder contains quick experiments that run CLIP in zero-shot mode to score document vs non-document categories on the scanned-image dataset.

## Files

- `clip_zero_shot.py`: main entry point. Loads a CLIP zero-shot classifier, evaluates every image in the dataset, and writes a CSV with per-image scores.
- `prompts/doc_vs_non_doc.json`: prompt groups used for zero-shot classification. The top-level keys are logical groups (e.g., `document`, `non_document`) and each value is a list of textual prompts describing that group.

## Running

```bash
source .venv/bin/activate
pip install open_clip_torch  # first time only
PYTHONPATH=. python experiments/clip_baselines/clip_zero_shot.py \
  --dataset-root /Users/admin/Desktop/dataset \
  --output-csv experiments/clip_baselines/out/clip_doc_vs_non_doc.csv \
  --prompts experiments/clip_baselines/prompts/doc_vs_non_doc.json \
  --sample-per-class 50
```

This writes per-image scores (raw CLIP score, per-group max score, and `doc_minus_non_doc`) that you can feed into the same evaluation harness we used for `DocumentScorer`. By default it uses `open_clip`'s `ViT-L-14` pretrained on `openai`; override `--model`/`--pretrained` for other checkpoints. The script currently skips the `Docs` and `Handwritten` folders to keep the test set focused on mixed-layout documents and clearly non-document scans.
