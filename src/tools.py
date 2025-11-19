import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont
import pytesseract
import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    import open_clip
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    F = None  # type: ignore
    open_clip = None  # type: ignore

try:
    from surya.settings import settings as surya_settings  # type: ignore
    from surya.foundation import FoundationPredictor  # type: ignore
    from surya.layout import LayoutPredictor  # type: ignore
    from surya.recognition import RecognitionPredictor  # type: ignore
    from surya.common.surya.schema import TaskNames as SuryaTaskNames  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    surya_settings = None  # type: ignore
    FoundationPredictor = None  # type: ignore
    LayoutPredictor = None  # type: ignore
    RecognitionPredictor = None  # type: ignore
    SuryaTaskNames = None  # type: ignore

logger = logging.getLogger(__name__)

try:
    from chart_extractor import MiniCPMVEngine, run_minicpm_vqa
except ImportError:  # pragma: no cover - optional dependency
    MiniCPMVEngine = None  # type: ignore
    run_minicpm_vqa = None  # type: ignore

try:  # Optional: only needed when using LLM fallback for chart JSON repair
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore


class SuryaRuntime:
    """Lazy initializer for Surya models (layout + formula recognition)."""

    _lock = threading.Lock()
    _foundation = None
    _layout = None
    _recognition = None
    _init_error: Optional[str] = None

    @classmethod
    def available(cls) -> bool:
        return all(
            dependency is not None
            for dependency in (
                surya_settings,
                FoundationPredictor,
                LayoutPredictor,
                RecognitionPredictor,
                SuryaTaskNames,
            )
        )

    @classmethod
    def _ensure_models(cls):
        if not cls.available():
            raise RuntimeError(
                "Surya OCR models are unavailable. Install surya-ocr to enable formula detection."
            )

        with cls._lock:
            if cls._init_error:
                raise RuntimeError(cls._init_error)

            if cls._foundation is None:
                try:
                    # Force CPU by default to avoid MPS/GPU quirks on local machines.
                    desired_device = "cpu"
                    current_pref = getattr(surya_settings, "TORCH_DEVICE", None)
                    if str(current_pref).lower() != desired_device:
                        surya_settings.TORCH_DEVICE = desired_device  # type: ignore[attr-defined]
                        try:  # Ensure cached computed field lines up with override
                            surya_settings.__dict__["TORCH_DEVICE_MODEL"] = desired_device  # type: ignore[assignment]
                        except Exception:
                            pass
                        os.environ.setdefault("SURYA__TORCH_DEVICE", desired_device)
                    surya_settings.DISABLE_TQDM = True  # type: ignore[attr-defined]

                    cls._patch_pad_sequence()

                    cls._foundation = FoundationPredictor(device=desired_device)
                    cls._layout = LayoutPredictor(cls._foundation)
                    cls._recognition = RecognitionPredictor(cls._foundation)
                except Exception as exc:  # pragma: no cover - defensive guard
                    cls._init_error = str(exc)
                    raise

        return cls._foundation, cls._layout, cls._recognition

    @classmethod
    def get_layout(cls):
        _, layout, _ = cls._ensure_models()
        return layout

    @classmethod
    def get_recognition(cls):
        _, _, recognition = cls._ensure_models()
        return recognition

    @staticmethod
    def _patch_pad_sequence() -> None:
        """Provide padding_side support for torch versions that lack it."""
        try:
            from surya.common.surya import processor as processor_module  # type: ignore
        except Exception:
            return

        if getattr(processor_module, "_codex_pad_side_patched", False):
            return

        try:
            from torch.nn.utils.rnn import pad_sequence as torch_pad_sequence  # type: ignore
        except Exception:
            return

        def _pad_sequence_with_side(
            sequences,
            batch_first: bool = True,
            padding_side: Optional[str] = "left",
            padding_value: float = 0.0,
        ):
            if padding_side in (None, "right"):
                return torch_pad_sequence(
                    sequences, batch_first=batch_first, padding_value=padding_value
                )
            if padding_side != "left":
                raise ValueError(f"Unsupported padding_side '{padding_side}'")

            if not sequences:
                raise ValueError("pad_sequence_with_side received empty sequences.")

            max_len = max(int(seq.shape[0]) for seq in sequences)
            trailing_shape = sequences[0].shape[1:]
            if batch_first:
                out_shape = (len(sequences), max_len) + trailing_shape
            else:
                out_shape = (max_len, len(sequences)) + trailing_shape

            output = sequences[0].new_full(out_shape, padding_value)

            for idx, seq in enumerate(sequences):
                length = int(seq.shape[0])
                if batch_first:
                    output[idx, max_len - length : max_len] = seq
                else:
                    output[max_len - length : max_len, idx] = seq
            return output

        def _patched_pad_sequence(
            sequences,
            batch_first: bool = True,
            padding_side: Optional[str] = "left",
            padding_value: float = 0.0,
        ):
            if padding_side in (None, "right"):
                return torch_pad_sequence(
                    sequences, batch_first=batch_first, padding_value=padding_value
                )
            return _pad_sequence_with_side(
                sequences,
                batch_first=batch_first,
                padding_side=padding_side,
                padding_value=padding_value,
            )

        processor_module.pad_sequence = _patched_pad_sequence  # type: ignore[attr-defined]
        processor_module._codex_pad_side_patched = True  # type: ignore[attr-defined]


class ToolResult:
    """Standard tool result format"""
    def __init__(self, success: bool, data: Dict[str, Any], error: str = None):
        self.success = success
        self.data = data
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error
        }


class ImageLoader:
    """Tool: Load and validate image files"""

    @staticmethod
    def execute(image_path: str) -> ToolResult:
        try:
            if not os.path.exists(image_path):
                return ToolResult(False, {}, f"Image file not found: {image_path}")

            image = Image.open(image_path).convert("RGB")
            w, h = image.size

            return ToolResult(True, {
                "path": image_path,
                "width": w,
                "height": h,
                "aspect_ratio": h / w,
                "megapixels": (w * h) / 1_000_000
            })
        except Exception as e:
            return ToolResult(False, {}, f"Failed to load image: {str(e)}")


class ImagePreprocessor:
    """Tool: Apply various preprocessing techniques"""

    @staticmethod
    def execute(image_path: str, method: str = "adaptive") -> ToolResult:
        try:
            image = Image.open(image_path).convert("RGB")

            if method == "original":
                processed = image
            elif method == "grayscale":
                processed = ImageOps.grayscale(image).convert("RGB")
            elif method == "contrast":
                processed = ImageEnhance.Contrast(image).enhance(1.6)
            elif method == "adaptive":
                gray = np.array(ImageOps.grayscale(image), dtype=np.uint8)
                thresh = cv2.adaptiveThreshold(
                    gray, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31, 10
                )
                processed = Image.fromarray(thresh).convert("RGB")
            else:
                return ToolResult(False, {}, f"Unknown preprocessing method: {method}")

            # Save preprocessed image temporarily
            temp_path = f"/tmp/preprocessed_{method}_{os.path.basename(image_path)}"
            processed.save(temp_path)

            return ToolResult(True, {
                "method": method,
                "input_path": image_path,
                "output_path": temp_path,
                "size": processed.size
            })
        except Exception as e:
            return ToolResult(False, {}, f"Preprocessing failed: {str(e)}")


class OCRExtractor:
    """Tool: Extract text using OCR"""

    @staticmethod
    def execute(image_path: str, config: str = "--psm 6") -> ToolResult:
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, config=config)

            # Calculate basic text metrics
            clean_text = text.strip()
            words = clean_text.split()
            lines = [line.strip() for line in clean_text.split('\n') if line.strip()]

            return ToolResult(True, {
                "text": clean_text,
                "char_count": len(clean_text.replace(" ", "").replace("\n", "")),
                "word_count": len(words),
                "line_count": len(lines),
                "avg_word_length": sum(len(w) for w in words) / max(1, len(words)),
                "has_meaningful_content": len(words) > 5
            })
        except Exception as e:
            return ToolResult(False, {}, f"OCR extraction failed: {str(e)}")


class DocumentScorer:
    """Tool: Calculate document likelihood scores using CLIP (with heuristic fallback)."""

    _clip_threshold = float(os.getenv("CLIP_DOC_THRESHOLD", "0.4"))

    @staticmethod
    def execute(image_path: str, text_data: Dict[str, Any]) -> ToolResult:
        start = time.time()
        try:
            clip_data = CLIPDocumentRuntime.score(image_path)
            clip_data.update(
                {
                    "backend": "clip",
                    "runtime_ms": (time.time() - start) * 1000.0,
                    "threshold": DocumentScorer._clip_threshold,
                    "is_document": clip_data["score_document"] >= DocumentScorer._clip_threshold,
                    "final_score": clip_data["score_document"],
                    "confidence": clip_data["score_document"],
                }
            )
            return ToolResult(True, clip_data)
        except Exception as exc:
            logger.warning("CLIP scoring unavailable, falling back to heuristic scorer: %s", exc)
            return DocumentScorer._heuristic_score(image_path, text_data)

    @staticmethod
    def _heuristic_score(image_path: str, text_data: Dict[str, Any]) -> ToolResult:
        """Legacy heuristic scorer, used only as a fallback."""
        try:
            image = Image.open(image_path)
            w, h = image.size

            char_count = text_data.get("char_count", 0)
            pixel_area = w * h
            char_density = min(1.0, (char_count / (pixel_area / 1000)))

            word_count = text_data.get("word_count", 0)
            has_content = text_data.get("has_meaningful_content", False)
            content_score = 1.0 if has_content else 0.0

            aspect_ratio = h / w
            aspect_score = 1.0 if 0.5 <= aspect_ratio <= 2.0 else 0.5

            final_score = min(1.0, (char_density * 0.5 + content_score * 0.3 + aspect_score * 0.2))

            return ToolResult(
                True,
                {
                    "backend": "heuristic",
                    "char_density": char_density,
                    "content_score": content_score,
                    "aspect_score": aspect_score,
                    "final_score": final_score,
                    "confidence": final_score,
                    "is_document": final_score >= DocumentScorer._clip_threshold,
                    "threshold": DocumentScorer._clip_threshold,
                    "metrics": {
                        "chars_per_1k_pixels": (char_count / (pixel_area / 1000)),
                        "aspect_ratio": aspect_ratio,
                        "word_count": word_count,
                    },
                },
            )
        except Exception as exc:
            return ToolResult(False, {}, f"Scoring failed: {exc}")


class CLIPDocumentRuntime:
    """Lazy loader for CLIP zero-shot document scoring."""

    _lock = threading.Lock()
    _model = None
    _preprocess = None
    _text_emb = None
    _labels: List[str] = []
    _reverse: Dict[str, str] = {}
    _device = "cpu"

    @staticmethod
    def _default_prompts_path() -> Path:
        env_override = os.getenv("CLIP_PROMPTS_PATH")
        if env_override:
            return Path(env_override)
        root = Path(__file__).resolve().parents[1]
        return root / "experiments" / "clip_baselines" / "prompts" / "doc_vs_non_doc.json"

    @staticmethod
    def _model_name() -> str:
        return os.getenv("CLIP_MODEL_NAME", "ViT-L-14")

    @staticmethod
    def _pretrained_name() -> str:
        return os.getenv("CLIP_PRETRAINED", "openai")

    @classmethod
    def available(cls) -> bool:
        return open_clip is not None and torch is not None

    @classmethod
    def _load_prompts(cls, path: Path) -> Tuple[List[str], Dict[str, str]]:
        if not path.exists():
            raise FileNotFoundError(f"CLIP prompt file not found: {path}")
        config = json.loads(path.read_text())
        labels: List[str] = []
        reverse: Dict[str, str] = {}
        for group, prompts in config.items():
            for prompt in prompts:
                labels.append(prompt)
                reverse[prompt] = group
        return labels, reverse

    @classmethod
    def _ensure_runtime(cls):
        if not cls.available():
            raise RuntimeError(
                "open_clip_torch/torch not installed. Install them to enable CLIP-based scoring."
            )
        with cls._lock:
            if cls._model is not None and cls._text_emb is not None:
                return

            prompts_path = cls._default_prompts_path()
            cls._labels, cls._reverse = cls._load_prompts(prompts_path)

            model_name = cls._model_name()
            pretrained = cls._pretrained_name()
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"

            cls._model, cls._preprocess = open_clip.create_model_from_pretrained(model_name, pretrained)
            cls._model.eval().to(cls._device)

            tokenizer = open_clip.get_tokenizer(model_name)
            with torch.no_grad():
                text_tokens = tokenizer(cls._labels)
                text_emb = cls._model.encode_text(text_tokens.to(cls._device))
                cls._text_emb = F.normalize(text_emb, dim=-1)

    @classmethod
    def score(cls, image_path: str) -> Dict[str, Any]:
        cls._ensure_runtime()
        if cls._model is None or cls._preprocess is None or cls._text_emb is None:
            raise RuntimeError("CLIP runtime not initialized.")

        image = cls._preprocess(Image.open(image_path).convert("RGB"))
        image = image.unsqueeze(0).to(cls._device)
        with torch.no_grad():
            image_emb = cls._model.encode_image(image)
            image_emb = F.normalize(image_emb, dim=-1)
            logits = (100.0 * image_emb @ cls._text_emb.T).squeeze(0)
            probs = logits.softmax(dim=-1).cpu()

        candidate_labels = cls._labels
        reverse_map = cls._reverse
        prompt_scores = {label: float(probs[idx]) for idx, label in enumerate(candidate_labels)}

        group_scores: Dict[str, float] = {}
        for prompt, score in prompt_scores.items():
            group = reverse_map[prompt]
            group_scores[group] = max(group_scores.get(group, 0.0), score)

        top_idx = int(probs.argmax())
        top_label = candidate_labels[top_idx]
        top_group = reverse_map[top_label]

        return {
            "clip_top_label": top_label,
            "clip_top_group": top_group,
            "clip_top_score": float(probs[top_idx]),
            "score_document": group_scores.get("document", 0.0),
            "score_non_document": group_scores.get("non_document", 0.0),
            "doc_minus_non_doc": group_scores.get("document", 0.0) - group_scores.get("non_document", 0.0),
        }


class DoclingLayoutAnalyzer:
    """Tool: High-quality layout analysis via Docling (figures, tables)"""

    _converter = None
    _converter_available: Optional[bool] = None
    _converter_lock = threading.Lock()
    _docling_version: Optional[str] = None

    @classmethod
    def preload(cls) -> bool:
        """Warm converter so the first call is fast."""
        return cls._get_converter() is not None

    @classmethod
    def _get_converter(cls):
        if cls._converter_available is False:
            return None

        with cls._converter_lock:
            if cls._converter is not None:
                return cls._converter

            try:
                from docling.document_converter import (
                    DocumentConverter,
                    ImageFormatOption,
                    PdfFormatOption,
                )
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import (
                    ThreadedPdfPipelineOptions,
                    LayoutOptions,
                )
                from docling.datamodel.accelerator_options import AcceleratorOptions
                import importlib.metadata as importlib_metadata
            except ImportError:
                cls._converter_available = False
                return None

            for name in ("docling", "docling.pipeline", "docling.backend"):
                logging.getLogger(name).setLevel(logging.WARNING)

            pipeline_options = ThreadedPdfPipelineOptions(
                do_ocr=False,
                do_table_structure=True,
                do_formula_enrichment=False,
                do_code_enrichment=False,
                generate_page_images=False,
                generate_picture_images=False,
                layout_options=LayoutOptions(),
                accelerator_options=AcceleratorOptions(device="cpu"),
                ocr_batch_size=1,
                layout_batch_size=1,
                table_batch_size=1,
                batch_polling_interval_seconds=0.1,
                queue_max_size=4,
            )

            image_option = ImageFormatOption(pipeline_options=pipeline_options)
            format_options = {InputFormat.IMAGE: image_option}
            allowed_formats = [InputFormat.IMAGE]

            try:
                format_options[InputFormat.PDF] = PdfFormatOption(
                    pipeline_options=pipeline_options.model_copy()
                )
                allowed_formats.append(InputFormat.PDF)
            except Exception:
                pass

            try:
                cls._converter = DocumentConverter(
                    allowed_formats=allowed_formats,
                    format_options=format_options,
                )
                cls._docling_version = importlib_metadata.version("docling")
                cls._converter_available = True
            except Exception:
                cls._converter = None
                cls._converter_available = False

            return cls._converter

    @staticmethod
    def _serialize_bbox(
        bbox,
        page_no: int,
        page_size: Tuple[float, float],
        label: str,
        extra: Dict[str, Any],
        seen: set,
        regions: List[Dict[str, Any]],
    ) -> bool:
        page_w, page_h = page_size
        if page_w <= 0 or page_h <= 0:
            return False

        try:
            bbox_top_left = bbox.to_top_left_origin(page_h)
        except Exception:
            bbox_top_left = bbox

        left = max(0.0, min(page_w, bbox_top_left.l))
        top = max(0.0, min(page_h, bbox_top_left.t))
        right = max(left + 1.0, min(page_w, bbox_top_left.r))
        bottom = max(top + 1.0, min(page_h, bbox_top_left.b))

        width = max(1.0, right - left)
        height = max(1.0, bottom - top)

        signature = (
            label,
            int(page_no),
            int(round(left)),
            int(round(top)),
            int(round(right)),
            int(round(bottom)),
        )
        if signature in seen:
            return False
        seen.add(signature)

        normalized = [
            round(left / page_w, 6),
            round(top / page_h, 6),
            round(width / page_w, 6),
            round(height / page_h, 6),
        ]

        region = {
            "type": label,
            "page": int(page_no),
            "bbox": [
                int(round(left)),
                int(round(top)),
                int(round(width)),
                int(round(height)),
            ],
            "normalized_bbox": normalized,
            "confidence": float(extra.pop("confidence", 0.95)),
            "source": "docling",
        }
        region.update(extra)
        regions.append(region)
        return True

    @classmethod
    def execute(
        cls,
        image_path: str,
        include_figures: bool = True,
        include_tables: bool = True,
    ) -> ToolResult:
        if not os.path.exists(image_path):
            return ToolResult(False, {}, f"File not found: {image_path}")

        converter = cls._get_converter()
        if converter is None:
            return ToolResult(
                False,
                {},
                "Docling layout analyzer unavailable; install docling to enable it.",
            )

        start_ts = time.time()
        try:
            conversion = converter.convert(image_path)
        except Exception as exc:
            return ToolResult(False, {}, f"Docling conversion failed: {exc}")
        runtime_ms = (time.time() - start_ts) * 1000.0

        doc = conversion.document
        if not getattr(doc, "pages", None):
            return ToolResult(False, {}, "Docling returned no page information.")

        page_sizes: Dict[int, Tuple[float, float]] = {}
        for page_no, page in doc.pages.items():
            if page.size is None:
                continue
            page_sizes[int(page_no)] = (
                float(page.size.width),
                float(page.size.height),
            )

        if not page_sizes:
            return ToolResult(False, {}, "Docling produced pages without dimensions.")

        regions: List[Dict[str, Any]] = []
        seen_signatures: set = set()
        figure_count = 0
        table_count = 0

        if include_figures:
            for picture in getattr(doc, "pictures", []):
                if not getattr(picture, "prov", None):
                    continue
                caption = ""
                try:
                    caption = picture.caption_text(doc).strip()
                except Exception:
                    caption = ""
                extra = {
                    "confidence": 0.95,
                    "docling_label": getattr(
                        getattr(picture, "label", ""), "value", str(getattr(picture, "label", ""))
                    ),
                }
                if caption:
                    extra["caption"] = caption

                for provenance in picture.prov:
                    page_no = int(provenance.page_no)
                    if page_no not in page_sizes:
                        continue
                    if cls._serialize_bbox(
                        provenance.bbox,
                        page_no,
                        page_sizes[page_no],
                        "figure",
                        extra.copy(),
                        seen_signatures,
                        regions,
                    ):
                        figure_count += 1

        if include_tables:
            for table in getattr(doc, "tables", []):
                if not getattr(table, "prov", None):
                    continue

                extra = {
                    "confidence": 0.9,
                    "docling_label": getattr(
                        getattr(table, "label", ""), "value", str(getattr(table, "label", ""))
                    ),
                    "rows": getattr(getattr(table, "data", None), "num_rows", None),
                    "cols": getattr(getattr(table, "data", None), "num_cols", None),
                }
                extra = {k: v for k, v in extra.items() if v is not None}

                for provenance in table.prov:
                    page_no = int(provenance.page_no)
                    if page_no not in page_sizes:
                        continue
                    if cls._serialize_bbox(
                        provenance.bbox,
                        page_no,
                        page_sizes[page_no],
                        "table",
                        extra.copy(),
                        seen_signatures,
                        regions,
                    ):
                        table_count += 1

        regions.sort(key=lambda r: (r["page"], r["bbox"][1], r["bbox"][0]))
        for idx, region in enumerate(regions):
            region["region_id"] = idx

        page_dimensions_serializable = {
            page_no: {
                "width": int(round(size[0])),
                "height": int(round(size[1])),
            }
            for page_no, size in page_sizes.items()
        }

        if figure_count == 0 and table_count == 0:
            logger.info(
                "DoclingLayoutAnalyzer: no figure/table regions detected for %s",
                image_path,
            )

        payload = {
            "model": "docling",
            "model_version": cls._docling_version,
            "runtime_ms": round(runtime_ms, 2),
            "input_path": image_path,
            "total_regions": len(regions),
            "figure_regions": figure_count,
            "table_regions": table_count,
            "page_dimensions": page_dimensions_serializable,
            "regions": regions,
        }

        return ToolResult(True, payload)


class FormulaDetector:
    """Tool: Detect mathematical formulas using Surya layout with heuristic fallback."""

    _MIN_EDGE = 12
    _DEFAULT_CONFIDENCE = 0.25

    _HEURISTIC_MIN_CONFIDENCE = 0.45
    _HEURISTIC_PADDING = 8

    @classmethod
    def _detect_with_surya(
        cls,
        image: Image.Image,
        min_confidence: float,
    ) -> Tuple[List[Dict[str, Any]], Optional[Any]]:
        if not SuryaRuntime.available():
            return [], None

        img_width, img_height = image.size
        try:
            layout_predictor = SuryaRuntime.get_layout()
            layout_result = layout_predictor([image])[0]
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.debug("FormulaDetector: Surya layout failed: %s", exc)
            return []

        regions: List[Dict[str, Any]] = []
        for box in layout_result.bboxes:
            if box.label != "Equation":
                continue
            confidence = float(box.confidence or 0.0)
            if confidence < min_confidence:
                continue

            x1, y1, x2, y2 = box.bbox
            x1 = int(round(max(0, min(x1, img_width))))
            y1 = int(round(max(0, min(y1, img_height))))
            x2 = int(round(max(0, min(x2, img_width))))
            y2 = int(round(max(0, min(y2, img_height))))
            if x2 <= x1 or y2 <= y1:
                continue

            w = x2 - x1
            h = y2 - y1
            if w < cls._MIN_EDGE or h < cls._MIN_EDGE:
                continue

            regions.append(
                {
                    "type": "formula",
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "confidence": confidence,
                    "source": "surya-layout",
                    "polygon": [[float(px), float(py)] for px, py in box.polygon],
                    "position": int(getattr(box, "position", len(regions))),
                    "top_k": {k: float(v) for k, v in (box.top_k or {}).items()},
                }
            )
        return regions, layout_result

    _MATH_DELIMITERS = {"$", "\\", "{", "}", "[", "]"}
    _MATH_SYMBOLS = set("=+*/^_<>±≈≠∑∏∫√∞∂∇→⇒⇔≤≥≅≃≡⊂⊆⊃⊇∈∉∪∩⊕⊗∥⊥…·×÷≪≫∝∠∴°")
    _MATH_KEYWORDS = {
        "frac",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "log",
        "ln",
        "lim",
        "sum",
        "prod",
        "min",
        "max",
        "arg",
        "df",
        "dx",
        "dy",
        "dt",
        "partial",
        "gamma",
        "lambda",
        "omega",
        "alpha",
        "beta",
        "theta",
    }

    @classmethod
    def _heuristic_detect(
        cls,
        image: Image.Image,
        min_confidence: float,
        padding: int,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """Heuristic fallback using Tesseract line statistics."""
        width, height = image.size
        if roi:
            left, top, right, bottom = roi
            crop = image.crop((left, top, right, bottom))
            offset = (left, top)
        else:
            crop = image
            offset = (0, 0)

        try:
            ocr_data = pytesseract.image_to_data(
                crop,
                config="--psm 6 --oem 3",
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            logger.debug("FormulaDetector: heuristic OCR failed: %s", exc)
            return []

        n = len(ocr_data.get("text", []))
        if n == 0:
            return []

        lines: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
        for idx in range(n):
            text = (ocr_data["text"][idx] or "").strip()
            conf_raw = ocr_data.get("conf", ["-1"] * n)[idx]
            try:
                confidence = float(conf_raw)
            except (TypeError, ValueError):
                confidence = -1.0

            if not text and confidence < 0:
                continue

            left_val = int(ocr_data.get("left", [0] * n)[idx])
            top_val = int(ocr_data.get("top", [0] * n)[idx])
            w_val = int(ocr_data.get("width", [0] * n)[idx])
            h_val = int(ocr_data.get("height", [0] * n)[idx])
            if w_val <= 0 or h_val <= 0:
                continue

            key = (
                int(ocr_data.get("block_num", [0] * n)[idx]),
                int(ocr_data.get("par_num", [0] * n)[idx]),
                int(ocr_data.get("line_num", [0] * n)[idx]),
            )

            entry = lines.setdefault(
                key,
                {
                    "words": [],
                    "left": left_val,
                    "top": top_val,
                    "right": left_val + w_val,
                    "bottom": top_val + h_val,
                },
            )
            entry["words"].append({"text": text, "conf": confidence})
            entry["left"] = min(entry["left"], left_val)
            entry["top"] = min(entry["top"], top_val)
            entry["right"] = max(entry["right"], left_val + w_val)
            entry["bottom"] = max(entry["bottom"], top_val + h_val)

        candidates: List[Dict[str, Any]] = []
        for line_meta in lines.values():
            line_words = [w for w in line_meta["words"] if w["text"]]
            if not line_words:
                continue

            line_text = " ".join(w["text"] for w in line_words)
            clean_text = line_text.replace(" ", "")
            if not clean_text:
                continue

            letters = sum(ch.isalpha() for ch in clean_text)
            digits = sum(ch.isdigit() for ch in clean_text)
            specials = sum(
                1 for ch in clean_text if not ch.isalnum() and ch not in {".", ",", ":"}
            )
            latex_delims = sum(1 for ch in clean_text if ch in cls._MATH_DELIMITERS)
            symbol_hits = sum(1 for ch in clean_text if ch in cls._MATH_SYMBOLS)
            normalized_text = line_text.lower().replace("\\", " \\")
            token_set = {
                token for token in re.split(r"[^a-z]+", normalized_text) if token
            }
            keyword_hits = sum(1 for token in cls._MATH_KEYWORDS if token in token_set)

            math_char_count = sum(
                1 for ch in clean_text if ch in cls._MATH_SYMBOLS or ch in {"\\", "^", "_"}
            )
            has_equation_like = False
            for idx, ch in enumerate(clean_text):
                if ch != "=":
                    continue
                left_char = clean_text[idx - 1] if idx > 0 else ""
                right_char = clean_text[idx + 1] if idx + 1 < len(clean_text) else ""
                left_valid = left_char.isalnum() or left_char in {")", "]"}
                right_valid = right_char.isalnum() or right_char in {"(", "["}
                if left_valid and right_valid:
                    has_equation_like = True
                    break

            short_expression = (
                len(clean_text) <= 24
                and any(op in clean_text for op in {"+", "×", "÷", "*", "/", "^"})
                and letters >= 1
                and digits > 0
                and math_char_count >= 2
            )

            looks_latex = (
                "\\" in line_text
                or latex_delims > 0
                or keyword_hits > 0
                or "^" in clean_text
            )

            looks_numeric = digits > 0 and math_char_count >= 1
            has_symbols = math_char_count >= 2
            has_core_ops = any(ch in clean_text for ch in "=+×÷*/^\\{}[]")

            if letters >= 5 and digits == 0 and not looks_latex and not has_equation_like:
                continue

            if not (looks_latex or looks_numeric or has_equation_like or has_core_ops):
                continue

            qualifies = False
            if has_equation_like and (looks_latex or looks_numeric or has_symbols):
                qualifies = True
            elif looks_latex and (has_symbols or looks_numeric):
                qualifies = True
            elif short_expression:
                qualifies = True

            if not qualifies:
                continue

            score = 0.0
            if has_equation_like:
                score += 0.35
            if looks_latex:
                score += 0.35
                if has_symbols:
                    score += 0.15
                if digits > 0:
                    score += 0.15
            if symbol_hits >= 2:
                score += 0.15
            if short_expression:
                score += 0.15
            if looks_numeric:
                score += 0.1
            if math_char_count / max(1, len(clean_text)) > 0.25:
                score += 0.1
            if not looks_latex and not has_equation_like:
                score -= 0.2
            if digits > 0 and not (looks_latex or has_equation_like or has_core_ops):
                score -= 0.1

            score = max(0.0, min(1.0, score))
            if score < min_confidence:
                continue

            left_val = max(0, line_meta["left"] - padding) + offset[0]
            top_val = max(0, line_meta["top"] - padding) + offset[1]
            right_val = min(width, line_meta["right"] + padding + offset[0])
            bottom_val = min(height, line_meta["bottom"] + padding + offset[1])

            if (right_val - left_val) < cls._MIN_EDGE or (bottom_val - top_val) < cls._MIN_EDGE:
                continue

            candidates.append(
                {
                    "type": "formula",
                    "bbox": [
                        float(left_val),
                        float(top_val),
                        float(right_val - left_val),
                        float(bottom_val - top_val),
                    ],
                    "confidence": score,
                    "source": "heuristic-ocr",
                    "text_hint": line_text[:128],
                }
            )

        return candidates

    @staticmethod
    def _bbox_iou(a: List[float], b: List[float]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        a_right, a_bottom = ax + aw, ay + ah
        b_right, b_bottom = bx + bw, by + bh

        inter_left = max(ax, bx)
        inter_top = max(ay, by)
        inter_right = min(a_right, b_right)
        inter_bottom = min(a_bottom, b_bottom)
        if inter_right <= inter_left or inter_bottom <= inter_top:
            return 0.0
        inter_area = (inter_right - inter_left) * (inter_bottom - inter_top)
        area_a = aw * ah
        area_b = bw * bh
        union = max(area_a + area_b - inter_area, 1e-6)
        return inter_area / union

    @classmethod
    def execute(
        cls,
        image_path: str,
        min_confidence: float = _DEFAULT_CONFIDENCE,
        enable_fallback: bool = True,
    ) -> ToolResult:
        if not os.path.exists(image_path):
            return ToolResult(False, {}, f"File not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            return ToolResult(False, {}, f"Failed to load image: {exc}")

        img_width, img_height = image.size

        surya_regions: List[Dict[str, Any]] = []
        layout_result = None
        if SuryaRuntime.available():
            surya_regions, layout_result = cls._detect_with_surya(image, min_confidence)

        regions: List[Dict[str, Any]] = []
        regions.extend(surya_regions)

        # Decide whether to run fallback
        fallback_needed = enable_fallback and (
            not surya_regions
            or sum((r["bbox"][2] * r["bbox"][3]) for r in surya_regions) > 0.4 * (img_width * img_height)
        )

        heuristic_sources: List[Dict[str, Any]] = []

        if fallback_needed:
            heuristic_sources.extend(
                cls._heuristic_detect(
                    image,
                    min_confidence=cls._HEURISTIC_MIN_CONFIDENCE,
                    padding=cls._HEURISTIC_PADDING,
                )
            )

        if enable_fallback and layout_result is not None:
            img_area = img_width * img_height
            for box in layout_result.bboxes:
                label = (box.label or "").lower()
                if label not in {"text", "caption", "listitem", "pagefooter", "pageheader"}:
                    continue
                x1, y1, x2, y2 = box.bbox
                x1 = int(round(max(0, min(x1, img_width))))
                y1 = int(round(max(0, min(y1, img_height))))
                x2 = int(round(max(0, min(x2, img_width))))
                y2 = int(round(max(0, min(y2, img_height))))
                if x2 - x1 < cls._MIN_EDGE * 3 or y2 - y1 < cls._MIN_EDGE * 2:
                    continue
                area_ratio = ((x2 - x1) * (y2 - y1)) / max(img_area, 1)
                if area_ratio < 0.01 or area_ratio > 0.65:
                    continue
                roi_regions = cls._heuristic_detect(
                    image,
                    min_confidence=cls._HEURISTIC_MIN_CONFIDENCE,
                    padding=cls._HEURISTIC_PADDING,
                    roi=(x1, y1, x2, y2),
                )
                heuristic_sources.extend(roi_regions)

        for candidate in heuristic_sources:
            has_overlap = any(
                cls._bbox_iou(candidate["bbox"], existing["bbox"]) > 0.4
                for existing in regions
            )
            if not has_overlap:
                regions.append(candidate)

        for idx, region in enumerate(regions):
            region["region_id"] = idx
            bbox = region["bbox"]
            region["normalized_bbox"] = [
                bbox[0] / img_width if img_width else 0.0,
                bbox[1] / img_height if img_height else 0.0,
                bbox[2] / img_width if img_width else 0.0,
                bbox[3] / img_height if img_height else 0.0,
            ]

        payload = {
            "total_formulas": len(regions),
            "regions": regions,
            "image_dimensions": {"width": img_width, "height": img_height},
            "sources": sorted({region.get("source", "unknown") for region in regions}),
        }
        return ToolResult(True, payload)


class FormulaRecognizer:
    """Tool: Convert detected formulas into LaTeX using Surya's Texify pipeline."""

    _DEFAULT_PADDING = 12
    _MIN_EDGE = 10

    @classmethod
    def execute(
        cls,
        image_path: str,
        regions: Optional[List[Dict[str, Any]]] = None,
        max_regions: int = 12,
        padding: int = _DEFAULT_PADDING,
    ) -> ToolResult:
        if not SuryaRuntime.available():
            return ToolResult(
                False,
                {},
                "Formula recognition unavailable. Install surya-ocr to enable Texify.",
            )

        if not os.path.exists(image_path):
            return ToolResult(False, {}, f"File not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            return ToolResult(False, {}, f"Failed to load image: {exc}")

        if regions is None:
            detection_result = FormulaDetector.execute(image_path)
            if not detection_result.success:
                return detection_result
            detector_regions = detection_result.data.get("regions", [])
        else:
            detector_regions = regions

        if not detector_regions:
            return ToolResult(
                True,
                {"total_recognized": 0, "formulas": [], "used_regions": 0},
            )

        img_width, img_height = image.size
        prepared: List[Dict[str, Any]] = []
        for region in detector_regions[:max_regions]:
            bbox = region.get("bbox")
            if not bbox or len(bbox) != 4:
                continue

            x, y, w, h = bbox
            left = max(0, int(round(x - padding)))
            top = max(0, int(round(y - padding)))
            right = min(img_width, int(round(x + w + padding)))
            bottom = min(img_height, int(round(y + h + padding)))

            if right - left < cls._MIN_EDGE or bottom - top < cls._MIN_EDGE:
                continue

            prepared.append(
                {
                    "region": region,
                    "crop_bbox": [left, top, right, bottom],
                }
            )

        if not prepared:
            return ToolResult(
                True,
                {"total_recognized": 0, "formulas": [], "used_regions": 0},
            )

        try:
            recognition = SuryaRuntime.get_recognition()
        except Exception as exc:
            return ToolResult(False, {}, f"Failed to initialise Surya recognition model: {exc}")

        crop_list = [meta["crop_bbox"] for meta in prepared]
        try:
            ocr_results = recognition(
                [image],
                task_names=[SuryaTaskNames.block_without_boxes],
                bboxes=[crop_list],
                math_mode=True,
                sort_lines=False,
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            return ToolResult(False, {}, f"Surya Texify inference failed: {exc}")

        text_lines = ocr_results[0].text_lines if ocr_results else []
        formulas: List[Dict[str, Any]] = []
        for idx, meta in enumerate(prepared):
            line = text_lines[idx] if idx < len(text_lines) else None
            latex = (line.text or "").strip() if line else ""
            rec_conf = float(getattr(line, "confidence", 0.0)) if line else 0.0
            det_conf = float(meta["region"].get("confidence", 0.0) or 0.0)
            combined_conf = det_conf * rec_conf if rec_conf > 0 else det_conf

            left, top, right, bottom = meta["crop_bbox"]
            width_px = right - left
            height_px = bottom - top
            normalized = [
                left / img_width if img_width else 0.0,
                top / img_height if img_height else 0.0,
                width_px / img_width if img_width else 0.0,
                height_px / img_height if img_height else 0.0,
            ]

            entry = {
                "region_id": meta["region"].get("region_id"),
                "bbox": [float(left), float(top), float(width_px), float(height_px)],
                "normalized_bbox": normalized,
                "latex": latex,
                "confidence": round(min(max(combined_conf, 0.0), 1.0), 3),
                "recognition_confidence": rec_conf,
                "detector_confidence": det_conf,
                "source": "surya-texify",
            }
            meta["region"]["latex"] = latex
            formulas.append(entry)

        payload = {
            "total_recognized": sum(1 for item in formulas if item["latex"]),
            "formulas": formulas,
            "used_regions": len(prepared),
        }
        return ToolResult(True, payload)



class RegionVisualizer:
    """Tool: Render bounding boxes for detected regions onto an image."""

    _DEFAULT_COLORS = {
        "figure": (58, 151, 246),
        "table": (46, 204, 113),
        "caption": (155, 89, 182),
        "formula": (231, 76, 60),
    }

    @staticmethod
    def _resolve_color(region_type: Optional[str], index: int) -> Tuple[int, int, int]:
        if region_type:
            color = RegionVisualizer._DEFAULT_COLORS.get(region_type.lower())
            if color:
                return color
        palette = [
            (241, 196, 15),
            (52, 152, 219),
            (46, 204, 113),
            (231, 76, 60),
            (142, 68, 173),
            (230, 126, 34),
            (26, 188, 156),
        ]
        return palette[index % len(palette)]

    @staticmethod
    def execute(
        image_path: str,
        regions: List[Dict[str, Any]],
        output_path: Optional[str] = None,
        show_labels: bool = True,
        line_width: int = 4,
        label_fill_alpha: float = 0.7,
    ) -> ToolResult:
        if not os.path.exists(image_path):
            return ToolResult(False, {}, f"Image file not found: {image_path}")
        if not isinstance(regions, list) or not regions:
            return ToolResult(False, {}, "No regions provided to visualize.")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            return ToolResult(False, {}, f"Failed to load image: {exc}")

        draw = ImageDraw.Draw(image, "RGBA")
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

        valid_boxes = 0
        for idx, region in enumerate(regions):
            bbox = region.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x, y, w, h = bbox
            try:
                x1 = int(round(x))
                y1 = int(round(y))
                x2 = int(round(x + w))
                y2 = int(round(y + h))
            except TypeError:
                continue

            if x2 <= x1 or y2 <= y1:
                continue

            color = RegionVisualizer._resolve_color(region.get("type"), idx)
            outline = (*color, 255)
            draw.rectangle([x1, y1, x2, y2], outline=outline, width=max(1, line_width))

            if show_labels:
                label_parts = []
                region_type = region.get("type")
                if region_type:
                    label_parts.append(str(region_type))
                region_id = region.get("region_id")
                if region_id is not None:
                    label_parts.append(f"#{region_id}")
                source = region.get("source")
                if source:
                    label_parts.append(f"{source}")
                if not label_parts:
                    label_parts.append(f"region{idx}")
                text = " ".join(label_parts)
                if font and text:
                    try:
                        text_bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    except Exception:
                        continue
                    padding = 4
                    label_height = text_height + padding * 2
                    label_width = text_width + padding * 2
                    label_x1 = x1
                    label_y1 = max(0, y1 - label_height)
                    label_x2 = label_x1 + label_width
                    label_y2 = label_y1 + label_height
                    fill = (*color, int(255 * max(0.0, min(1.0, label_fill_alpha))))
                    draw.rectangle([label_x1, label_y1, label_x2, label_y2], fill=fill)
                    draw.text(
                        (label_x1 + padding, label_y1 + padding),
                        text,
                        fill=(0, 0, 0, 255),
                        font=font,
                    )

            valid_boxes += 1

        if valid_boxes == 0:
            return ToolResult(False, {}, "No valid bounding boxes were found in regions.")

        if output_path is None:
            stem, ext = os.path.splitext(image_path)
            output_path = f"{stem}_regions.png"
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        try:
            image.save(output_path)
        except Exception as exc:  # pragma: no cover - unexpected I/O errors
            return ToolResult(False, {}, f"Failed to save annotated image: {exc}")

        return ToolResult(
            True,
            {
                "output_path": output_path,
                "regions_rendered": valid_boxes,
                "image_path": image_path,
            },
        )


class ChartDataExtractor:
    """Tool: Extract chart data from cropped regions using MiniCPM-V via llama-cpp."""

    _MAX_ATTEMPTS = 2
    _NORMALIZER_MODEL = os.getenv("CHART_JSON_NORMALIZER_MODEL", "gpt-4o")

    _SCHEMA_REMINDER = (
        "Respond with STRICT JSON using these keys: status, reason, title, axes, legend, series, annotations, summary. "
        "When the chart is legible, set status to 'success' and reason to null. When not, set status to 'failure' and reason to a short explanation, "
        "but still include the other keys with null or empty values. Each axis entry must include label, units (null if absent), and range as an object like {\"min\": <number or null>, \"max\": <number or null>}. "
        "Each series entry must have a name and a points array with numeric x/y pairs (floats or ints). Legend is a list of the visible series names. Annotations is a list of text callouts (can be empty). Summary must mention the concrete numeric values you extracted."
    )

    _SCHEMA_EXAMPLE = (
        '{"status":"success","reason":null,"title":"Chart title or null","axes":[{"label":"X Axis","units":"iterations","range":{"min":0,"max":10}},{"label":"Y Axis","units":"accuracy","range":{"min":0.8,"max":1.0}}],'
        '"legend":["Method A","Method B"],'
        '"series":[{"name":"Method A","points":[{"x":0,"y":0.81},{"x":10,"y":0.93}]},{"name":"Method B","points":[{"x":0,"y":0.8},{"x":10,"y":0.88}]}],'
        '"annotations":[],"summary":"Method A improves from 0.81 to 0.93 accuracy between 0 and 10 iterations while Method B trails at 0.8→0.88."}'
    )

    @staticmethod
    def execute(
        image_path: str,
        bbox: Optional[List[float]] = None,
        normalized_bbox: Optional[List[float]] = None,
        page_dimensions: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
        region_id: Optional[str] = None,
        model_path: Optional[str] = None,
        mmproj_path: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ToolResult:
        if run_minicpm_vqa is None or MiniCPMVEngine is None or not MiniCPMVEngine.is_available():
            return ToolResult(
                False,
                {},
                "Chart extraction unavailable. Install llama-cpp-python with CLIP support.",
            )

        if question is None:
            question = ChartDataExtractor._default_question()

        model_path = model_path or os.environ.get("MINICPM_MODEL_PATH")
        mmproj_path = mmproj_path or os.environ.get("MINICPM_MMPROJ_PATH")
        if not model_path or not mmproj_path:
            return ToolResult(
                False,
                {},
                "MiniCPM-V model paths not configured. Set MINICPM_MODEL_PATH and "
                "MINICPM_MMPROJ_PATH environment variables.",
            )

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            return ToolResult(False, {}, f"Failed to load image for chart extraction: {exc}")

        crop_box = ChartDataExtractor._compute_crop_box(
            image_size=image.size,
            bbox=bbox,
            normalized_bbox=normalized_bbox,
            page_dimensions=page_dimensions,
        )
        if crop_box is None:
            return ToolResult(False, {}, "Invalid or missing bounding box for chart extraction.")

        x1, y1, x2, y2 = crop_box
        chart_crop = image.crop((x1, y1, x2, y2))

        prompt_attempts = ChartDataExtractor._build_prompt_attempts(question)
        parsed_json: Optional[Dict[str, Any]] = None
        parse_error: Optional[str] = None
        answer_text = ""
        raw_message = ""
        vqa_result: Optional[Dict[str, Any]] = None
        last_prompt_text: Optional[str] = None
        parse_source = "direct"

        for prompt in prompt_attempts:
            last_prompt_text = prompt["question"]
            try:
                vqa_result = run_minicpm_vqa(
                    chart_crop,
                    prompt["question"],
                    model_path=model_path,
                    mmproj_path=mmproj_path,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=prompt.get("system_prompt"),
                )
            except Exception as exc:
                return ToolResult(False, {}, f"Chart VQA failed: {exc}")

            answer_text = (vqa_result.get("answer") or "").strip()
            raw_message = (
                vqa_result.get("raw_response", {})
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            parsed_json, parse_error = ChartDataExtractor._parse_chart_json(answer_text)
            if parsed_json is None:
                parsed_json, parse_error = ChartDataExtractor._parse_chart_json(raw_message)

            if parsed_json is not None:
                break

        if parsed_json is None and (answer_text or raw_message):
            fallback_json = ChartDataExtractor._llm_structured_from_text(
                answer_text,
                raw_message,
                question,
            )
            if fallback_json is not None:
                parsed_json = fallback_json
                parse_source = "llm_fallback"

        if parsed_json is None or vqa_result is None:
            return ToolResult(
                False,
                {
                    "region_id": region_id,
                    "question": question,
                    "prompt_used": last_prompt_text,
                    "raw_answer": answer_text,
                    "raw_message": raw_message,
                    "model_path": vqa_result.get("model_path") if vqa_result else model_path,
                    "mmproj_path": vqa_result.get("mmproj_path") if vqa_result else mmproj_path,
                    "attempts": len(prompt_attempts),
                    "parse_source": parse_source,
                },
                parse_error or "Chart VQA returned non-JSON output",
            )

        status = parsed_json.get("status")
        if status != "success":
            reason = parsed_json.get("reason") or "chart unreadable"
            return ToolResult(
                False,
                {
                    "region_id": region_id,
                    "question": question,
                    "status": status,
                    "reason": reason,
                    "raw_answer": answer_text,
                    "raw_message": raw_message,
                },
                f"Chart model reported status '{status}': {reason}",
            )

        if not ChartDataExtractor._has_meaningful_series(parsed_json):
            return ToolResult(
                False,
                {
                    "region_id": region_id,
                    "question": question,
                    "status": status,
                    "raw_answer": answer_text,
                    "raw_message": raw_message,
                },
                "Chart extraction produced degenerate series",
            )

        data = {
            "region_id": region_id,
            "question": question,
            "crop_bbox": [x1, y1, x2, y2],
            "crop_dimensions": {"width": x2 - x1, "height": y2 - y1},
            "raw_answer": answer_text,
            "raw_message": raw_message,
            "parsed": parsed_json,
            "model_path": vqa_result.get("model_path"),
            "mmproj_path": vqa_result.get("mmproj_path"),
            "parse_source": parse_source,
        }
        return ToolResult(True, data)

    @staticmethod
    def _default_question() -> str:
        return (
            "Extract precise numeric data from this chart. "
            f"{ChartDataExtractor._SCHEMA_REMINDER} Sample JSON: {ChartDataExtractor._SCHEMA_EXAMPLE}"
        )

    @staticmethod
    def _system_prompt(retry: bool = False) -> str:
        prefix = (
            "You are ChartJSON, a data extraction assistant that MUST reply with valid JSON only. "
            "Never wrap output in Markdown or add commentary. "
        )
        if retry:
            prefix = (
                "RETRY: Your prior message was not pure JSON. Immediately output the JSON object now. "
            ) + prefix
        return prefix + ChartDataExtractor._SCHEMA_REMINDER + " Example: " + ChartDataExtractor._SCHEMA_EXAMPLE

    @staticmethod
    def _format_question(base_question: str, attempt: int) -> str:
        base = base_question.strip()
        reminder = ChartDataExtractor._SCHEMA_REMINDER + " Example: " + ChartDataExtractor._SCHEMA_EXAMPLE
        if attempt > 0:
            reminder = (
                "Second attempt: the previous response did not include JSON. "
                "Reply now with ONLY the JSON object. "
            ) + reminder
        return f"{base}\n\n{reminder}\nRemember: output JSON only."

    @classmethod
    def _build_prompt_attempts(cls, base_question: str) -> List[Dict[str, str]]:
        attempts: List[Dict[str, str]] = []
        for attempt in range(cls._MAX_ATTEMPTS):
            attempts.append(
                {
                    "question": cls._format_question(base_question, attempt),
                    "system_prompt": cls._system_prompt(retry=attempt > 0),
                }
            )
        return attempts

    @classmethod
    def _llm_structured_from_text(
        cls,
        answer_text: str,
        raw_message: str,
        question: str,
    ) -> Optional[Dict[str, Any]]:
        text = (answer_text or "").strip() or (raw_message or "").strip()
        if not text:
            return None
        if OpenAI is None:
            return None
        if not cls._NORMALIZER_MODEL:
            return None
        try:
            client = OpenAI()
        except Exception:
            return None

        instructions = (
            "Convert the assistant's free-form description of a chart into strict JSON. "
            f"{cls._SCHEMA_REMINDER} Respond with JSON only."
        )
        prompt = (
            "Original chart prompt: "
            + question
            + "\n\nAssistant output to convert:\n" + text
        )

        try:
            response = client.chat.completions.create(
                model=cls._NORMALIZER_MODEL,
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=400,
                temperature=0.0,
            )
        except Exception:
            return None

        content = response.choices[0].message.content if response.choices else None
        if not isinstance(content, str):
            return None
        parsed, _ = ChartDataExtractor._parse_chart_json(content)
        return parsed

    @staticmethod
    def _compute_crop_box(
        image_size: Tuple[int, int],
        bbox: Optional[List[float]],
        normalized_bbox: Optional[List[float]],
        page_dimensions: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[int, int, int, int]]:
        width, height = image_size
        if bbox and len(bbox) == 4:
            x, y, w, h = bbox
        elif normalized_bbox and len(normalized_bbox) == 4:
            norm_w = page_dimensions.get("width") if page_dimensions else width
            norm_h = page_dimensions.get("height") if page_dimensions else height
            x = normalized_bbox[0] * norm_w
            y = normalized_bbox[1] * norm_h
            w = normalized_bbox[2] * norm_w
            h = normalized_bbox[3] * norm_h
        else:
            return None

        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(width, int(round(x1 + max(1, w))))
        y2 = min(height, int(round(y1 + max(1, h))))

        if x2 - x1 < 5 or y2 - y1 < 5:
            return None

        return x1, y1, x2, y2

    @staticmethod
    def _parse_chart_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not text:
            return None, "Empty response"
        candidate = ChartDataExtractor._extract_json_block(text)
        if not candidate:
            return None, "No JSON object found"
        try:
            return json.loads(candidate), None
        except json.JSONDecodeError as exc:
            return None, f"Invalid JSON: {exc}"

    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        stack = []
        start_idx = None
        for idx, char in enumerate(text):
            if char == '{':
                if not stack:
                    start_idx = idx
                stack.append(char)
            elif char == '}' and stack:
                stack.pop()
                if not stack and start_idx is not None:
                    return text[start_idx: idx + 1]
        return None

    @staticmethod
    def _has_meaningful_series(parsed: Dict[str, Any]) -> bool:
        series = parsed.get("series")
        if not isinstance(series, list) or not series:
            return False
        numeric_points = []
        for entry in series:
            points = entry.get("points")
            if not isinstance(points, list):
                continue
            for point in points:
                if not isinstance(point, dict):
                    continue
                y_val = point.get("y")
                x_val = point.get("x")
                if isinstance(y_val, (int, float)) and isinstance(x_val, (int, float)):
                    numeric_points.append((entry.get("name"), x_val, y_val))
        return len(numeric_points) >= 2


class RegionDetector:
    """Tool: Detect and extract regions (figures, tables, captions) with grounding coordinates"""

    @staticmethod
    def execute(image_path: str, detect_figures: bool = True, detect_tables: bool = True, detect_captions: bool = True) -> ToolResult:
        try:
            image = cv2.imread(image_path)
            if image is None:
                return ToolResult(False, {}, f"Could not load image: {image_path}")

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape

            all_regions = []
            region_data = {}

            # 1. Detect figures (charts, graphs, diagrams)
            if detect_figures:
                figure_regions = RegionDetector._detect_figure_regions(image, gray)
                all_regions.extend(figure_regions)

            # 2. Detect tables (line grids)
            if detect_tables:
                table_regions = RegionDetector._detect_table_regions(image, gray)
                all_regions.extend(table_regions)

            # 3. Detect captions (text near figures/tables)
            if detect_captions and (detect_figures or detect_tables):
                caption_regions = RegionDetector._detect_caption_regions(image, gray, all_regions)
                all_regions.extend(caption_regions)

            # Remove overlapping regions and finalize
            final_regions = RegionDetector._remove_overlapping_regions(all_regions)

            # Assign unique IDs
            for i, region in enumerate(final_regions):
                region["region_id"] = i
                region_data[f"region_{i}"] = region

            return ToolResult(True, {
                "total_regions": len(final_regions),
                "regions": final_regions,
                "region_data": region_data,
                "image_dimensions": {"width": int(width), "height": int(height)}
            })
        except Exception as e:
            return ToolResult(False, {}, f"Region detection failed: {str(e)}")

    @staticmethod
    def _detect_figure_regions(image, gray):
        """Detect figure/chart regions using high-contrast rectangular areas"""
        figures = []

        # Edge detection to find high-contrast boundaries
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Dilate edges to connect nearby boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter by area (must be substantial)
            area = cv2.contourArea(contour)
            if area < 8000:  # Minimum figure size
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio (reasonable figure proportions)
            aspect_ratio = h / w if w > 0 else 0
            if not (0.3 <= aspect_ratio <= 3.0):  # Too thin/wide = likely not a figure
                continue

            # Check if region has good edge density (indicates charts/graphs)
            roi_edges = edges[y:y+h, x:x+w]
            edge_density = np.sum(roi_edges > 0) / (w * h)

            if edge_density > 0.02:  # Has sufficient internal structure
                # Check if it's not just a text block
                if RegionDetector._is_likely_figure(gray[y:y+h, x:x+w]):
                    # Extract text from the figure region (for chart labels, etc.)
                    figure_roi = image[y:y+h, x:x+w]
                    extracted_text, ocr_confidence = RegionDetector._extract_region_text(figure_roi, "figure")

                    figures.append({
                        "type": "figure",
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area": float(area),
                        "aspect_ratio": float(aspect_ratio),
                        "edge_density": float(edge_density),
                        "confidence": float(min(0.95, edge_density * 20)),  # Higher edge density = more confident
                        "extracted_text": extracted_text,
                        "ocr_confidence": float(ocr_confidence)
                    })

        return figures

    @staticmethod
    def _is_likely_figure(roi):
        """Check if a region is likely a figure vs pure text"""
        # Look for characteristics that distinguish figures from text:
        # 1. Variety in pixel intensities (graphs have lines, points, different shades)
        # 2. Non-uniform distribution (text is more uniform)

        # Calculate intensity variance
        intensity_var = np.var(roi)

        # Calculate horizontal vs vertical edge patterns
        sobelx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)

        horizontal_edges = np.sum(np.abs(sobelx))
        vertical_edges = np.sum(np.abs(sobely))

        # Figures typically have more diverse edge patterns
        edge_diversity = min(horizontal_edges, vertical_edges) / max(horizontal_edges, vertical_edges) if max(horizontal_edges, vertical_edges) > 0 else 0

        # Text has low intensity variance and uniform edge patterns
        # Figures have higher variance and diverse edge patterns
        return intensity_var > 1000 and edge_diversity > 0.3

    @staticmethod
    def _detect_table_regions(image, gray):
        """Detect table regions using multiple approaches"""
        tables = []

        # Approach 1: Find tables by detecting dense text regions that look structured
        tables.extend(RegionDetector._detect_tables_by_text_density(image, gray))

        # Approach 2: Find tables by line intersection (but with better filtering)
        tables.extend(RegionDetector._detect_tables_by_lines(image, gray))

        return tables

    @staticmethod
    def _detect_tables_by_text_density(image, gray):
        """Find tables by looking for dense, structured text regions"""
        tables = []

        # Apply threshold to get text regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Use larger kernels to group text into blocks (like table regions)
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))  # Connect words horizontally
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))  # Connect lines vertically

        # Apply morphological operations to form text blocks
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h)
        connected = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, kernel_v)

        # Find contours
        contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        img_height, img_width = gray.shape

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5000:  # Minimum table size
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Filter by dimensions and area ratio (more lenient for academic papers)
            area_ratio = area / (img_width * img_height)
            aspect_ratio = h / w if w > 0 else 0

            if (0.05 <= aspect_ratio <= 3.0 and    # More lenient aspect ratio
                0.005 <= area_ratio <= 0.4 and     # Allow smaller/larger regions
                w > 100 and h > 50):               # Minimum absolute size

                # Check if this region has table-like characteristics
                table_roi = gray[y:y+h, x:x+w]
                if RegionDetector._looks_like_table_region(table_roi):
                    # Extract text from the table region
                    extracted_text, ocr_confidence = RegionDetector._extract_region_text(image[y:y+h, x:x+w], "table")

                    # Calculate confidence based on size and characteristics
                    size_score = min(1.0, area_ratio * 5)  # Larger regions get higher scores
                    confidence = 0.6 + size_score * 0.3    # 0.6 to 0.9 range

                    tables.append({
                        "type": "table",
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area": float(area),
                        "aspect_ratio": float(aspect_ratio),
                        "area_ratio": float(area_ratio),
                        "confidence": float(confidence),
                        "extracted_text": extracted_text,
                        "ocr_confidence": float(ocr_confidence),
                        "detection_method": "text_density"
                    })

        return tables

    @staticmethod
    def _detect_tables_by_lines(image, gray):
        """Find tables using line detection (with improved filtering)"""
        tables = []

        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # Find intersections to locate table cells
        table_intersections = cv2.bitwise_and(horizontal_lines, vertical_lines)

        # Only proceed if we found actual line intersections
        if np.sum(table_intersections) > 100:  # Has enough intersection points
            # Use internal contours to find table regions
            contours, _ = cv2.findContours(table_intersections, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            img_height, img_width = gray.shape

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 2000:  # Minimum intersection cluster size
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                # Expand the region around intersections to capture full table
                padding = 50
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img_width - x, w + 2 * padding)
                h = min(img_height - y, h + 2 * padding)

                area = w * h
                area_ratio = area / (img_width * img_height)
                aspect_ratio = h / w if w > 0 else 0

                if (0.1 <= aspect_ratio <= 2.0 and
                    0.05 <= area_ratio <= 0.7 and
                    w > 200 and h > 100):

                    # Extract text from the table region
                    extracted_text, ocr_confidence = RegionDetector._extract_region_text(image[y:y+h, x:x+w], "table")

                    tables.append({
                        "type": "table",
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area": float(area),
                        "aspect_ratio": float(aspect_ratio),
                        "area_ratio": float(area_ratio),
                        "confidence": 0.7,
                        "extracted_text": extracted_text,
                        "ocr_confidence": float(ocr_confidence),
                        "detection_method": "line_intersections"
                    })

        return tables

    @staticmethod
    def _looks_like_table_region(roi):
        """Check if a region has table-like characteristics"""
        # Look for regular patterns that suggest tabular data

        # Check for horizontal line patterns (rows)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        horizontal_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_score = np.sum(horizontal_lines > 0) / roi.size

        # Check for vertical patterns (columns)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
        vertical_lines = cv2.morphologyEx(roi, cv2.MORPH_OPEN, vertical_kernel)
        vertical_score = np.sum(vertical_lines > 0) / roi.size

        # Check for regular text distribution
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        text_density = np.sum(binary > 0) / roi.size

        # Tables have moderate line patterns and text density
        return (horizontal_score > 0.01 and
                vertical_score > 0.005 and
                0.1 <= text_density <= 0.6)

    @staticmethod
    def _detect_caption_regions(image, gray, existing_regions):
        """Detect text captions near figures and tables"""
        captions = []

        # Find text regions using connected components
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find connected components (potential text areas)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

        for i in range(1, num_labels):  # Skip background (label 0)
            x, y, w, h, area = stats[i]

            # Filter by size (captions are moderate-sized text blocks)
            if not (200 <= area <= 15000):  # Too small/large
                continue

            # Filter by aspect ratio (captions are usually horizontal text)
            aspect_ratio = h / w if w > 0 else 0
            if aspect_ratio > 0.8:  # Too square/tall = likely not caption
                continue

            # Check if this text region is near a figure/table
            for region in existing_regions:
                if RegionDetector._is_near_region([x, y, w, h], region["bbox"]):
                    # Extract text from the caption region
                    caption_roi = image[y:y+h, x:x+w]
                    extracted_text, ocr_confidence = RegionDetector._extract_region_text(caption_roi, "caption")

                    captions.append({
                        "type": "caption",
                        "bbox": [int(x), int(y), int(w), int(h)],
                        "area": float(area),
                        "aspect_ratio": float(aspect_ratio),
                        "near_region": region["type"],
                        "confidence": 0.7,  # Moderate confidence for text detection
                        "extracted_text": extracted_text,
                        "ocr_confidence": float(ocr_confidence)
                    })
                    break

        return captions

    @staticmethod
    def _is_near_region(caption_bbox, region_bbox, max_distance=100):
        """Check if caption is near a figure/table region"""
        cx1, cy1, cw1, ch1 = caption_bbox
        cx2, cy2, cw2, ch2 = region_bbox

        # Calculate centers
        center1 = (cx1 + cw1//2, cy1 + ch1//2)
        center2 = (cx2 + cw2//2, cy2 + ch2//2)

        # Calculate distance between centers
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5

        return distance <= max_distance

    @staticmethod
    def _remove_overlapping_regions(regions):
        """Remove overlapping regions, keeping the one with higher confidence"""
        if not regions:
            return []

        # Sort by confidence (highest first)
        sorted_regions = sorted(regions, key=lambda r: r.get("confidence", 0), reverse=True)
        final_regions = []

        for region in sorted_regions:
            # Check if this region significantly overlaps with any already accepted region
            is_overlapping = False
            for accepted in final_regions:
                if RegionDetector._calculate_overlap(region["bbox"], accepted["bbox"]) > 0.5:
                    is_overlapping = True
                    break

            if not is_overlapping:
                final_regions.append(region)

        return final_regions

    @staticmethod
    def _extract_region_text(roi, region_type):
        """Extract text from a specific region using optimized OCR settings"""
        try:
            # Convert to grayscale if needed
            if len(roi.shape) == 3:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = roi.copy()

            # Apply preprocessing based on region type
            if region_type == "table":
                processed_roi = RegionDetector._preprocess_table_region(gray_roi)
                config = '--psm 6 --oem 3'  # Uniform text block
            elif region_type == "figure":
                processed_roi = RegionDetector._preprocess_figure_region(gray_roi)
                config = '--psm 8 --oem 3'  # Single word/line
            else:  # caption
                processed_roi = RegionDetector._preprocess_text_region(gray_roi)
                config = '--psm 6 --oem 3'  # Uniform text block

            # Extract text with confidence
            try:
                import pytesseract
                data = pytesseract.image_to_data(processed_roi, config=config, output_type=pytesseract.Output.DICT)

                # Filter out low-confidence detections
                words = []
                confidences = []
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) > 30:  # Minimum confidence threshold
                        word = data['text'][i].strip()
                        if word:
                            words.append(word)
                            confidences.append(int(data['conf'][i]))

                if words:
                    extracted_text = ' '.join(words)
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    return extracted_text, avg_confidence
                else:
                    return "", 0

            except Exception:
                # Fallback to simple text extraction
                text = pytesseract.image_to_string(processed_roi, config=config).strip()
                return text, 50 if text else 0  # Default confidence

        except Exception:
            return "", 0

    @staticmethod
    def _preprocess_table_region(gray_roi):
        """Preprocess table region for better OCR"""
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_roi)

        # Apply thresholding to get clean text
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    @staticmethod
    def _preprocess_figure_region(gray_roi):
        """Preprocess figure region for OCR (extract text labels)"""
        # Use adaptive threshold for varying backgrounds
        binary = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Dilate slightly to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        dilated = cv2.dilate(binary, kernel, iterations=1)

        return dilated

    @staticmethod
    def _preprocess_text_region(gray_roi):
        """Preprocess text/caption region for OCR"""
        # Simple denoising and thresholding for text
        denoised = cv2.GaussianBlur(gray_roi, (3, 3), 0)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    @staticmethod
    def _calculate_overlap(bbox1, bbox2):
        """Calculate overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)

        if left >= right or top >= bottom:
            return 0.0

        intersection = (right - left) * (bottom - top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0


class OutputGenerator:

    @staticmethod
    def execute(output_dir: str, text: str, metadata: Dict[str, Any]) -> ToolResult:
        try:
            os.makedirs(output_dir, exist_ok=True)

            text_path = os.path.join(output_dir, "text.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)

            # Generate and save summary
            summary = OutputGenerator._generate_summary(text, metadata)
            summary_path = os.path.join(output_dir, "summary.md")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)

            return ToolResult(True, {
                "text_file": text_path,
                "summary_file": summary_path,
                "files_created": ["text.txt", "summary.md"]
            })
        except Exception as e:
            return ToolResult(False, {}, f"Output generation failed: {str(e)}")

    @staticmethod
    def _generate_summary(text: str, metadata: Dict[str, Any]) -> str:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        word_count = len(text.split())

        summary = f"""# Document Analysis Summary

        **Status**: {'Document Detected' if metadata.get('is_document') else 'Non-Document'}
        **Confidence**: {metadata.get('confidence', 0):.1%}
        **Word Count**: {word_count}
        **Lines**: {len(lines)}

        ## Content Preview
        """

        if lines:
            preview_lines = lines[:5]
            for line in preview_lines:
                display_line = line[:100] + "..." if len(line) > 100 else line
                summary += f"- {display_line}\n"
        else:
            summary += "No readable content found.\n"

        return summary


TOOLS = {
    "load_image": ImageLoader,
    "preprocess": ImagePreprocessor,
    "extract_text": OCRExtractor,
    "score_document": DocumentScorer,
    "analyze_layout": DoclingLayoutAnalyzer,
    "detect_regions": RegionDetector,
    "detect_formulas": FormulaDetector,
    "recognize_formulas": FormulaRecognizer,
    "visualize_regions": RegionVisualizer,
    "extract_chart_data": ChartDataExtractor,
    "generate_output": OutputGenerator
}


def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    if tool_name not in TOOLS:
        return ToolResult(False, {}, f"Unknown tool: {tool_name}").to_dict()

    tool_class = TOOLS[tool_name]
    result = tool_class.execute(**kwargs)
    return result.to_dict()
