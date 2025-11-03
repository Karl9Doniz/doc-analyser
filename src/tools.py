import json
import logging
import os
import threading
import time
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from chart_extractor import MiniCPMVEngine, run_minicpm_vqa
except ImportError:  # pragma: no cover - optional dependency
    MiniCPMVEngine = None  # type: ignore
    run_minicpm_vqa = None  # type: ignore


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
    """Tool: Calculate document likelihood scores"""

    @staticmethod
    def execute(image_path: str, text_data: Dict[str, Any]) -> ToolResult:
        try:
            image = Image.open(image_path)
            w, h = image.size

            # Text density score
            char_count = text_data.get("char_count", 0)
            pixel_area = w * h
            char_density = min(1.0, (char_count / (pixel_area / 1000)))  # chars per 1000 pixels

            # Content quality score
            word_count = text_data.get("word_count", 0)
            has_content = text_data.get("has_meaningful_content", False)
            content_score = 1.0 if has_content else 0.0

            # Aspect ratio score (document-like proportions)
            aspect_ratio = h / w
            aspect_score = 1.0 if 0.5 <= aspect_ratio <= 2.0 else 0.5

            # Combined score
            final_score = (char_density * 0.5 + content_score * 0.3 + aspect_score * 0.2)

            return ToolResult(True, {
                "char_density": char_density,
                "content_score": content_score,
                "aspect_score": aspect_score,
                "final_score": min(1.0, final_score),
                "metrics": {
                    "chars_per_1k_pixels": (char_count / (pixel_area / 1000)),
                    "aspect_ratio": aspect_ratio,
                    "word_count": word_count
                }
            })
        except Exception as e:
            return ToolResult(False, {}, f"Scoring failed: {str(e)}")


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


class ChartDataExtractor:
    """Tool: Extract chart data from cropped regions using MiniCPM-V via llama-cpp."""

    _default_prompt = (
        "You are an analytical assistant that reads scientific charts. "
        "Look carefully at the image and reply with pure JSON (no Markdown or commentary)."
        "\nRequired format when you CAN read the chart:\n"
        "{"
        "\"status\": \"success\", "
        "\"title\": <string>, "
        "\"axes\": [ {\"label\": <string>, \"unit\": <string or null>, \"min\": <number or null>, \"max\": <number or null>} , ... ], "
        "\"series\": [ {\"name\": <string>, \"points\": [ {\"x\": <number>, \"y\": <number>, \"label\": <string or null>} , ... ] } , ... ], "
        "\"legend\": [<strings>], "
        "\"annotations\": [<strings>], "
        "\"summary\": <string summarising the numeric trend>"
        "}"
        "\nEvery numeric value MUST come from the chart exactly; use floats/ints, not strings."
        "\nIf you cannot confidently read the numbers (blurred, cropped, ambiguous), respond with:\n"
        "{\"status\": \"unreadable\", \"reason\": <concise explanation>}"
        "\nNever guess or invent values."
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
            question = ChartDataExtractor._default_prompt

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

        try:
            vqa_result = run_minicpm_vqa(
                chart_crop,
                question,
                model_path=model_path,
                mmproj_path=mmproj_path,
                max_tokens=max_tokens,
                temperature=temperature,
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

        if parsed_json is None:
            return ToolResult(
                False,
                {
                    "region_id": region_id,
                    "question": question,
                    "raw_answer": answer_text,
                    "raw_message": raw_message,
                    "model_path": vqa_result.get("model_path"),
                    "mmproj_path": vqa_result.get("mmproj_path"),
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
        }
        return ToolResult(True, data)

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
        for entry in series:
            points = entry.get("points")
            if not isinstance(points, list) or len(points) < 2:
                continue
            y_values = [p.get("y") for p in points if isinstance(p, dict) and isinstance(p.get("y"), (int, float))]
            if len(set(y_values)) > 1:
                return True
        return False


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
    "extract_chart_data": ChartDataExtractor,
    "generate_output": OutputGenerator
}


def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    if tool_name not in TOOLS:
        return ToolResult(False, {}, f"Unknown tool: {tool_name}").to_dict()

    tool_class = TOOLS[tool_name]
    result = tool_class.execute(**kwargs)
    return result.to_dict()
