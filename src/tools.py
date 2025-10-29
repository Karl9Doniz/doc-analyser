import os
from typing import Dict, Any
from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import cv2
import numpy as np


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


class FigureDetector:
    """Tool: Detect and extract figures/tables from documents"""

    @staticmethod
    def execute(image_path: str, min_area: int = 5000, extract_tables: bool = True) -> ToolResult:
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            figures = []
            table_data = []

            if extract_tables:
                # Enhanced table detection
                tables = FigureDetector._detect_tables(image, gray)
                figures.extend(tables["regions"])
                table_data = tables["extracted_data"]

            # General figure detection
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Skip if already detected as table
                    is_duplicate = any(
                        abs(x - fig["bbox"][0]) < 50 and abs(y - fig["bbox"][1]) < 50
                        for fig in figures
                    )
                    if not is_duplicate:
                        figures.append({
                            "id": len(figures),
                            "type": "figure",
                            "bbox": [x, y, w, h],
                            "area": area,
                            "aspect_ratio": h / w
                        })

            return ToolResult(True, {
                "figure_count": len(figures),
                "figures": figures,
                "table_data": table_data,
                "total_figure_area": sum(f["area"] for f in figures)
            })
        except Exception as e:
            return ToolResult(False, {}, f"Figure detection failed: {str(e)}")

    @staticmethod
    def _detect_tables(image, gray):
        """Enhanced table detection using line detection"""
        try:
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

            # Detect lines
            horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)

            # Combine lines to find table structure
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)

            # Find table regions
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            tables = []
            extracted_data = []

            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area > 10000:  # Minimum table size
                    x, y, w, h = cv2.boundingRect(contour)

                    # Extract table region for OCR
                    table_region = image[y:y+h, x:x+w]
                    table_text = FigureDetector._extract_table_text(table_region)

                    tables.append({
                        "id": i,
                        "type": "table",
                        "bbox": [x, y, w, h],
                        "area": area,
                        "aspect_ratio": h / w
                    })

                    extracted_data.append({
                        "table_id": i,
                        "bbox": [x, y, w, h],
                        "raw_text": table_text,
                        "structured_data": FigureDetector._parse_table_structure(table_text)
                    })

            return {"regions": tables, "extracted_data": extracted_data}
        except Exception:
            return {"regions": [], "extracted_data": []}

    @staticmethod
    def _extract_table_text(table_region):
        """Extract text from table region with table-specific OCR settings"""
        try:
            # Convert to PIL Image for pytesseract
            table_pil = cv2.cvtColor(table_region, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil_image = Image.fromarray(table_pil)

            # Use PSM 6 for uniform block of text (good for tables)
            config = "--psm 6 -c preserve_interword_spaces=1"
            text = pytesseract.image_to_string(pil_image, config=config)
            return text.strip()
        except Exception:
            return ""

    @staticmethod
    def _parse_table_structure(text):
        """Attempt to parse table structure from OCR text"""
        try:
            lines = [line.strip() for line in text.split('\n') if line.strip()]

            # Lines with multiple spaces likely represent table rows
            table_rows = []
            for line in lines:
                if '  ' in line:
                    columns = [col.strip() for col in line.split('  ') if col.strip()]
                    if len(columns) > 1:
                        table_rows.append(columns)

            return {
                "rows": table_rows,
                "row_count": len(table_rows),
                "estimated_columns": max(len(row) for row in table_rows) if table_rows else 0
            }
        except Exception:
            return {"rows": [], "row_count": 0, "estimated_columns": 0}


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
    "detect_figures": FigureDetector,
    "generate_output": OutputGenerator
}


def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    if tool_name not in TOOLS:
        return ToolResult(False, {}, f"Unknown tool: {tool_name}").to_dict()

    tool_class = TOOLS[tool_name]
    result = tool_class.execute(**kwargs)
    return result.to_dict()