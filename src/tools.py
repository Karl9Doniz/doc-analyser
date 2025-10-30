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
    "detect_regions": RegionDetector,
    "generate_output": OutputGenerator
}


def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    if tool_name not in TOOLS:
        return ToolResult(False, {}, f"Unknown tool: {tool_name}").to_dict()

    tool_class = TOOLS[tool_name]
    result = tool_class.execute(**kwargs)
    return result.to_dict()