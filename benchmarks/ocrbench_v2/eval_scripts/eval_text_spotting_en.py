"""
Evaluation script for OCRBench v2 — Text Spotting EN only.
Scores predictions exclusively for the "text spotting en" task type.
"""

import re
import ast
import json
import argparse
from tqdm import tqdm
from spotting_metric import spotting_evaluation


def strip_markdown_fences(text):
    """Remove markdown code fences like ```json ... ``` or ```python ... ```."""
    return re.sub(r"```\w*\n?", "", text).strip()


def _parse_box(fields):
    """[x1, y1, x2, y2, text] from five raw fields, or None if malformed, out of 0-1000, or empty."""
    try:
        x1, y1, x2, y2 = (int(str(f).strip()) for f in fields[:4])
    except (ValueError, TypeError):
        return None
    text = str(fields[4]).replace("\n", "").strip().strip('"').strip("'")
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if not all(0 <= v <= 1000 for v in (x1, y1, x2, y2)) or x1 == x2 or y1 == y2:
        return None
    return [x1, y1, x2, y2, text]


def extract_bounding_boxes(predict_str):
    """Parse [x1, y1, x2, y2, text] boxes (fences stripped, inverted coords fixed), or None."""
    predict_str = strip_markdown_fences(predict_str)
    try:
        data = ast.literal_eval(predict_str)
    except Exception:
        data = None

    if data is not None:
        if not isinstance(data, (list, tuple)):
            return None
        candidates = [
            item for item in data if isinstance(item, (list, tuple)) and len(item) >= 5
        ]
    else:
        # Fallback: regex parsing
        items = re.findall(r"[\[\(]\s*([^\[\]\(\)]*?)\s*[\]\)]", predict_str)
        if not items:
            return None
        candidates = [
            parts for parts in (i.split(",", 4) for i in items) if len(parts) == 5
        ]

    results, seen = [], set()
    for fields in candidates:
        box = _parse_box(fields)
        if box is None or tuple(box) in seen:
            continue
        seen.add(tuple(box))
        results.append(box)
    return results or None


def _polygon_to_box(coords):
    """Axis-aligned bbox of an x,y polygon, as 8 points (same reduction as spotting_metric.py)."""
    xs, ys = coords[0::2], coords[1::2]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def spotting_evaluation_normalized(prediction_list, img_metas):
    """spotting_evaluation, but inverted coords are normalized instead of discarded."""
    normalized = []
    for item in prediction_list:
        if len(item) != 5:
            continue
        x1, y1, x2, y2, rec = item
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x1 == x2 or y1 == y2:
            continue
        normalized.append([x1, y1, x2, y2, rec])

    if not normalized:
        return 0

    return spotting_evaluation(normalized, img_metas)


def process_predictions(input_path, output_path):
    with open(input_path, "r") as f:
        predict_file = json.load(f)

    res_data_list = []

    for index, data_item in enumerate(tqdm(predict_file, desc="Scoring text spotting")):
        if data_item["type"] != "text spotting en":
            # Skip non-text-spotting items (shouldn't be any, but just in case)
            res_data_list.append(data_item)
            continue

        # Parse bbox/content from answers if not present (HF dataset format)
        # GT format: x1,y1,x2,y2,...,xN,yN,text (variable number of coordinate pairs)
        if "bbox" not in data_item and "answers" in data_item and data_item["answers"]:
            bboxes, contents = [], []
            for line in data_item["answers"][0].strip().split("\n"):
                parts = line.split(",")
                # Find where coordinates end: scan from start while parts are integers
                num_coords = 0
                for p in parts:
                    try:
                        int(p.strip())
                        num_coords += 1
                    except ValueError:
                        break
                # Need at least 8 coords (4 points) and some text
                if num_coords >= 8 and num_coords < len(parts):
                    coords = [int(p.strip()) for p in parts[:num_coords]]
                    bboxes.append(_polygon_to_box(coords))
                    contents.append(",".join(parts[num_coords:]))
                elif num_coords >= 8 and num_coords == len(parts):
                    # Numeric text (e.g. "1700"): coords come in pairs, so an odd count
                    # means the last value is the text
                    if num_coords % 2 == 1:
                        coords = [int(p.strip()) for p in parts[: num_coords - 1]]
                        bboxes.append(_polygon_to_box(coords))
                        contents.append(parts[num_coords - 1].strip())
            data_item["bbox"] = bboxes
            data_item["content"] = contents

        if "bbox" not in data_item or "content" not in data_item:
            data_item["score"] = 0
        elif not isinstance(data_item["predict"], str):
            data_item["score"] = 0
        else:
            predict_bbox = extract_bounding_boxes(data_item["predict"])
            if not predict_bbox:
                data_item["score"] = 0
            else:
                data_item["score"] = spotting_evaluation_normalized(
                    predict_bbox, data_item
                )

        res_data_list.append(data_item)

    # Print summary
    mean_score, total_len = 0, 0
    for item in res_data_list:
        if item["type"] == "text spotting en" and "ignore" not in item:
            total_len += 1
            mean_score += item["score"]

    mean_score = mean_score / total_len if total_len > 0 else 0
    print(
        f"\nTask text spotting en, total instructions: {total_len}, average score: {mean_score:.3f}\n"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(predict_file, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate text spotting EN predictions from OCRBench v2."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input prediction JSON file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the results JSON file.",
    )

    args = parser.parse_args()
    process_predictions(args.input_path, args.output_path)
