"""RefCOCO grounding: predict one box per referring expression; Acc@IoU=0.5.

Reports a strict single-interpretation score plus a format-tolerant `oracle`
sub-score that tries {pixel, pixel2x, norm1000, norm1} x {xyxy, yxyx} against GT
(pixel2x covers models like Inkling that report in a 2x-upscaled image space).
"""

from __future__ import annotations

import re
from collections import defaultdict

from bench_core.media import encode_image
from bench_core.request import Message, ReasoningSpec, Request, TextPart

NAME = "refcoco"
ID_KEY = "id"
PRIMARY_METRIC = "accuracy"
# variant = split for base RefCOCO, or "plus-<split>" / "g-<split>" for
# RefCOCO+ / RefCOCOg (same underlying task, lmms-lab packaging).
VARIANTS = [
    "val",
    "testA",
    "testB",
    "test",
    "plus-val",
    "plus-testA",
    "plus-testB",
    "g-val",
    "g-test",
]
DEFAULTS = {"reasoning": "off", "rate_limit": 25, "max_in_flight": 8}

_DATASETS = {
    "": "lmms-lab/RefCOCO",
    "plus": "lmms-lab/RefCOCO+",
    "g": "lmms-lab/RefCOCOg",
}
_MAX_SIDE = 1024
_DATASET = None  # full-run split; images read lazily by idx to bound memory
IOU_THRESHOLD = 0.5
_THRESHOLDS = [0.3, 0.5, 0.7, 0.75, 0.9]

_PROMPT = (
    "Please provide the bounding box coordinate of the region this sentence describes: "
    "{expression}\n\n"
    "Output the coordinates in the format [x_min, y_min, x_max, y_max]."
)

# --- production single-interpretation parser (from refcoco.py) ---
BOX_LINE_PATTERN = re.compile(
    r"(?im)^\s*(?:box|answer|bounding\s*box)[\s:]*"
    r"\[?\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*"
    r"(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)\s*\]?"
)
BOXED_PATTERN = re.compile(
    r"\\boxed\{\s*\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?\s*\}"
)
BARE_4TUPLE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
JSON_TLBR_PATTERN = re.compile(
    r'"top_left"\s*:\s*\{[^}]*?"x"\s*:\s*(-?\d+(?:\.\d+)?)[^}]*?"y"\s*:\s*(-?\d+(?:\.\d+)?)'
    r'[^}]*?\}[^}]*?"bottom_right"\s*:\s*\{[^}]*?"x"\s*:\s*(-?\d+(?:\.\d+)?)'
    r'[^}]*?"y"\s*:\s*(-?\d+(?:\.\d+)?)',
    re.DOTALL,
)
JSON_BOX2D_PATTERN = re.compile(
    r'"box_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*'
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def _normalize_to_pixels(box, width, height, order):
    max_coord = max(abs(c) for c in box)
    if max_coord <= 1.0:
        sx, sy = float(width), float(height)
    elif max_coord <= 1000 and (
        max(width, height) > 1000
        or box[0] > width
        or box[2] > width
        or box[1] > height
        or box[3] > height
    ):
        sx, sy = width / 1000.0, height / 1000.0
    else:
        sx = sy = 1.0
    if order == "xyxy":
        return [box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy]
    return [box[1] * sx, box[0] * sy, box[3] * sx, box[2] * sy]


def parse_box(text: str, width: int, height: int):
    if not text:
        return None
    m = None
    for match in JSON_TLBR_PATTERN.finditer(text):
        m = match
    if m is not None:
        return _normalize_to_pixels(
            [float(m.group(i)) for i in range(1, 5)], width, height, "xyxy"
        )
    m = None
    for match in JSON_BOX2D_PATTERN.finditer(text):
        m = match
    if m is not None:
        return _normalize_to_pixels(
            [float(m.group(i)) for i in range(1, 5)], width, height, "yxyx"
        )
    for pattern in (BOX_LINE_PATTERN, BOXED_PATTERN):
        matches = list(pattern.finditer(text))
        if matches:
            m = matches[-1]
            return _normalize_to_pixels(
                [float(m.group(i)) for i in range(1, 5)], width, height, "xyxy"
            )
    matches = list(BARE_4TUPLE.finditer(text[-800:]))
    if matches:
        m = matches[-1]
        return _normalize_to_pixels(
            [float(m.group(i)) for i in range(1, 5)], width, height, "xyxy"
        )
    return None


def compute_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ax1, ax2 = min(ax1, ax2), max(ax1, ax2)
    ay1, ay2 = min(ay1, ay2), max(ay1, ay2)
    bx1, bx2 = min(bx1, bx2), max(bx1, bx2)
    by1, by2 = min(by1, by2), max(by1, by2)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def coco_bbox_to_xyxy(bbox, sx=1.0, sy=1.0):
    x, y, w, h = bbox
    return [x * sx, y * sy, (x + w) * sx, (y + h) * sy]


# --- format-tolerant oracle (from reeval_format_tolerant.py) ---
_TUPLE = BARE_4TUPLE
_PAREN_PAIRS = re.compile(
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*[^()]*?\s*"
    r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
)


def _extract_tuples(text: str):
    out = []
    for m in _TUPLE.finditer(text):
        out.append(tuple(float(g) for g in m.groups()))
    for m in JSON_TLBR_PATTERN.finditer(text):
        out.append(tuple(float(g) for g in m.groups()))
    for m in _PAREN_PAIRS.finditer(text):
        out.append(tuple(float(g) for g in m.groups()))
    return out


def _all_interpretations(nums, w, h):
    n0, n1, n2, n3 = nums
    mx = max(abs(c) for c in nums)
    yield "pixel-xyxy", [n0, n1, n2, n3]
    yield "pixel-yxyx", [n1, n0, n3, n2]
    if mx > min(w, h):
        yield "pixel2x-xyxy", [n0 / 2, n1 / 2, n2 / 2, n3 / 2]
        yield "pixel2x-yxyx", [n1 / 2, n0 / 2, n3 / 2, n2 / 2]
    if mx <= 1000:
        yield (
            "norm1000-xyxy",
            [n0 * w / 1000, n1 * h / 1000, n2 * w / 1000, n3 * h / 1000],
        )
        yield (
            "norm1000-yxyx",
            [n1 * w / 1000, n0 * h / 1000, n3 * w / 1000, n2 * h / 1000],
        )
    if mx <= 1.0:
        yield "norm1-xyxy", [n0 * w, n1 * h, n2 * w, n3 * h]
        yield "norm1-yxyx", [n1 * w, n0 * h, n3 * w, n2 * h]


def _best_box(response: str, w: int, h: int, gt):
    best_iou, best_label = 0.0, None
    for nums in _extract_tuples(response):
        for label, box in _all_interpretations(nums, w, h):
            v = compute_iou(box, gt)
            if v > best_iou:
                best_iou, best_label = v, label
    return best_iou, best_label


# --- benchmark interface ---
def _sent_dims(w, h, max_side=_MAX_SIDE):
    longest = max(w, h)
    if longest <= max_side:
        return w, h
    scale = max_side / longest
    return round(w * scale), round(h * scale)


def _parse_variant(variant: str) -> tuple[str, str]:
    if variant.startswith("plus-"):
        return _DATASETS["plus"], variant[len("plus-") :]
    if variant.startswith("g-"):
        return _DATASETS["g"], variant[len("g-") :]
    return _DATASETS[""], variant


def _mk_sample(row, i, image, idx=None) -> dict | None:
    answers = row.get("answer")
    if isinstance(answers, str):
        exprs = [answers]
    elif isinstance(answers, list):
        exprs = [str(a) for a in answers if str(a).strip()]
    else:
        exprs = []
    if not exprs:
        return None
    ow, oh = image.size  # header-only; not retained on the full path
    sw, sh = _sent_dims(ow, oh)
    gt = coco_bbox_to_xyxy(list(row["bbox"]), sw / ow, sh / oh)
    s = {
        "id": f"{row.get('question_id', i)}_{i}",
        "expression": exprs[0],
        "sent_w": sw,
        "sent_h": sh,
        "gt_bbox_xyxy": gt,
    }
    if idx is None:
        s["image"] = image  # smoke embeds the streamed image
    else:
        s["idx"] = idx  # full run reads it lazily from _DATASET
    return s


def load_samples(sample_size: int | None = None, variant: str = "val") -> list[dict]:
    global _DATASET
    dataset_id, split = _parse_variant(variant)

    if sample_size:
        from bench_core.datautil import load_rows

        rows = load_rows(dataset_id, split, sample_size)
        out = (_mk_sample(r, i, r["image"]) for i, r in enumerate(rows))
        return [s for s in out if s]

    # full run: keep the split memory-mapped and read images lazily by idx
    # (list(ds) materializes all ~8.8k decoded images and OOMs CI).
    from datasets import load_dataset

    _DATASET = load_dataset(dataset_id, split=split)
    samples = []
    for i in range(len(_DATASET)):
        row = _DATASET[i]
        s = _mk_sample(row, i, row["image"], idx=i)
        if s:
            samples.append(s)
    return samples


def build_request(sample: dict, mode: str) -> Request:
    prompt = _PROMPT.format(expression=sample["expression"])
    if "image" in sample:
        image = sample["image"]  # streamed smoke: embedded
    elif "idx" in sample:
        image = _DATASET[sample["idx"]]["image"]  # full run: decoded lazily
    else:
        raise KeyError("RefCOCO sample missing both 'image' and 'idx'")
    img = encode_image(image, "image/jpeg", max_side=_MAX_SIDE)
    return Request(
        [Message("user", [TextPart(prompt), img])], ReasoningSpec(mode), temperature=0.0
    )


def parse(response, sample):
    return parse_box(response.text or "", sample["sent_w"], sample["sent_h"])


def _sweep(ious):
    n = len(ious)
    return {
        f"acc@{t}": (sum(1 for i in ious if i >= t) / n if n else 0)
        for t in _THRESHOLDS
    }


def score(records: list[dict], samples: list[dict]) -> dict:
    by_id = {s["id"]: s for s in samples}
    latest = {r["id"]: r for r in records if r["id"] in by_id}

    strict_ious, oracle_ious = [], []
    strict_correct = oracle_correct = unparsed = total = 0
    labels: dict[str, int] = defaultdict(int)
    for rid, r in latest.items():
        s = by_id[rid]
        gt, w, h = s["gt_bbox_xyxy"], s["sent_w"], s["sent_h"]
        total += 1
        pred = r.get("prediction")
        iou = compute_iou(pred, gt) if pred else 0.0
        strict_ious.append(iou)
        if pred is not None and iou >= IOU_THRESHOLD:
            strict_correct += 1
        if pred is None:
            unparsed += 1
        oiou, olabel = _best_box(r.get("response") or "", w, h, gt)
        oracle_ious.append(oiou)
        if oiou >= IOU_THRESHOLD:
            oracle_correct += 1
        if olabel:
            labels[olabel] += 1

    return {
        "accuracy": strict_correct / total if total else 0.0,
        "correct": strict_correct,
        "total": total,
        "unparsed": unparsed,
        "mean_iou": sum(strict_ious) / len(strict_ious) if strict_ious else 0.0,
        "iou_thresholds": _sweep(strict_ious),
        "oracle": {
            "accuracy": oracle_correct / total if total else 0.0,
            "correct": oracle_correct,
            "total": total,
            "mean_iou": sum(oracle_ious) / len(oracle_ious) if oracle_ious else 0.0,
            "interpretation_counts": dict(labels),
        },
    }
