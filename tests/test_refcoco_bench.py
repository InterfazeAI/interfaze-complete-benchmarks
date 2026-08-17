"""RefCOCO: parse/IoU/oracle units + offline scoring-parity replay.

Strict replay re-parses each archived response and matches Acc@0.5; the oracle
replay re-runs best-of-interpretation and matches the archived _oracle_ metrics.
"""

import json
import pathlib

import pytest

from benchmarks.obj_detection import bench

RESULTS = pathlib.Path(__file__).resolve().parent.parent / "results"
STRICT_TAGS = [
    "refcoco_val_fireworks_accounts_fireworks_models_inkling",
    "refcoco_val_openrouter_thinkingmachines_inkling-small",
    "refcoco_testA_gemini_gemini-3.7-flash",
]
ORACLE_BASE = "refcoco_val_fireworks_accounts_fireworks_models_inkling"


def test_parse_variant_restores_plus_and_g_datasets():
    assert bench._parse_variant("val") == ("lmms-lab/RefCOCO", "val")
    assert bench._parse_variant("testA") == ("lmms-lab/RefCOCO", "testA")
    assert bench._parse_variant("plus-testB") == ("lmms-lab/RefCOCO+", "testB")
    assert bench._parse_variant("g-test") == ("lmms-lab/RefCOCOg", "test")


def test_build_request_embedded_image_or_lazy_idx(monkeypatch):
    """Smokes embed the image; full runs carry an index and read the image
    lazily from _DATASET (so the samples list doesn't hold all 8.8k images)."""
    from PIL import Image

    img = Image.new("RGB", (16, 16))
    base = {
        "id": "x",
        "expression": "cat",
        "sent_w": 16,
        "sent_h": 16,
        "gt_bbox_xyxy": [0, 0, 1, 1],
    }

    req = bench.build_request({**base, "image": img}, "off")
    assert len(req.messages[0].parts) == 2  # text + image

    monkeypatch.setattr(bench, "_DATASET", {5: {"image": img}})
    req2 = bench.build_request({**base, "idx": 5}, "off")
    assert len(req2.messages[0].parts) == 2


def test_compute_iou():
    assert bench.compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert bench.compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_parse_box_pixel_and_box2d():
    assert bench.parse_box("[10, 20, 30, 40]", 100, 100) == [10, 20, 30, 40]
    # box_2d is yxyx -> swapped to xyxy
    assert bench.parse_box('{"box_2d": [20, 10, 40, 30]}', 1000, 1000) == pytest.approx(
        [10, 20, 30, 40]
    )


def test_best_box_recovers_pixel2x():
    # a box reported in 2x-upscaled space; oracle should halve it to match GT
    iou, label = bench._best_box("[100, 100, 200, 200]", 100, 100, [50, 50, 100, 100])
    assert label == "pixel2x-xyxy"
    assert iou == 1.0


def _replay(tag):
    resp_path = RESULTS / f"{tag}_responses.jsonl"
    samples, records = [], []
    for line in resp_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        w, h = rec["image_width"], rec["image_height"]
        samples.append(
            {
                "id": rec["id"],
                "gt_bbox_xyxy": rec["gt_bbox_xyxy"],
                "sent_w": w,
                "sent_h": h,
            }
        )
        records.append(
            {
                "id": rec["id"],
                "response": rec["response"],
                "prediction": bench.parse_box(rec["response"], w, h),
            }
        )
    return bench.score(records, samples)


@pytest.mark.parametrize("tag", STRICT_TAGS)
def test_strict_scoring_parity(tag):
    resp_path = RESULTS / f"{tag}_responses.jsonl"
    metrics_path = RESULTS / f"{tag}_metrics.json"
    if not (resp_path.exists() and metrics_path.exists()):
        pytest.skip(f"archived run {tag} not present")
    archived = json.loads(metrics_path.read_text())
    m = _replay(tag)
    assert m["total"] == archived["total"]
    assert m["accuracy"] == pytest.approx(archived["accuracy"])
    assert m["unparsed"] == archived["unparsed"]


def test_oracle_scoring_parity():
    resp_path = RESULTS / f"{ORACLE_BASE}_responses.jsonl"
    oracle_metrics = RESULTS / f"{ORACLE_BASE}_oracle_metrics.json"
    if not (resp_path.exists() and oracle_metrics.exists()):
        pytest.skip("archived oracle run not present")
    archived = json.loads(oracle_metrics.read_text())
    m = _replay(ORACLE_BASE)
    assert m["oracle"]["accuracy"] == pytest.approx(archived["accuracy"])
