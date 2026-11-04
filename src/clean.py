"""Cleaning rules: raw COCO val2017 -> manifests/.

Run from the project root:
    python -m src.clean

Reads   val2017/ and annotations/captions_val2017.json
Writes  manifests/manifest.json, manifests/drop_log.csv, manifests/cleaning_summary.json
Never deletes, moves, or modifies image files.

Rules (rationale and visual checks: notebooks/02_cleaning.ipynb):
  1. Drop grayscale images: single-channel files, and RGB files whose 99th-percentile
     per-pixel chroma (max(R,G,B) - min(R,G,B)) is below a threshold.
  2. Drop images whose long/short side ratio exceeds a cutoff (center crop would lose too much).
  3. Keep images with zero COCO object annotations (captions are still valid).
  4. Remove duplicate captions within an image (case/punctuation-insensitive); keep all unique ones.
"""
import csv
import json
import re
from collections import Counter

import numpy as np
from PIL import Image

from src.data import ANN_DIR, IMG_DIR, MANIFEST_DIR

CONFIG = {
    "aspect_ratio_max": 2.0,     # drop if long/short side > this
    "gray_chroma_p99_max": 15,   # drop RGB image if 99th-pct per-pixel chroma < this
    "gray_check_size": 128,      # downsample to this square (bicubic) before measuring chroma
}


# ---------- loading ----------
def load_coco_captions(path=ANN_DIR / "captions_val2017.json"):
    """Return (images: {id: record}, raw_captions: {id: [caption, ...]})."""
    with open(path) as f:
        cap = json.load(f)
    images = {im["id"]: im for im in cap["images"]}
    raw = {}
    for a in cap["annotations"]:
        raw.setdefault(a["image_id"], []).append(a["caption"])
    return images, raw


# ---------- rule 1: grayscale ----------
def chroma_p99(path, size=CONFIG["gray_check_size"]):
    """Return (p99 chroma or None if not RGB, PIL mode)."""
    with Image.open(path) as im:
        if im.mode != "RGB":
            return None, im.mode
        a = np.asarray(im.resize((size, size), Image.BICUBIC), dtype=np.int16)
    return float(np.percentile(a.max(-1) - a.min(-1), 99)), "RGB"


def find_grayscale(images, img_dir=IMG_DIR, threshold=CONFIG["gray_chroma_p99_max"]):
    """Return (drops: {id: [(reason, value)]}, chroma: {id: p99 or None})."""
    drops, chroma = {}, {}
    for iid, im in images.items():
        c, mode = chroma_p99(img_dir / im["file_name"])
        chroma[iid] = c
        if c is None:
            drops[iid] = [("grayscale_single_channel", mode)]
        elif c < threshold:
            drops[iid] = [("grayscale_rgb_encoded", round(c, 1))]
    return drops, chroma


# ---------- rule 2: aspect ratio ----------
def aspect_ratios(images):
    return {i: max(im["width"], im["height"]) / min(im["width"], im["height"]) for i, im in images.items()}


def find_extreme_aspect(images, cutoff=CONFIG["aspect_ratio_max"]):
    """Return (drops: {id: [(reason, value)]}, ratio: {id: long/short})."""
    ratio = aspect_ratios(images)
    return {i: [("aspect_ratio", round(r, 2))] for i, r in ratio.items() if r > cutoff}, ratio


# ---------- rule 4: caption dedupe ----------
def norm_caption(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def dedupe_captions(caps):
    """Keep the first occurrence (original text, stripped) of each normalized caption."""
    seen, out = set(), []
    for c in caps:
        k = norm_caption(c)
        if k and k not in seen:
            seen.add(k)
            out.append(c.strip())
    return out


# ---------- pipeline ----------
def merge_drops(*drop_dicts):
    merged = {}
    for d in drop_dicts:
        for i, reasons in d.items():
            merged.setdefault(i, []).extend(reasons)
    return merged


def build(images, raw_captions, drops):
    """Apply drops + caption dedupe. Return (manifest, summary)."""
    n_raw_caps = sum(len(v) for v in raw_captions.values())
    kept = sorted(i for i in images if i not in drops)
    captions = {i: dedupe_captions(raw_captions[i]) for i in kept}
    n_caps = sum(len(v) for v in captions.values())
    n_gray = sum(1 for v in drops.values() if any(r.startswith("grayscale") for r, _ in v))

    manifest = {
        "source": "COCO val2017 (captions_val2017.json)",
        "config": CONFIG,
        "n_images": len(kept),
        "n_captions": n_caps,
        "images": [
            {"image_id": i, "file_name": images[i]["file_name"],
             "width": images[i]["width"], "height": images[i]["height"],
             "captions": captions[i]}
            for i in kept
        ],
    }
    summary = {
        "config": CONFIG,
        "stages": [
            {"stage": "start", "images": len(images), "captions": n_raw_caps},
            {"stage": "after grayscale", "images": len(images) - n_gray},
            {"stage": "after aspect ratio", "images": len(kept)},
            {"stage": "after caption dedupe", "images": len(kept), "captions": n_caps},
        ],
        "dropped_by_reason": dict(Counter(r for v in drops.values() for r, _ in v)),
        "dropped_images_total": len(drops),
        "duplicate_captions_removed": sum(len(raw_captions[i]) for i in kept) - n_caps,
        "unique_captions_per_image": {str(k): v for k, v in
                                      sorted(Counter(len(c) for c in captions.values()).items())},
    }
    return manifest, summary


def write_outputs(manifest, summary, drops, images, out_dir=MANIFEST_DIR):
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    with open(out_dir / "cleaning_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    with open(out_dir / "drop_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "file_name", "reasons", "values"])
        for i in sorted(drops):
            w.writerow([i, images[i]["file_name"],
                        ";".join(r for r, _ in drops[i]), ";".join(str(v) for _, v in drops[i])])


def run():
    images, raw = load_coco_captions()
    gray, _ = find_grayscale(images)
    wide, _ = find_extreme_aspect(images)
    drops = merge_drops(gray, wide)
    manifest, summary = build(images, raw, drops)
    write_outputs(manifest, summary, drops, images)
    return manifest, summary


if __name__ == "__main__":
    _, s = run()
    for st in s["stages"]:
        print(f"{st['stage']:<22} {st['images']:>5,} images" +
              (f"   {st['captions']:>6,} captions" if "captions" in st else ""))
    print("dropped by reason:", s["dropped_by_reason"])
    print(f"wrote {MANIFEST_DIR}/manifest.json, drop_log.csv, cleaning_summary.json")
