"""Train / held-out split for LoRA fine-tuning (experiment design, not cleaning).

Split by IMAGE, never by caption: every caption of an image lands on the same side, so no
training caption describes a held-out photo.

- train:    LoRA fine-tuning uses these images and ALL of their captions.
- held_out: everything evaluated (baseline, guidance/sampler sweep, before/after LoRA,
            FID/KID, CLIP score) uses only these images and their prompts.json caption.

1,000 held-out images is enough for KID (FID is noisier at this size).
Writes manifests/split.json. Run: python -m src.split
"""
import json
import random

from src.data import MANIFEST_DIR, load_manifest

SPLIT_SEED = 7615
N_HELD_OUT = 1000
SPLIT_PATH = MANIFEST_DIR / "split.json"


def make_split(manifest, n_held_out=N_HELD_OUT, seed=SPLIT_SEED):
    """Return {"train": [image_id, ...], "held_out": [image_id, ...]}, both sorted."""
    ids = sorted(e["image_id"] for e in manifest["images"])
    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    held_out = sorted(shuffled[:n_held_out])
    train = sorted(shuffled[n_held_out:])
    return {"train": train, "held_out": held_out}


def check_split(split, manifest):
    """Fail loudly if the split overlaps or loses/invents images."""
    train, held = set(split["train"]), set(split["held_out"])
    all_ids = {e["image_id"] for e in manifest["images"]}
    assert not train & held, "an image is on both sides"
    assert train | held == all_ids, "split does not cover the manifest exactly"


def write_split(split, seed=SPLIT_SEED, path=SPLIT_PATH):
    with open(path, "w") as f:
        json.dump({"seed": seed,
                   "n_train": len(split["train"]),
                   "n_held_out": len(split["held_out"]),
                   **split}, f, indent=1)


def load_split(path=SPLIT_PATH):
    """Return {"train": [...], "held_out": [...]}."""
    with open(path) as f:
        d = json.load(f)
    return {"train": d["train"], "held_out": d["held_out"]}


def main():
    manifest = load_manifest()
    split = make_split(manifest)
    check_split(split, manifest)
    write_split(split)
    by_id = {e["image_id"]: e for e in manifest["images"]}
    n_train_caps = sum(len(by_id[i]["captions"]) for i in split["train"])
    print(f"train:    {len(split['train']):,} images, {n_train_caps:,} captions")
    print(f"held_out: {len(split['held_out']):,} images")
    print(f"Wrote {SPLIT_PATH}")


if __name__ == "__main__":
    main()
