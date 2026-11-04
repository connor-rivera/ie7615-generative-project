"""Prompt selection (experiment design, not cleaning).

One caption per image becomes the generation prompt, giving one generated image per real image
(COCO FID-30k convention). Each pick is seeded by (seed, image_id), so changing a cleaning
threshold later does not reshuffle the prompts of the other images.

Called from notebooks/03_text_embeddings.ipynb, which writes manifests/prompts.json.
Every later notebook reads prompts.json, so all experiments use identical prompts.
"""
import json
import random

from src.data import MANIFEST_DIR

PROMPT_SEED = 7615
PROMPTS_PATH = MANIFEST_DIR / "prompts.json"


def select_prompts(manifest, seed=PROMPT_SEED):
    """Return {image_id: prompt} with one caption per manifest image."""
    return {e["image_id"]: random.Random(f"{seed}:{e['image_id']}").choice(e["captions"])
            for e in manifest["images"]}


def write_prompts(prompts, seed=PROMPT_SEED, path=PROMPTS_PATH):
    with open(path, "w") as f:
        json.dump({"seed": seed, "n_prompts": len(prompts),
                   "prompts": [{"image_id": i, "prompt": p} for i, p in sorted(prompts.items())]},
                  f, indent=1)


def load_prompts(path=PROMPTS_PATH):
    """Return {image_id: prompt}."""
    with open(path) as f:
        return {e["image_id"]: e["prompt"] for e in json.load(f)["prompts"]}
