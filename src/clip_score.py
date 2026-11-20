"""CLIP score: how well a generated image matches its caption.

score = 100 * max(cos(CLIP_image(img), CLIP_text(caption)), 0)   (torchmetrics convention)
Higher is better; typical text-to-image values are ~25-35.

Model: openai/clip-vit-large-patch14 (ViT-L/14). Caveat for the write-up: SD 1.5's text
encoder is this same CLIP's text half, so CLIP score is a somewhat "home-field" judge.
It measures caption match, not realism (that's KID/FID).

  python -m src.clip_score run_log     # fills the clip_score column of results/part2/run_log.csv
"""
import csv
import sys

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.generate_samples import select_device
from src.sweep import LOG_FIELDS, LOG_PATH, OUT_DIR

CLIP_ID = "openai/clip-vit-large-patch14"


class ClipScorer:
    def __init__(self, device=None):
        self.device = device or select_device()
        self.model = CLIPModel.from_pretrained(CLIP_ID).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_ID)

    @torch.inference_mode()
    def score(self, images, captions, batch_size=16):
        """images: list of PIL images or paths; returns list of floats."""
        out = []
        for i in range(0, len(images), batch_size):
            imgs = [Image.open(x).convert("RGB") if not isinstance(x, Image.Image) else x
                    for x in images[i:i + batch_size]]
            inputs = self.processor(text=captions[i:i + batch_size], images=imgs,
                                    return_tensors="pt", padding=True, truncation=True,
                                    max_length=77).to(self.device)
            # full forward gives the projected embeddings as tensors (transformers 5's
            # get_image_features returns a model-output object, not a tensor)
            o = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                           pixel_values=inputs["pixel_values"])
            img_emb, txt_emb = o.image_embeds, o.text_embeds
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
            cos = (img_emb * txt_emb).sum(dim=-1)
            out += (100 * cos.clamp(min=0)).float().cpu().tolist()
        return out


def fill_run_log():
    rows = list(csv.DictReader(open(LOG_PATH)))
    todo = [r for r in rows if not r["clip_score"]]
    if todo:
        scores = ClipScorer().score([OUT_DIR / r["file"] for r in todo],
                                    [r["caption"] for r in todo])
        for r, s in zip(todo, scores):
            r["clip_score"] = f"{s:.2f}"
    with open(LOG_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Scored {len(todo)} images; {len(rows)} rows in {LOG_PATH}")


if __name__ == "__main__":
    if sys.argv[1:] == ["run_log"]:
        fill_run_log()
    else:
        sys.exit(__doc__)
