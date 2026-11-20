"""Step 1 check: our CLIP encoder is correctly wired into Stable Diffusion.

Generates the same caption with the same seed twice:
  A) pipeline(prompt=text)                      -> SD encodes the text itself
  B) pipeline(prompt_embeds=..., negative_prompt_embeds=...)  -> our encoder does it
and compares every pixel. Identical images prove the hand-off is exact, including
the "" (blank) embedding that classifier-free guidance uses.

Run: python -m src.check_prompt_embeds
"""
import json
import time

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

from src.data import MANIFEST_DIR
from src.generate_samples import MODEL_ID, pipeline_dtype, seeded_generator, select_device
from text_embeddings.encode import TextEmbeddingEncoder

STEPS = 30
GUIDANCE = 7.5


def main():
    first = json.load(open(MANIFEST_DIR / "sweep_prompts.json"))["prompts"][0]
    caption, seed = first["prompt"], first["seed"]

    device = select_device()
    dtype = pipeline_dtype(device)
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, use_safetensors=True).to(device)
    pipe.set_progress_bar_config(disable=True)

    # A) text in: the pipeline encodes the caption and "" itself
    t0 = time.time()
    img_text = pipe(prompt=caption, negative_prompt="", num_inference_steps=STEPS,
                    guidance_scale=GUIDANCE, generator=seeded_generator(device, seed)).images[0]
    t_text = time.time() - t0

    # B) embeddings in: our encoder makes the caption + blank pair
    encoder = TextEmbeddingEncoder(device=str(device), dtype=dtype)
    caption_embeds, blank_embeds = encoder.encode_with_blank(caption)
    t0 = time.time()
    img_embeds = pipe(prompt_embeds=caption_embeds, negative_prompt_embeds=blank_embeds,
                      num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                      generator=seeded_generator(device, seed)).images[0]
    t_embeds = time.time() - t0

    a, b = np.asarray(img_text, dtype=np.int16), np.asarray(img_embeds, dtype=np.int16)
    diff = np.abs(a - b)
    print(f"caption: {caption!r}  seed={seed}  device={device}  dtype={dtype}")
    print(f"embeds: caption {tuple(caption_embeds.shape)} {caption_embeds.dtype}, "
          f"blank {tuple(blank_embeds.shape)}; blank is all zeros? "
          f"{bool((blank_embeds == 0).all())}")
    print(f"time: text {t_text:.1f}s, embeds {t_embeds:.1f}s")
    print(f"pixels differing: {(diff > 0).any(axis=-1).sum():,} of {a.shape[0] * a.shape[1]:,}"
          f"  (max difference {diff.max()})")
    print("RESULT:", "IDENTICAL - encoder is wired in correctly" if diff.max() == 0
          else "DIFFERENT - check dtype/tokenizer settings")


if __name__ == "__main__":
    main()
