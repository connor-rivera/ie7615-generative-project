"""Baseline + one-knob-at-a-time sweeps on stock Stable Diffusion 1.5 (inference only).

Every image uses the 15 held-out sweep captions (manifests/sweep_prompts.json) with their
fixed seeds, and OUR CLIP encoder's embeddings (caption + "" blank for classifier-free
guidance). Only one setting changes per sweep; the rest stay at BASELINE.

  python -m src.sweep baseline      # 15 images: DDIM, 30 steps, guidance 7.5
  python -m src.sweep guidance      # guidance 1, 3, 5, 7.5, 10, 15
  python -m src.sweep sampler       # DDIM, Euler, DPM++
  python -m src.sweep steps         # 10, 20, 30, 50
  python -m src.sweep all

Images that already exist are skipped, so shared settings are generated once and an
interrupted run resumes where it stopped. Each new image appends one row to the run log.
"""
import argparse
import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import diffusers
import torch
import transformers
from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler,
                       StableDiffusionPipeline)

from src.data import MANIFEST_DIR, PROJECT_ROOT
from src.generate_samples import MODEL_ID, pipeline_dtype, seeded_generator, select_device
from text_embeddings.encode import TextEmbeddingEncoder

OUT_DIR = PROJECT_ROOT / "results" / "part2"
IMG_DIR = OUT_DIR / "images"
LOG_PATH = OUT_DIR / "run_log.csv"
META_PATH = OUT_DIR / "run_meta.json"

SAMPLERS = {"ddim": DDIMScheduler, "euler": EulerDiscreteScheduler,
            "dpmpp": DPMSolverMultistepScheduler}
BASELINE = {"sampler": "ddim", "steps": 30, "guidance": 7.5}
SWEEPS = {
    "baseline": ("baseline", [None]),
    "guidance": ("guidance", [1, 3, 5, 7.5, 10, 15]),
    "sampler": ("sampler", ["ddim", "euler", "dpmpp"]),
    "steps": ("steps", [10, 20, 30, 50]),
}
LOG_FIELDS = ["run_id", "timestamp", "knob", "value", "image_id", "type", "caption", "seed",
              "sampler", "steps", "guidance", "negative", "seconds", "clip_score", "file"]


def conditions(sweep):
    """[(knob, value, settings)] for one sweep; settings = BASELINE with one key changed.

    'best' = the combination proposed by src.select_settings (results/part2/best_settings.json).
    """
    if sweep == "best":
        s = json.loads((OUT_DIR / "best_settings.json").read_text())["settings"]
        s = {"sampler": s["sampler"], "steps": int(s["steps"]), "guidance": float(s["guidance"])}
        base = {**BASELINE, "guidance": float(BASELINE["guidance"])}
        return [("baseline", "-", BASELINE)] if s == base else [("best", "-", s)]
    knob, values = SWEEPS[sweep]
    out = []
    for v in values:
        s = dict(BASELINE)
        if knob != "baseline":
            s[knob] = v
        is_base = s == BASELINE
        out.append(("baseline" if is_base else knob, "-" if is_base else v, s))
    return out


def image_name(s, image_id, seed):
    return f"{s['sampler']}_s{s['steps']}_g{s['guidance']}_img{image_id}_seed{seed}.png"


def write_meta(device, dtype):
    META_PATH.write_text(json.dumps({
        "note": ("Inference only: Stable Diffusion 1.5 and its CLIP text encoder are frozen. "
                 "No weights are trained here; this run log is the record of every generated "
                 "image and its settings."),
        "model_id": MODEL_ID, "device": str(device), "dtype": str(dtype),
        "baseline": BASELINE, "negative": '"" (blank; classifier-free guidance)',
        "prompts": "manifests/sweep_prompts.json (15 held-out captions, seeds 7615-7629)",
        "versions": {"torch": torch.__version__, "diffusers": diffusers.__version__,
                     "transformers": transformers.__version__,
                     "python": platform.python_version()},
    }, indent=1))


def run(sweeps):
    prompts = json.load(open(MANIFEST_DIR / "sweep_prompts.json"))["prompts"]
    todo = [c for name in sweeps for c in conditions(name)]
    # drop repeats (the baseline settings appear in every sweep)
    seen, unique = set(), []
    for knob, value, s in todo:
        key = tuple(sorted(s.items()))
        if key not in seen:
            seen.add(key)
            unique.append((knob, value, s))

    device = select_device()
    dtype = pipeline_dtype(device)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    write_meta(device, dtype)

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=dtype, use_safetensors=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    base_scheduler_config = pipe.scheduler.config

    encoder = TextEmbeddingEncoder(device=str(device), dtype=dtype)
    captions = [p["prompt"] for p in prompts]
    caption_embeds, blank_embeds = encoder.encode_with_blank(captions)

    new_log = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        log = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new_log:
            log.writeheader()
        for knob, value, s in unique:
            pipe.scheduler = SAMPLERS[s["sampler"]].from_config(base_scheduler_config)
            made = 0
            for k, p in enumerate(prompts):
                name = image_name(s, p["image_id"], p["seed"])
                if (IMG_DIR / name).exists():
                    continue
                t0 = time.time()
                image = pipe(prompt_embeds=caption_embeds[k:k + 1],
                             negative_prompt_embeds=blank_embeds[k:k + 1],
                             num_inference_steps=s["steps"], guidance_scale=s["guidance"],
                             generator=seeded_generator(device, p["seed"])).images[0]
                secs = time.time() - t0
                image.save(IMG_DIR / name)
                log.writerow({
                    "run_id": (f"{knob}-{p['image_id']}" if value == "-"
                               else f"{knob}-{value}-{p['image_id']}"),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "knob": knob, "value": value, "image_id": p["image_id"],
                    "type": p["type"], "caption": p["prompt"], "seed": p["seed"],
                    "sampler": s["sampler"], "steps": s["steps"], "guidance": s["guidance"],
                    "negative": "blank", "seconds": f"{secs:.2f}", "clip_score": "",
                    "file": f"images/{name}"})
                f.flush()
                made += 1
            print(f"{knob}={value}: {made} new images ({s})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", choices=[*SWEEPS, "best", "all"])
    args = parser.parse_args()
    run(list(SWEEPS) if args.sweep == "all" else [args.sweep])


if __name__ == "__main__":
    main()
