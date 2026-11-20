"""Before/after evaluation: stock SD vs SD + COCO LoRA, on the 1,000 HELD-OUT images.

  python -m src.evaluate real                          # save the 1,000 real held-out photos (512 crop)
  python -m src.evaluate generate stock                # 1,000 images, stock SD, best settings
  python -m src.evaluate generate lora --lora DIR      # same captions + seeds, LoRA fused at scale 1
  python -m src.evaluate metrics                       # KID, FID, IS, CLIP score -> metrics.json
  python -m src.evaluate grids --lora DIR              # before/after, generality, training progress

Same caption (prompts.json) and same seed (= image_id) per image in every condition, generated
from our CLIP encoder's embeddings + "" blank, at the settings in results/part2/best_settings.json.
"""
import argparse
import csv
import json
import textwrap
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw

from src.data import MANIFEST_DIR, PROJECT_ROOT, load_image, load_manifest
from src.generate_samples import PROMPTS as M1_PROMPTS
from src.generate_samples import MODEL_ID, pipeline_dtype, seeded_generator, select_device
from src.grids import THUMB, LABEL_W, HEADER_H, _font
from src.prompts import load_prompts
from src.split import load_split
from src.sweep import OUT_DIR as PART2_DIR, SAMPLERS
from text_embeddings.encode import TextEmbeddingEncoder

EVAL_DIR = PROJECT_ROOT / "results" / "eval"
REAL_DIR = EVAL_DIR / "real"


def best_settings():
    d = json.loads((PART2_DIR / "best_settings.json").read_text())
    s = d["settings"]
    return {"sampler": s["sampler"], "steps": int(s["steps"]), "guidance": float(s["guidance"])}


def save_real():
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    by_id = {e["image_id"]: e for e in load_manifest()["images"]}
    for i in load_split()["held_out"]:
        p = REAL_DIR / f"img{i}.png"
        if not p.exists():
            load_image(by_id[i]).save(p)
    print(f"{len(list(REAL_DIR.glob('*.png')))} real held-out images in {REAL_DIR}")


def make_pipe(settings, lora=None, scale=1.0):
    device = select_device()
    dtype = pipeline_dtype(device)
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype,
                                                   use_safetensors=True).to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.scheduler = SAMPLERS[settings["sampler"]].from_config(pipe.scheduler.config)
    if lora:
        pipe.load_lora_weights(lora, adapter_name="coco")
        pipe.fuse_lora(lora_scale=scale)
    return pipe, device, dtype


def generate(cond, lora=None, scale=1.0, limit=None):
    s = best_settings()
    out = EVAL_DIR / cond
    out.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts()
    ids = load_split()["held_out"][:limit]
    todo = [i for i in ids if not (out / f"img{i}.png").exists()]
    print(f"{cond}: {len(todo)} of {len(ids)} to generate at {s}" + (f", LoRA {lora} x{scale}"
                                                                        if lora else ""))
    if not todo:
        return
    pipe, device, dtype = make_pipe(s, lora, scale)
    enc = TextEmbeddingEncoder(device=str(device), dtype=dtype)
    log_path = EVAL_DIR / "eval_log.csv"
    new = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["cond", "image_id", "caption", "seed", "sampler", "steps", "guidance",
                        "lora", "lora_scale", "seconds", "file"])
        t_all = time.time()
        for n, i in enumerate(todo, 1):
            cap = prompts[i]
            c, b = enc.encode_with_blank(cap)
            t0 = time.time()
            img = pipe(prompt_embeds=c, negative_prompt_embeds=b,
                       num_inference_steps=s["steps"], guidance_scale=s["guidance"],
                       generator=seeded_generator(device, i)).images[0]
            img.save(out / f"img{i}.png")
            w.writerow([cond, i, cap, i, s["sampler"], s["steps"], s["guidance"],
                        lora or "", scale if lora else 0, f"{time.time() - t0:.2f}",
                        f"{cond}/img{i}.png"])
            if n % 25 == 0:
                f.flush()
                rate = (time.time() - t_all) / n
                print(f"  {n}/{len(todo)}  {rate:.1f}s/img  ~{rate * (len(todo) - n) / 60:.0f} min left",
                      flush=True)


def metrics():
    import torch_fidelity
    from src.clip_score import ClipScorer
    prompts = load_prompts()
    held = load_split()["held_out"]
    results = {}
    scorer = None
    for cond in sorted(d.name for d in EVAL_DIR.iterdir()
                       if d.is_dir() and d.name not in {"real", "grids"}):
        files = sorted((EVAL_DIR / cond).glob("img*.png"))
        if len(files) < len(held):
            print(f"skip {cond}: only {len(files)} images")
            continue
        m = torch_fidelity.calculate_metrics(
            input1=str(EVAL_DIR / cond), input2=str(REAL_DIR), cuda=False, fid=True, kid=True,
            isc=True, kid_subset_size=500, kid_subsets=100, verbose=False)
        scorer = scorer or ClipScorer()
        ids = [int(p.stem[3:]) for p in files]
        clip = scorer.score(files, [prompts[i] for i in ids])
        results[cond] = {"n": len(files), "FID": m["frechet_inception_distance"],
                         "KID_mean": m["kernel_inception_distance_mean"],
                         "KID_std": m["kernel_inception_distance_std"],
                         "IS_mean": m["inception_score_mean"], "IS_std": m["inception_score_std"],
                         "CLIP_mean": sum(clip) / len(clip)}
        print(cond, {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in results[cond].items()})
    (EVAL_DIR / "metrics.json").write_text(json.dumps(
        {"settings": best_settings(), "reference": "1,000 held-out real COCO val2017 images "
         "(512 center crop)", "kid_subset_size": 500, "results": results}, indent=1))


def _grid(rows, col_labels, cells, path):
    """rows: list of row labels; cells(r, c) -> PIL image or None."""
    W, H = LABEL_W + THUMB * len(col_labels), HEADER_H + THUMB * len(rows)
    g = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(g)
    for c, lab in enumerate(col_labels):
        d.text((LABEL_W + c * THUMB + 8, 10), lab, fill="black", font=_font(18))
    for r, lab in enumerate(rows):
        y = HEADER_H + r * THUMB
        d.multiline_text((8, y + 8), "\n".join(textwrap.wrap(lab, 30)), fill="black",
                         font=_font(15), spacing=3)
        for c in range(len(col_labels)):
            im = cells(r, c)
            if im is not None:
                g.paste(im.convert("RGB").resize((THUMB, THUMB), Image.BICUBIC),
                        (LABEL_W + c * THUMB, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    g.save(path, quality=90)
    print(f"Saved {path}")


def grids(lora, scales=(0.0, 0.5, 1.0)):
    s = best_settings()
    sweep = json.load(open(MANIFEST_DIR / "sweep_prompts.json"))["prompts"]
    by_id = {e["image_id"]: e for e in load_manifest()["images"]}
    pipe, device, dtype = make_pipe(s)
    pipe.load_lora_weights(lora, adapter_name="coco")
    enc = TextEmbeddingEncoder(device=str(device), dtype=dtype)
    cache = EVAL_DIR / "grids" / "cells"
    cache.mkdir(parents=True, exist_ok=True)

    def gen(prompt, seed, scale, tag):
        p = cache / f"{tag}_seed{seed}_x{scale}.png"
        if not p.exists():
            pipe.set_adapters(["coco"], adapter_weights=[scale])
            c, b = enc.encode_with_blank(prompt)
            pipe(prompt_embeds=c, negative_prompt_embeds=b, num_inference_steps=s["steps"],
                 guidance_scale=s["guidance"], generator=seeded_generator(device, seed)
                 ).images[0].save(p)
        return Image.open(p)

    labels = [f"LoRA x{x}" + (" (stock)" if x == 0 else "") for x in scales]
    _grid([f"[{p['type']}] {p['prompt']}" for p in sweep], ["real COCO photo"] + labels,
          lambda r, c: load_image(by_id[sweep[r]["image_id"]]) if c == 0 else
          gen(sweep[r]["prompt"], sweep[r]["seed"], scales[c - 1], f"img{sweep[r]['image_id']}"),
          EVAL_DIR / "grids" / "grid_lora_before_after.jpg")
    _grid([f"[non-COCO] {p}" for p in M1_PROMPTS], labels,
          lambda r, c: gen(M1_PROMPTS[r], 7615 + r, scales[c], f"m1_{r}"),
          EVAL_DIR / "grids" / "grid_generality.jpg")

    samples = Path(lora).parents[1] / "samples"
    steps = sorted({int(p.stem.split("_")[0][4:]) for p in samples.glob("step*_p0.png")})
    from src.train_lora import SAMPLE_PROMPTS
    _grid([f"step {st}" + (" (stock)" if st == 0 else "") for st in steps],
          [textwrap.shorten(p, 34) for p, _ in SAMPLE_PROMPTS],
          lambda r, c: Image.open(samples / f"step{steps[r]:05d}_p{c}.png"),
          EVAL_DIR / "grids" / "grid_training_progress.jpg")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("real")
    g = sub.add_parser("generate")
    g.add_argument("cond")
    g.add_argument("--lora")
    g.add_argument("--scale", type=float, default=1.0)
    g.add_argument("--limit", type=int)
    sub.add_parser("metrics")
    gr = sub.add_parser("grids")
    gr.add_argument("--lora", required=True)
    a = ap.parse_args()
    if a.cmd == "real":
        save_real()
    elif a.cmd == "generate":
        generate(a.cond, a.lora, a.scale, a.limit)
    elif a.cmd == "metrics":
        metrics()
    else:
        grids(a.lora)


if __name__ == "__main__":
    main()
