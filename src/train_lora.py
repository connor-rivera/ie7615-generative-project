"""LoRA fine-tuning of Stable Diffusion 1.5's U-Net on the cleaned COCO TRAIN split.

What is trained: rank-r LoRA adapters on the U-Net's attention projections
(to_q, to_k, to_v, to_out.0: self- and cross-attention). Everything else (U-Net base weights,
VAE, CLIP text encoder) is frozen.

One training step = the standard diffusion training quiz (Diffusion deck slides 23, 25-26):
  real image -> VAE latent (cached) ; caption -> CLIP embedding (10% replaced by "" so
  classifier-free guidance keeps working) ; add noise at a random timestep ; U-Net + LoRA
  predicts the noise ; loss = MSE(predicted, true) ; update ONLY the LoRA weights.

Logged (results/lora/<run>/):
  train_log.csv  every step: loss, smoothed loss, learning rate, seconds/step
  val_log.csv    every --val-every steps: denoising loss on FIXED images / timesteps / noise
                 - val:   64 held-out images (never trained on). Step 0 = stock SD.
                 - probe: 64 training images. probe falling while val rises = overfitting.
  samples/       fixed prompts + seeds rendered at every checkpoint (step 0 = stock SD)
  checkpoints/   LoRA weights (loadable with pipe.load_lora_weights) + optimizer state

  python -m src.train_lora --steps 4000            # resumes automatically if checkpoints exist
"""
import argparse
import csv
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler,
                       StableDiffusionPipeline, UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

from src.data import PROJECT_ROOT, load_image, load_manifest
from src.generate_samples import MODEL_ID, select_device
from src.split import load_split

LATENT_CACHE = PROJECT_ROOT / "results" / "lora" / "latent_cache.pt"
SAMPLE_PROMPTS = [  # 2 held-out COCO captions + 2 non-COCO prompts (generality), fixed seeds
    ("A black and white cow walking across a lush green field.", 7618),
    ("A man and woman are playing catch in the backyard.", 7624),
    ("A golden retriever wearing sunglasses on a sunny beach.", 7616),
    ("An astronaut riding a horse across a desert under a blue sky.", 7618),
]
TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]


def to_tensor(img):
    """PIL 512x512 RGB -> (3, 512, 512) float in [-1, 1]."""
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)


@torch.no_grad()
def build_latent_cache(vae, device, ids_by_name, batch=8, path=LATENT_CACHE):
    """Encode every needed image once; store the VAE posterior mean and std (fp16)."""
    if path.exists():
        cache = torch.load(path)
        if all(i in cache["mean"] for ids in ids_by_name.values() for i in ids):
            return cache
    by_id = {e["image_id"]: e for e in load_manifest()["images"]}
    need = sorted({i for ids in ids_by_name.values() for i in ids})
    mean, std = {}, {}
    t0 = time.time()
    for k in range(0, len(need), batch):
        chunk = need[k:k + batch]
        x = torch.stack([to_tensor(load_image(by_id[i])) for i in chunk]).to(device)
        dist = vae.encode(x).latent_dist
        for j, i in enumerate(chunk):
            mean[i] = dist.mean[j].half().cpu()
            std[i] = dist.std[j].half().cpu()
        if k % 400 == 0:
            print(f"  cached {k + len(chunk)}/{len(need)} latents ({time.time() - t0:.0f}s)")
    cache = {"mean": mean, "std": std}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    print(f"Latent cache: {len(need)} images in {time.time() - t0:.0f}s -> {path}")
    return cache


def sample_latents(cache, ids, scaling, gen=None):
    m = torch.stack([cache["mean"][i] for i in ids]).float()
    s = torch.stack([cache["std"][i] for i in ids]).float()
    eps = torch.randn(m.shape, generator=gen)
    return (m + s * eps) * scaling


@torch.no_grad()
def encode_text(tokenizer, text_encoder, captions, device):
    ids = tokenizer(captions, padding="max_length", truncation=True,
                    max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
    return text_encoder(ids.to(device))[0]


class FixedEval:
    """Deterministic denoising loss: same images, timesteps, latent draws and noise every time."""
    TIMESTEPS = [100, 300, 500, 700, 900]

    def __init__(self, name, ids, captions, cache, scaling, tokenizer, text_encoder, device):
        g = torch.Generator().manual_seed(1234)
        self.name, self.device = name, device
        self.latents = sample_latents(cache, ids, scaling, g)
        self.noise = torch.randn(len(self.TIMESTEPS), *self.latents.shape, generator=g)
        self.text = encode_text(tokenizer, text_encoder, captions, device)

    @torch.no_grad()
    def loss(self, unet, noise_sched, batch=4):
        total, n = 0.0, 0
        for ti, t in enumerate(self.TIMESTEPS):
            for k in range(0, len(self.latents), batch):
                x0 = self.latents[k:k + batch].to(self.device)
                eps = self.noise[ti, k:k + batch].to(self.device)
                ts = torch.full((len(x0),), t, device=self.device, dtype=torch.long)
                pred = unet(noise_sched.add_noise(x0, eps, ts), ts,
                            self.text[k:k + batch]).sample
                total += F.mse_loss(pred.float(), eps, reduction="sum").item()
                n += eps.numel()
        return total / n


@torch.no_grad()
def render_samples(unet, vae, tokenizer, text_encoder, out_dir, step, device):
    sched = DPMSolverMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
                                   unet=unet, scheduler=sched, safety_checker=None,
                                   feature_extractor=None, requires_safety_checker=False)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()  # lowers the float32 sampling memory peak (~30 GB -> less)
    out_dir.mkdir(parents=True, exist_ok=True)
    was_training = unet.training
    unet.eval()
    for k, (prompt, seed) in enumerate(SAMPLE_PROMPTS):
        img = pipe(prompt, num_inference_steps=25, guidance_scale=7.5,
                   generator=torch.Generator().manual_seed(seed)).images[0]
        img.save(out_dir / f"step{step:05d}_p{k}.png")
    unet.train(was_training)


def free_device_memory(device):
    """Return cached GPU buffers to the OS. On mps the allocator otherwise keeps the training
    buffers AND the differently-shaped eval/sampling buffers, grows past physical memory
    (38 GB seen on a 36 GB Mac, 2026-11-20) and swaps until progress stalls."""
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def save_checkpoint(unet, optimizer, lr_sched, step, ckpt_dir):
    d = ckpt_dir / f"step{step:05d}"
    d.mkdir(parents=True, exist_ok=True)
    StableDiffusionPipeline.save_lora_weights(
        d, unet_lora_layers=get_peft_model_state_dict(unet), safe_serialization=True)
    torch.save({"optimizer": optimizer.state_dict(), "lr_sched": lr_sched.state_dict(),
                "step": step}, d / "train_state.pt")
    return d


def latest_checkpoint(ckpt_dir):
    ckpts = sorted(ckpt_dir.glob("step*/train_state.pt")) if ckpt_dir.exists() else []
    return ckpts[-1].parent if ckpts else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="rank8")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=None, help="default = rank (scale 1.0)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--caption-dropout", type=float, default=0.1)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--no-samples", action="store_true")
    ap.add_argument("--seed", type=int, default=7615)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end test: 16 train / 8 val / 8 probe images, separate cache")
    args = ap.parse_args()
    args.alpha = args.alpha or args.rank

    out = PROJECT_ROOT / "results" / "lora" / args.run
    ckpt_dir, sample_dir = out / "checkpoints", out / "samples"
    out.mkdir(parents=True, exist_ok=True)
    device = select_device()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").to(device)
    noise_sched = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)
    text_encoder.eval()
    vae.eval()

    unet.add_adapter(LoraConfig(r=args.rank, lora_alpha=args.alpha, init_lora_weights="gaussian",
                                target_modules=TARGET_MODULES))
    if args.grad_checkpointing:
        unet.enable_gradient_checkpointing()
    params = [p for p in unet.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in unet.parameters())

    # data
    split = load_split()
    by_id = {e["image_id"]: e for e in load_manifest()["images"]}
    train_ids = split["train"][:16] if args.smoke else split["train"]
    n_eval = 8 if args.smoke else 64
    val_ids = sorted(random.Random(1).sample(split["held_out"], n_eval))
    probe_ids = sorted(random.Random(2).sample(train_ids, n_eval))
    scaling = vae.config.scaling_factor
    cache = build_latent_cache(
        vae, device, {"train": train_ids, "val": val_ids},
        path=LATENT_CACHE.with_name("latent_cache_smoke.pt") if args.smoke else LATENT_CACHE)
    first_cap = lambda i: by_id[i]["captions"][0]
    val = FixedEval("val", val_ids, [first_cap(i) for i in val_ids], cache, scaling,
                    tokenizer, text_encoder, device)
    probe = FixedEval("probe", probe_ids, [first_cap(i) for i in probe_ids], cache, scaling,
                      tokenizer, text_encoder, device)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    lr_sched = get_scheduler("cosine", optimizer, num_warmup_steps=args.warmup,
                             num_training_steps=args.steps)

    start = 0
    resume = latest_checkpoint(ckpt_dir)
    if resume:
        sd = StableDiffusionPipeline.lora_state_dict(resume)[0]
        sd = {k.replace("unet.", "", 1): v for k, v in sd.items() if k.startswith("unet.")}
        set_peft_model_state_dict(unet, sd)
        st = torch.load(resume / "train_state.pt")
        optimizer.load_state_dict(st["optimizer"])
        lr_sched.load_state_dict(st["lr_sched"])
        start = st["step"]
        print(f"Resumed from {resume} (step {start})")

    (out / "config.json").write_text(json.dumps({
        **vars(args), "device": str(device), "train_images": len(train_ids),
        "train_captions": sum(len(by_id[i]["captions"]) for i in train_ids),
        "val_images": len(val_ids), "probe_images": len(probe_ids),
        "trainable_params": n_train, "unet_total_params": n_total,
        "target_modules": TARGET_MODULES, "precision": "float32 (training)",
        "noise_schedule": "DDPM scaled_linear, 1000 steps (SD 1.5's own)",
        "loss": "MSE(predicted noise, true noise)"}, indent=1))
    print(f"Trainable LoRA params: {n_train:,} of {n_total:,} ({100 * n_train / n_total:.2f}%)")

    train_log = open(out / "train_log.csv", "a", newline="")
    tw = csv.writer(train_log)
    val_log = open(out / "val_log.csv", "a", newline="")
    vw = csv.writer(val_log)
    if start == 0:
        tw.writerow(["step", "loss", "loss_ema", "lr", "sec_per_step", "elapsed_min"])
        vw.writerow(["step", "val_loss", "probe_loss"])
        unet.eval()
        v0, p0 = val.loss(unet, noise_sched), probe.loss(unet, noise_sched)
        vw.writerow([0, f"{v0:.6f}", f"{p0:.6f}"])
        val_log.flush()
        print(f"step 0 (stock SD): val {v0:.5f}  probe {p0:.5f}")
        if not args.no_samples:
            render_samples(unet, vae, tokenizer, text_encoder, sample_dir, 0, device)

    unet.train()
    ema, t_start = None, time.time()
    order = []
    for step in range(start + 1, args.steps + 1):
        t0 = time.time()
        if len(order) < args.batch:
            order += rng.sample(train_ids, len(train_ids))
        ids, order = order[:args.batch], order[args.batch:]
        caps = ["" if rng.random() < args.caption_dropout else rng.choice(by_id[i]["captions"])
                for i in ids]
        x0 = sample_latents(cache, ids, scaling).to(device)
        eps = torch.randn_like(x0)
        ts = torch.randint(0, noise_sched.config.num_train_timesteps, (len(ids),), device=device)
        text = encode_text(tokenizer, text_encoder, caps, device)

        pred = unet(noise_sched.add_noise(x0, eps, ts), ts, text).sample
        loss = F.mse_loss(pred.float(), eps)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        lr_sched.step()

        lv = loss.item()
        if not math.isfinite(lv):
            raise SystemExit(f"Non-finite loss at step {step}; stopping (last checkpoint kept).")
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        tw.writerow([step, f"{lv:.6f}", f"{ema:.6f}", f"{lr_sched.get_last_lr()[0]:.3e}",
                     f"{time.time() - t0:.3f}", f"{(time.time() - t_start) / 60:.2f}"])
        if step % 25 == 0:
            train_log.flush()
            print(f"step {step}/{args.steps}  loss_ema {ema:.5f}  "
                  f"{(time.time() - t_start) / (step - start):.2f}s/step", flush=True)

        if step % args.val_every == 0 or step == args.steps:
            del pred, loss, x0, eps, text
            free_device_memory(device)
            unet.eval()
            v, p = val.loss(unet, noise_sched), probe.loss(unet, noise_sched)
            vw.writerow([step, f"{v:.6f}", f"{p:.6f}"])
            val_log.flush()
            print(f"  val {v:.5f}  probe {p:.5f}", flush=True)
            unet.train()
            free_device_memory(device)
        if step % args.ckpt_every == 0 or step == args.steps:
            d = save_checkpoint(unet, optimizer, lr_sched, step, ckpt_dir)
            if not args.no_samples:
                render_samples(unet, vae, tokenizer, text_encoder, sample_dir, step, device)
            free_device_memory(device)
            print(f"  checkpoint -> {d}", flush=True)

    train_log.close()
    val_log.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
