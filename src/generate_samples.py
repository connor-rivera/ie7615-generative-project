"""Generate the five reproducible prompt-to-image samples for Milestone 1."""

import argparse
import json
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DEFAULT_OUTPUT_DIR = Path("results/milestone1")
DEFAULT_SEED = 7615
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
IMAGE_SIZE = 512

PROMPTS = [
    "A red double-decker bus driving through a rainy city street at night.",
    "A golden retriever wearing sunglasses on a sunny beach.",
    "A bowl of fresh fruit on a wooden table beside a bright window.",
    "An astronaut riding a horse across a desert under a blue sky.",
    "A small sailboat on a calm lake surrounded by snowy mountains.",
]

NEGATIVE_PROMPT = "blurry, distorted, low quality, duplicate, text, watermark"


def select_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pipeline_dtype(device):
    return torch.float16 if device.type in {"cuda", "mps"} else torch.float32


def seeded_generator(device, seed):
    generator_device = "cpu" if device.type == "mps" else device.type
    return torch.Generator(device=generator_device).manual_seed(seed)


def save_grid(images, path):
    columns = 3
    rows = 2
    grid = Image.new("RGB", (columns * IMAGE_SIZE, rows * IMAGE_SIZE), "white")
    for index, image in enumerate(images):
        x = (index % columns) * IMAGE_SIZE
        y = (index // columns) * IMAGE_SIZE
        grid.paste(image.convert("RGB"), (x, y))
    grid.save(path)


def generate_samples(
    output_dir=DEFAULT_OUTPUT_DIR,
    model_id=MODEL_ID,
    seed=DEFAULT_SEED,
    steps=DEFAULT_STEPS,
    guidance_scale=DEFAULT_GUIDANCE_SCALE,
    device_name="auto",
):
    if len(PROMPTS) != 5:
        raise RuntimeError(f"Milestone 1 requires exactly five prompts, found {len(PROMPTS)}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device_name)
    dtype = pipeline_dtype(device)
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(device)
    pipeline.enable_attention_slicing()

    images = []
    runs = []
    for index, prompt in enumerate(PROMPTS, start=1):
        run_seed = seed + index - 1
        result = pipeline(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=seeded_generator(device, run_seed),
        )
        image = result.images[0]
        image_name = f"run_{index:02d}_seed_{run_seed}.png"
        image.save(output_dir / image_name)
        images.append(image)

        nsfw_flags = result.nsfw_content_detected
        runs.append(
            {
                "run": index,
                "prompt": prompt,
                "seed": run_seed,
                "image": image_name,
                "nsfw_detected": bool(nsfw_flags[0]) if nsfw_flags else None,
            }
        )
        print(f"Saved {image_name}: {prompt}")

    grid_name = "five_prompt_grid.png"
    save_grid(images, output_dir / grid_name)

    metadata = {
        "model_id": model_id,
        "device": str(device),
        "dtype": str(dtype),
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "negative_prompt": NEGATIVE_PROMPT,
        "grid": grid_name,
        "runs": runs,
    }
    with open(output_dir / "runs.json", "w") as file:
        json.dump(metadata, file, indent=2)

    print(f"Generated five samples in {output_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate five reproducible Stable Diffusion 1.5 samples."
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "mps", "cpu"),
        default="auto",
    )
    args = parser.parse_args()

    generate_samples(
        output_dir=args.output,
        model_id=args.model_id,
        seed=args.seed,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
