
"""
Text embedding module for our COCO diffusion project.
Converts text prompts into CLIP embeddings compatible with
Stable Diffusion 1.5.
This module only handles text encoding.
"""

import json
import argparse
from pathlib import Path

import torch
from transformers import CLIPTokenizer, CLIPTextModel


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


class TextEmbeddingEncoder:
    def __init__(self, model_id=MODEL_ID, device=None):


        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        self.model_id = model_id

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer"
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder"
        ).to(self.device)

        # We are using pretrained CLIP representations.
        # No gradients or weight updates are needed for our baseline.
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        print(f"CLIP encoder loaded on {self.device}")

    def encode(self, prompts):


        if isinstance(prompts, str):
            prompts = [prompts]

        if not isinstance(prompts, (list, tuple)):
            raise TypeError("Prompts must be a string or list of strings.")

        if not prompts or any(
            not isinstance(p, str) or not p.strip()
            for p in prompts
        ):
            raise ValueError("Every prompt must contain non-empty text.")

        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )

        input_ids = tokens.input_ids.to(self.device)

        with torch.inference_mode():
            embeddings = self.text_encoder(
                input_ids=input_ids
            ).last_hidden_state

        expected_shape = (
            len(prompts),
            self.tokenizer.model_max_length,
            self.text_encoder.config.hidden_size
        )

        if tuple(embeddings.shape) != expected_shape:
            raise RuntimeError(
                f"Expected {expected_shape}, got {tuple(embeddings.shape)}"
            )

        if not torch.isfinite(embeddings).all().item():
            raise RuntimeError("Embeddings contain NaN or infinity.")

        return embeddings

    def save(self, prompts, output_dir="text_embeddings/outputs"):

        if isinstance(prompts, str):
            prompts = [prompts]

        embeddings = self.encode(prompts)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            embeddings.cpu(),
            output_dir / "embeddings.pt"
        )

        metadata = {
            "model_id": self.model_id,
            "shape": list(embeddings.shape),
            "prompts": list(prompts)
        }

        with open(output_dir / "prompts.json", "w") as file:
            json.dump(metadata, file, indent=2)

        print("Embedding shape:", tuple(embeddings.shape))
        print("Validation: PASSED")
        print("Files saved to:", output_dir)

        return embeddings


def main():
    # This command-line interface lets teammates supply their
    # own prompts without editing the encoder implementation.
    parser = argparse.ArgumentParser(
        description="Generate SD 1.5 CLIP text embeddings"
    )

    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument(
        "--prompt",
        type=str,
        help="One caption supplied directly"
    )

    source.add_argument(
        "--prompts-file",
        type=str,
        help="Path to a JSON file containing a list of captions"
    )

    parser.add_argument(
        "--output",
        default="text_embeddings/outputs",
        help="Directory for generated embeddings"
    )

    args = parser.parse_args()

    if args.prompt is not None:
        prompts = [args.prompt]
    else:
        with open(args.prompts_file) as file:
            prompts = json.load(file)

    encoder = TextEmbeddingEncoder()
    encoder.save(prompts, args.output)


if __name__ == "__main__":
    main()
