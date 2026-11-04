"""Loading the cleaned dataset.

The cleaned dataset = original COCO val2017 images (unchanged on disk)
                    + manifests/manifest.json (which images/captions are kept)
                    + center_crop_resize() (how each image is processed at load time).
"""
import json
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = PROJECT_ROOT / "val2017"
ANN_DIR = PROJECT_ROOT / "annotations"
MANIFEST_DIR = PROJECT_ROOT / "manifests"
MANIFEST_PATH = MANIFEST_DIR / "manifest.json"

TARGET_SIZE = 512  # Stable Diffusion 1.5 native resolution
RESAMPLE = Image.BICUBIC


def load_manifest(path=MANIFEST_PATH):
    """Return the manifest dict. Image entries are in manifest["images"]."""
    with open(path) as f:
        return json.load(f)


def center_crop_resize(img, size=TARGET_SIZE):
    """Center-crop to a square on the short side, then resize to size x size. Returns RGB."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), RESAMPLE)


def load_image(entry, img_dir=IMG_DIR, size=TARGET_SIZE, transform=True):
    """Open one manifest entry's image. transform=False returns the original (RGB) image."""
    with Image.open(Path(img_dir) / entry["file_name"]) as im:
        im.load()
        return center_crop_resize(im, size) if transform else im.convert("RGB")
