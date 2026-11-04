# Text-to-image diffusion on COCO dataset

Conditioning a pretrained Stable Diffusion 1.5 on CLIP text embeddings, experimenting with noise schedules, sampling steps, conditioning strength, and classifier-free guidance, and evaluating with FID/KID, CLIP Score, and qualitative review.

**Status:** dataset prepared (4,785 images w/ 23,925 captions). Embedding pipeline next.

# Get the data
From https://cocodataset.org/\#download, download only:


| File | URL | Size |
|---|---|---|
| 2017 Val images | http://images.cocodataset.org/zips/val2017.zip | 816 MB |
| 2017 Train/Val annotations | http://images.cocodataset.org/annotations/annotations_trainval2017.zip | 253 MB |

Unzip both into the repo root. From `annotations/`, keep only `captions_val2017.json` and
`instances_val2017.json`. The rest are train-split or unused annotation types.

Expected layout:
```
<repo>/
├── val2017/          5,000 .jpg
├── annotations/
│   ├── captions_val2017.json
│   └── instances_val2017.json
├── src/  notebooks/  manifests/
```

## 2. Environment
Tested with Python 3.13 on macOS (Apple Silicon, PyTorch `mps` backend).

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name t2i-diffusion --display-name "Python (t2i-diffusion)"
```
Select the **Python (t2i-diffusion)** kernel when opening the notebooks.

Check the GPU backend:
```bash
python -c "import torch; print('mps:', torch.backends.mps.is_available(), '| cuda:', torch.cuda.is_available())"
```
- **NVIDIA / CUDA:** first install the CUDA build of `torch`/`torchvision` from https://pytorch.org, then `pip install -r requirements.txt`.
- **First run** downloads model weights (CLIP ViT-L/14 ≈ 1.7 GB; Stable Diffusion 1.5 later) to `~/.cache/huggingface/`.

## 3. Rebuild the cleaned dataset
```bash
python -m src.clean
```
No image files are modified. The cleaned dataset is the original images + `manifests/manifest.json`,
with a center-crop → 512×512 (bicubic) transform applied at load time (`src/data.py`).

| Stage | Images | Captions |
|---|---|---|
| Start (COCO val2017) | 5,000 | 25,014 |
| Drop grayscale (10 single-channel + 155 RGB-encoded) | 4,835 | |
| Drop aspect ratio > 2.0 (53) | 4,785 | |
| Remove duplicate captions (13) | 4,785 | 23,925 |

Verify your rebuild matches:
```bash
shasum -a 256 manifests/manifest.json
# db1ffb879cbd46b8c1272036e4c13d7c3f43c509c9e930df14f03072e8d7c888
```
Every dropped image and its reason is in `manifests/drop_log.csv`. Rationale and visual checks
are in `notebooks/02_cleaning.ipynb`.

## Repository layout
| Path | Contents |
|---|---|
| `src/clean.py` | Cleaning rules (`python -m src.clean`) |
| `src/data.py` | `load_manifest()`, `load_image()`, `center_crop_resize()` |
| `src/prompts.py` | One seeded caption per image as the generation prompt |
| `notebooks/01_data_integrity_check.ipynb` | Verifies the download: counts, ID mapping, readable files |
| `notebooks/02_cleaning.ipynb` | Cleaning walkthrough with visual checks |
| `manifests/` | `manifest.json`, `drop_log.csv`, `cleaning_summary.json` |

## Data license
COCO annotations are CC BY 4.0. Images remain under their individual Flickr licenses and are
not redistributed here. Download them from the official source above.
