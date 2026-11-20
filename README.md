# Text-to-image diffusion on COCO dataset

Conditioning a pretrained Stable Diffusion 1.5 on CLIP text embeddings, experimenting with noise schedules, sampling steps, conditioning strength, and classifier-free guidance, and evaluating with FID/KID, CLIP Score, and qualitative review.

**Status:** dataset prepared (4,785 images w/ 23,925 captions), CLIP text-embedding baseline implemented, and five Stable Diffusion samples generated.

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
Tested with Python 3.13 and 3.14 on macOS (Apple Silicon, PyTorch `mps` backend).

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

## 4. Build the text-embedding baseline

Create one deterministic caption prompt for each cleaned image:

```bash
python -m src.prompts
```

Open `notebooks/04_text_embeddings.ipynb` and run all cells. The notebook writes `manifests/prompts.json` with 4,785 image-caption pairs and saves the five-example CLIP tensor, metadata, and cosine-similarity visualization to `results/milestone1/text_embeddings/`.

## 5. Generate the five Milestone 1 images

Open `notebooks/03_five_prompt_image_runs.ipynb` and run all cells, or run:

```bash
python -m src.generate_samples
```

The pipeline uses Stable Diffusion 1.5 with five fixed prompts and deterministic seeds. It saves five 512×512 images, a comparison grid, and `runs.json` to `results/milestone1/`.

## Repository layout
| Path | Contents |
|---|---|
| `src/clean.py` | Cleaning rules (`python -m src.clean`) |
| `src/data.py` | `load_manifest()`, `load_image()`, `center_crop_resize()` |
| `src/prompts.py` | One seeded caption per image as the generation prompt |
| `src/generate_samples.py` | Five reproducible Stable Diffusion 1.5 prompt-to-image runs |
| `notebooks/01_data_integrity_check.ipynb` | Verifies the download: counts, ID mapping, readable files |
| `notebooks/02_cleaning.ipynb` | Cleaning walkthrough with visual checks |
| `notebooks/03_five_prompt_image_runs.ipynb` | Generates and displays the five sample images |
| `notebooks/04_text_embeddings.ipynb` | Demonstrates the CLIP text-to-embedding baseline |
| `manifests/` | Cleaned dataset, drop log, summary, and deterministic prompt manifest |
| `results/milestone1/` | Generated images, run metadata, and embedding evidence |

## Data license
COCO annotations are CC BY 4.0. Images remain under their individual Flickr licenses and are
not redistributed here. Download them from the official source above.


---

## Milestone 2: Model Integration, Experiments & LoRA Fine-Tuning

**Status: Completed**

Milestone 2 builds on the COCO dataset preparation and CLIP text-embedding pipeline developed in Milestone 1.

The objective was to integrate CLIP with Stable Diffusion 1.5, perform controlled image-generation experiments, and fine-tune the diffusion model using LoRA.

All experiments were executed using Google Colab Pro with an NVIDIA Tesla T4 GPU.

### 1. CLIP and Stable Diffusion Integration

**Notebook:** `notebooks/05_milestone2_model_integration.ipynb`

Integrated the CLIP text encoder with the pretrained Stable Diffusion 1.5 pipeline.

Two text-conditioning approaches were compared:
- Standard Stable Diffusion prompt encoding.
- Explicit CLIP embeddings passed into the diffusion pipeline.

**Results:**
- Generated five pairs of 512 × 512 images.
- Verified matching positive and negative text embeddings.
- Confirmed identical final latents and image pixels across both approaches.
- Saved generated images and reproducibility metadata.

This validated the explicit Transformer-to-diffusion conditioning pipeline.

### 2. Diffusion Parameter Experiments

**Notebook:** `notebooks/06_milestone2_parameter_experiments.ipynb`

Performed controlled experiments using COCO captions and fixed random seeds.

| Parameter | Configurations |
|---|---|
| Classifier-free guidance | 1.0, 4.0, 7.5, 10.0 |
| Inference steps | 15, 30, 50 |
| Noise schedulers | PNDM, DDIM, Euler |
| Resolution | 512 × 512 |
| Baseline guidance | 7.5 |
| Baseline steps | 30 |

**Results:**
- Generated 5 baseline images, 21 parameter-sweep images, and 1 smoke-test image.
- Produced comparison grids for guidance strength, sampling steps, and schedulers.
- Recorded experiment configurations, inference times, and run metadata.

**Observations:**
- CFG 7.5 produced reasonable prompt adherence across the tested examples.
- Increasing sampling steps from 30 to 50 did not consistently improve visual quality.
- Different schedulers produced variations in image composition.
- Results are exploratory because the parameter sweeps used a small prompt set and one seed per prompt.

### 3. LoRA Fine-Tuning

**Notebook:** `notebooks/07_lora_finetuning.ipynb`

Fine-tuned Stable Diffusion 1.5 using Low-Rank Adaptation (LoRA).

A food-related subset of COCO 2017 training images was used.

**Training configuration:**

| Parameter | Value |
|---|---|
| Base model | Stable Diffusion 1.5 |
| Dataset | COCO train2017, food-related subset |
| Training images | 500 |
| Image resolution | 512 × 512 |
| LoRA rank | 8 |
| Learning rate | 0.0001 |
| Batch size | 1 |
| Gradient accumulation | 4 |
| Optimizer updates | 300 |
| Trainable parameters | 1,594,368 |
| GPU | NVIDIA Tesla T4 |

The CLIP text encoder and VAE remained frozen. LoRA adapters were trained on selected UNet attention layers.

**Results:**
- Successfully completed 300 optimizer updates.
- Saved LoRA adapter weights and training logs.
- Generated training-loss visualizations.
- Compared pretrained and fine-tuned model outputs on held-out COCO prompts.
- Performed an additional custom pizza-prompt comparison.

Fine-tuning changed the generated images, but visual comparisons did not establish consistent improvements in image quality.

One pretrained comparison image was blocked by the safety checker and was not treated as an image-quality failure.

### 4. Generated Results

The Milestone 2 results are organized under:

`results/milestone2/`

The saved results include:

- Baseline image gallery.
- Classifier-free guidance comparison.
- Sampling-step comparison.
- Scheduler comparison.
- Pretrained versus LoRA image comparisons.
- Training-loss graph and experiment logs.
- Selected individual generated images.

The larger training datasets and model checkpoint files are stored separately in Google Drive rather than committed to GitHub.

### 5. Running Milestone 2

The notebooks are designed for Google Colab with an NVIDIA GPU.

Run the notebooks in the following order:

1. `05_milestone2_model_integration.ipynb`
2. `06_milestone2_parameter_experiments.ipynb`
3. `07_lora_finetuning.ipynb`

Each notebook contains its own setup instructions and Google Drive output handling.

Notebook 07 requires access to its selected COCO training images and captions. The pretrained model is obtained through Hugging Face Diffusers.

Training results and LoRA checkpoints are saved persistently in Google Drive.

### 6. Milestone 2 Deliverables

- [x] CLIP and diffusion model integration
- [x] Baseline conditional image generation
- [x] Classifier-free guidance experiments
- [x] Noise scheduler and sampling-step experiments
- [x] 5–10 early image samples
- [x] LoRA fine-tuning and training logs
- [x] Generated comparison images
- [x] One-page observations summary prepared



