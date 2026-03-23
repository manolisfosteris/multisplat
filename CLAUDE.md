# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GaussCtrl is a text-driven 3D Gaussian Splatting editing method (ECCV 2024). It edits 3DGS scenes by: (1) rendering all views and inverting them to DDIM latents, (2) editing images using ControlNet+depth with a cross-view attention mechanism (`CrossViewAttnProcessor`) for multi-view consistency, and (3) fine-tuning the 3DGS model on the edited images.

This is a fork with active modifications, primarily in `gc_pipeline.py` and `utils.py`, adding epipolar geometry constraints to the cross-view attention.

## Installation

```bash
conda create -n gaussctrl python=3.8
conda activate gaussctrl
conda install cuda -c nvidia/label/cuda-11.8.0
pip install nerfstudio==1.0.0
pip install gsplat==0.1.3   # or 0.1.2
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
pip install -r requirements.txt
pip install -e .
```

Verify: `ns-train -h`

## Key Commands

**Train a base 3DGS model (prerequisite):**
```bash
ns-train splatfacto --output-dir unedited_models --experiment-name bear nerfstudio-data --data data/bear
```

**Run GaussCtrl editing:**
```bash
ns-train gaussctrl \
  --load-checkpoint {path/to/step-000029999.ckpt} \
  --experiment-name EXPERIMENT_NAME \
  --output-dir outputs \
  --pipeline.datamanager.data {path/to/data} \
  --pipeline.edit_prompt "YOUR EDIT PROMPT" \
  --pipeline.reverse_prompt "DESCRIPTION OF UNEDITED SCENE" \
  --pipeline.guidance_scale 5 \
  --pipeline.chunk_size 3 \
  --pipeline.langsam_obj 'OBJECT'  # optional, omit for environment editing
```

**View results:**
```bash
ns-viewer --load-config {outputs/.../config.yml}
```

**Render:**
```bash
# Dataset views
ns-gaussctrl-render dataset --load-config {outputs/.../config.yml} --output_path {render/NAME}
# Camera path video
ns-gaussctrl-render camera-path --load-config {outputs/.../config.yml} --camera-path-filename data/NAME/camera_paths/render-path.json --output_path render/NAME.mp4
```

## Architecture

The codebase is a NeRFStudio plugin. Entry point registered via `pyproject.toml` at `gaussctrl.gc_config:gaussctrl_method`.

**Execution flow** (`gc_trainer.py:GaussCtrlTrainer.setup`):
1. Load pretrained splatfacto checkpoint
2. `pipeline.render_reverse()` — render all training views, run DDIM inversion to get latent `z_0` per view; optionally run LangSAM to get object masks
3. `pipeline.edit_images()` — edit all views with ControlNet+depth using `CrossViewAttnProcessor`; edited images are written back to `datamanager.train_data[idx]["image"]`
4. Train 3DGS on edited images for `render_rate` (default 500) steps

**Key files:**
- `gc_pipeline.py` — `GaussCtrlPipeline`: orchestrates render→invert→edit→train loop; contains `render_reverse()`, `edit_images()`, `image2latent()`, `depth2disparity()`
- `utils.py` — `CrossViewAttnProcessor`: custom diffusers attention processor that blends self-attention with cross-view attention from 4 reference frames; `create_reprojection_mask()`: computes epipolar/reprojection masks from camera intrinsics/extrinsics; `compute_attn()`: applies optional epipolar mask to attention scores
- `gc_trainer.py` — `GaussCtrlTrainer`: extends NeRFStudio `Trainer`; custom `train()` loop runs only `render_rate` steps (not the full 30k splatfacto schedule)
- `gc_model.py` — `GaussCtrlModel`: extends `SplatfactoModel` with LPIPS + L1 losses
- `gc_datamanager.py` — `GaussCtrlDataManager`: subsamples training views (default: 40 views from 4 subsets of 10); stores per-view `depth_image`, `z_0_image`, `mask_image`, `unedited_image`, `image` in `self.train_data`
- `gc_dataset.py` — `GCDataset`: loads depth `.npy` files and latent `.npy` files alongside images
- `gc_config.py` — Assembles `MethodSpecification` with all optimizer configs; registers as `"gaussctrl"` method

**Cross-view attention** (`utils.py`):
- `CrossViewAttnProcessor` replaces standard self-attention in UNet; for each token, it attends to tokens from 4 fixed reference views
- In the `Epipolar` branch (current work): before each attention call, `set_camera_data()` stores camera matrices; inside `__call__`, depth maps are downsampled to match current UNet feature resolution, `create_reprojection_mask()` computes a `[B, N, N]` binary mask, and attention probabilities for target frames are zeroed out for non-corresponding reference pixels then re-normalized

**Data flow in `edit_images()`:**
- Reference views (4 by default) are prepended to each chunk batch
- Batch sent to pipe: `[ref_0..ref_3, target_0..target_k]`
- After diffusion, only `[self.num_ref_views:]` images are kept and stored back

**Important constants/defaults:**
- `diffusion_ckpt`: `stabilityai/stable-diffusion-xl-base-1.0` (SDXL, upgraded from SD1.4)
- `controlnet_ckpt`: `diffusers/controlnet-depth-sdxl-1.0`
- `diffusion_resolution`: 1024 (SDXL native; set to 768 to save VRAM)
- `ref_view_num`: 4 reference frames
- `chunk_size`: 1 (SDXL needs more VRAM than SD1.4); increase if GPU allows
- `subset_num=4`, `sampled_views_every_subset=10` → 40 total training views by default
- Debug edited images are saved to `/data/leuven/385/vsc38511/outputs/debug_edited_images/` (hardcoded path in `gc_pipeline.py`)
- VAE scale factor is read dynamically from `self.pipe.vae.config.scaling_factor` (0.13025 for SDXL)
- Render resolution is decoupled from diffusion resolution: renders at camera res, resized to `diffusion_resolution` for VAE/ControlNet, resized back after editing
- Prompts are pre-encoded through SDXL's dual text encoders once in `__init__` and reused via `_batch_prompt_embeds()`
- **Render cache**: SDXL latents are a different shape than SD1.4 — use a separate `cache_dir` when switching models
