# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GaussCtrl is a text-driven 3D Gaussian Splatting editing method (ECCV 2024). It edits 3DGS scenes by: (1) rendering all views and inverting them to DDIM latents, (2) editing images using ControlNet+depth with a cross-view attention mechanism (`CrossViewAttnProcessor`) for multi-view consistency, and (3) fine-tuning the 3DGS model on the edited images.

This is a fork with active modifications. The main active branch is `Seq_Ref_IP_Adapter`, which adds:
- Sequential reference view editing for better multi-view consistency
- IP-Adapter support for style guidance from a reference image
- FVS (Farthest View Sampling) reference view selection
- Render cache to skip re-rendering on subsequent runs

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

**Run GaussCtrl editing (full pipeline):**
```bash
ns-train gaussctrl --load-checkpoint {path/to/step-000029999.ckpt} --experiment-name EXPERIMENT_NAME --output-dir outputs --pipeline.datamanager.data {path/to/data} --pipeline.edit_prompt "YOUR EDIT PROMPT" --pipeline.reverse_prompt "DESCRIPTION OF UNEDITED SCENE" --pipeline.guidance_scale 5 --pipeline.chunk_size 1 --pipeline.langsam_obj 'OBJECT' --pipeline.cache_dir {path/to/cache} --pipeline.ip_adapter_image_path {path/to/ref.webp} --pipeline.ip_adapter_scale 0.6
```

**Run diffusion editing only (no 3DGS, P100-compatible):**
```bash
python scripts/edit_from_cache.py --cache_dir {path/to/cache} --output_dir {path/to/output} --edit_prompt "YOUR EDIT PROMPT" --ip_adapter_image_path {path/to/ref.webp} --ip_adapter_scale 0.6 --langsam_obj OBJECT --guidance_scale 5 --chunk_size 1
```

**Retrain 3DGS from pre-edited images (V100 required):**
```bash
python scripts/retrain_from_edited.py --load_checkpoint {path/to/step-000029999.ckpt} --edited_images_dir {path/to/edited/images} --data {path/to/data} --cache_dir {path/to/cache} --experiment_name EXPERIMENT_NAME --output_dir {path/to/output}
```

**Render:**
```bash
# Dataset views
ns-gaussctrl-render dataset --load-config {outputs/.../config.yml} --output_path {render/NAME}
# Camera path video (all videos saved to /data/leuven/385/vsc38511/outputs/VIDEOS/)
ns-gaussctrl-render camera-path --load-config {outputs/.../config.yml} --camera-path-filename data/NAME/camera_paths/render-path.json --output_path /data/leuven/385/vsc38511/outputs/VIDEOS/NAME.mp4
```

**Evaluate:**
```bash
python metrics/evaluate.py --edited_dir {render/NAME/rgb} --original_dir {render/original_SCENE/rgb} --edit_prompt "YOUR EDIT PROMPT" --reverse_prompt "ORIGINAL DESCRIPTION"
```

## Architecture

The codebase is a NeRFStudio plugin. Entry point registered via `pyproject.toml` at `gaussctrl.gc_config:gaussctrl_method`.

**Execution flow** (`gc_trainer.py:GaussCtrlTrainer.setup`):
1. Load pretrained splatfacto checkpoint
2. `pipeline.render_reverse()` — render all training views, run DDIM inversion to get latent `z_0` per view; optionally run LangSAM to get object masks; saves/loads from `cache_dir` if set
3. `pipeline.edit_images()` — edit all views with ControlNet+depth; edited images written back to `datamanager.train_data[idx]["image"]`
4. Train 3DGS on edited images for `render_rate` (default 500) steps

**Key files:**
- `gc_pipeline.py` — `GaussCtrlPipeline`: orchestrates render→invert→edit→train loop; contains `render_reverse()`, `edit_images()`, `edit_reference_views_sequential()`, `_load_ip_adapter()`, `_build_combined_attn_procs()`
- `utils.py` — `CrossViewAttnProcessor`: custom diffusers attention processor that replaces UNet `attn1` (self-attention) with cross-view attention from reference frames
- `gc_trainer.py` — `GaussCtrlTrainer`: extends NeRFStudio `Trainer`; custom `train()` loop runs only `render_rate` steps
- `gc_model.py` — `GaussCtrlModel`: extends `SplatfactoModel` with LPIPS + L1 losses
- `gc_datamanager.py` — `GaussCtrlDataManager`: subsamples training views (default: 40 views); stores per-view `depth_image`, `z_0_image`, `mask_image`, `unedited_image`, `image` in `self.train_data`
- `gc_dataset.py` — `GCDataset`: loads depth `.npy` and latent `.npy` files alongside images
- `gc_config.py` — Assembles `MethodSpecification` with all optimizer configs; registers as `"gaussctrl"` method
- `scripts/edit_from_cache.py` — standalone diffusion editing script (no 3DGS, works on P100)
- `scripts/retrain_from_edited.py` — standalone 3DGS retraining script from pre-edited images (requires V100)
- `metrics/evaluate.py` — CLIP evaluation script (clip_score, clip_dir, clip_img)

**Cross-view attention** (`utils.py`):
- `CrossViewAttnProcessor` replaces `attn1` (self-attention) in the UNet; each token attends to tokens from reference views instead of only its own view
- Critical for multi-view consistency — experiments confirmed that removing it causes near-zero `clip_dir` and poor 3DGS reconstruction
- IP-Adapter operates on `attn2` (cross-attention) independently; `_build_combined_attn_procs()` merges both without interference

**IP-Adapter integration:**
- Loaded via `_load_ip_adapter()` at the start of `edit_images()` (deferred from `__init__`)
- Reference image optionally segmented with LangSAM before use
- `_build_combined_attn_procs()`: `attn1` → `CrossViewAttnProcessor`, `attn2` → IP-Adapter processors
- Controlled by `--pipeline.ip_adapter_image_path` and `--pipeline.ip_adapter_scale`

**Sequential reference view editing** (`edit_reference_views_sequential()`):
- Reference views are edited one-by-one: `ref_0` attends to all original ref latents, subsequent refs attend to already-edited ones
- Produces edited ref latents that target views then attend to via cross-view attention

**Data flow in `edit_images()`:**
- Reference views (4 by default, selected via FVS) are prepended to each chunk batch
- Batch sent to pipe: `[ref_0..ref_3, target_0..target_k]`
- After diffusion, only `[self.num_ref_views:]` images are kept and stored back

**Important constants/defaults:**
- `diffusion_ckpt`: `runwayml/stable-diffusion-v1-5` (SD 1.5)
- `controlnet_ckpt`: `lllyasviel/sd-controlnet-depth`
- `ref_view_num`: 4 reference frames (selected via FVS by default)
- `chunk_size`: 1
- 40 total training views by default
- Debug edited images saved to `/data/leuven/385/vsc38511/outputs/debug_edited_images/{experiment_name}/`
- **Render cache**: cache `.npy` files (depth, z0, rgb, mask) saved per-view as `{idx:04d}_{type}.npy`

## Paths (HPC)
- Unedited models: `/data/leuven/385/vsc38511/unedited_models/{scene}/`
- Render cache — bear: `/data/leuven/385/vsc38511/outputs/cache/stable_diffusion_1.5/`
- Render cache — face: `/data/leuven/385/vsc38511/cache/stable_diffusion_1.5/face/`
- Edited outputs: `/data/leuven/385/vsc38511/outputs/{scene}/{experiment_name}/`
- Debug edited images: `/data/leuven/385/vsc38511/outputs/debug_edited_images/{experiment_name}/`
- Rendered frames: `/data/leuven/385/vsc38511/render/{experiment_name}/rgb/`
- Original bear renders: `/data/leuven/385/vsc38511/render/original_bear/rgb/`
- Original face renders: `/data/leuven/385/vsc38511/render/original_face/rgb/`
- Videos: `/data/leuven/385/vsc38511/outputs/VIDEOS/`
- Results & experiments: `/data/leuven/385/vsc38511/Results/` (outside git repo, shared across branches)
- HuggingFace cache: `/scratch/leuven/385/vsc38511/.cache/huggingface/` (set `HF_HOME` to this)
