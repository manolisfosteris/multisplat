# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

MultiSplat is a text- and image-guided 3D Gaussian Splatting editing method that extends [GaussCtrl](https://github.com/ActiveVisionLab/gaussctrl) (ECCV 2024). Given a pretrained `splatfacto` scene, it (1) renders all views and DDIM-inverts them to noisy latents, (2) edits reference views one-by-one with SD 1.5 + depth-ControlNet + cross-view attention + IP-Adapter, then edits every target view attending to the edited references, and (3) fine-tunes the 3DGS model on the edited images.

Two operating modes:
- **Multimodal Mode** — user provides a reference image (`--pipeline.ip_adapter_image_path`).
- **Self-Referential Mode** — no external image; `--pipeline.auto_ip_from_refs True` scores the edited references with ImageReward and reuses the best one as the IP-Adapter input.

## Branches
- `main` — canonical showcase branch. Rename commits from GaussCtrl → MultiSplat live here.
- `Seq_Ref_IP_Adapter` — full-fat variant with render cache (`--pipeline.cache_dir`) and standalone diffusion/retrain scripts (`scripts/edit_from_cache.py`, `scripts/retrain_from_edited.py`).

## Installation

See `README.md` for the full recipe. Short version:

```bash
conda create -n multisplat python=3.8
conda activate multisplat
conda install cuda -c nvidia/label/cuda-11.8.0
pip install nerfstudio==1.0.0
pip install gsplat==0.1.3
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git
pip install -r requirements.txt
pip install -e .
```

Verify: `ns-train -h` — `multisplat` must appear in the method list.

## Key Commands

**Train a base 3DGS model (prerequisite):**
```bash
ns-train splatfacto --output-dir unedited_models --experiment-name my_scene nerfstudio-data --data data/my_scene
```

**MultiSplat editing (full pipeline):**
```bash
ns-train multisplat --load-checkpoint {path/to/step-000029999.ckpt} --experiment-name EXPERIMENT --output-dir outputs --pipeline.datamanager.data {path/to/data} --pipeline.edit_prompt "YOUR EDIT PROMPT" --pipeline.reverse_prompt "DESCRIPTION OF UNEDITED SCENE" --pipeline.guidance_scale 5 --pipeline.chunk_size 1 --pipeline.langsam_obj 'OBJECT' --pipeline.cache_dir {path/to/cache} --pipeline.ip_adapter_image_path {path/to/ref.webp} --pipeline.ip_adapter_scale 0.6
```

**Run diffusion editing only (no 3DGS, P100-compatible):**
```bash
python scripts/edit_from_cache.py --cache_dir {path/to/cache} --output_dir {path/to/output} --edit_prompt "YOUR EDIT PROMPT" --ip_adapter_image_path {path/to/ref.webp} --ip_adapter_scale 0.6 --langsam_obj OBJECT --guidance_scale 5 --chunk_size 1
```

**Retrain 3DGS from pre-edited images (V100 required):**
```bash
python scripts/retrain_from_edited.py --load_checkpoint {path/to/step-000029999.ckpt} --edited_images_dir {path/to/edited/images} --data {path/to/data} --cache_dir {path/to/cache} --experiment_name EXPERIMENT --output_dir {path/to/output}
```

**Render:**
```bash
ns-multisplat-render dataset --load-config {outputs/.../config.yml} --output_path {render/NAME}
ns-multisplat-render camera-path --load-config {outputs/.../config.yml} --camera-path-filename data/NAME/camera_paths/render-path.json --output_path {NAME}.mp4
```

**Evaluate:**
```bash
python metrics/evaluate.py --edited_dir {render/NAME/rgb} --original_dir {render/original_SCENE/rgb} --edit_prompt "YOUR EDIT PROMPT" --reverse_prompt "ORIGINAL DESCRIPTION"
```

## Architecture

NeRFStudio plugin. Entry point registered via `pyproject.toml` at `multisplat.config:multisplat_method`.

**Execution flow** (`trainer.py:MultiSplatTrainer.setup`):
1. Load pretrained splatfacto checkpoint
2. `pipeline.render_reverse()` — render all training views, DDIM-invert to latent `z_0` per view; optionally run LangSAM for object masks
3. `pipeline.edit_images()` — edit all views with ControlNet+depth+IP-Adapter+CrossViewAttn; edited images written back to `datamanager.train_data[idx]["image"]`
4. Train 3DGS on edited images for `render_rate` (default 500) steps

**Key files:**
- `multisplat/pipeline.py` — `MultiSplatPipeline`: orchestrates render→invert→edit→train; hosts `render_reverse()`, `edit_images()`, `edit_reference_views_sequential()`, `_load_ip_adapter()`, `_auto_select_ip_from_refs()`, `_build_combined_attn_procs()`, `select_reference_views_fvs()`
- `multisplat/utils.py` — `CrossViewAttnProcessor`: diffusers attention processor that replaces UNet `attn1` self-attention with cross-view attention over reference frames
- `multisplat/trainer.py` — `MultiSplatTrainer`: extends NeRFStudio `Trainer`; short 500-step fine-tune after editing
- `multisplat/model.py` — `MultiSplatModel`: extends `SplatfactoModel` with LPIPS + L1 losses
- `multisplat/datamanager.py` — `MultiSplatDataManager`: subsamples 40 training views; stores per-view `depth_image`, `z_0_image`, `mask_image`, `unedited_image`, `image`
- `multisplat/dataset.py` — `MultiSplatDataset`: loads depth `.npy` and latent `.npy` files alongside images
- `multisplat/config.py` — Assembles `MethodSpecification`; registers the `"multisplat"` method
- `scripts/edit_from_cache.py` — standalone diffusion editing script (no 3DGS, works on P100)
- `scripts/retrain_from_edited.py` — standalone 3DGS retraining script from pre-edited images (requires V100)
- `metrics/evaluate.py` — CLIP evaluation script (`clip_score`, `clip_dir`, `clip_img`)

**Cross-view attention** (`utils.py`):
- `CrossViewAttnProcessor` replaces `attn1` (self-attention); each token attends to tokens from reference views instead of only its own view
- Critical for multi-view consistency — removing it causes near-zero `clip_dir` and poor 3DGS reconstruction
- IP-Adapter operates on `attn2` (cross-attention) independently; `_build_combined_attn_procs()` merges both without interference

**IP-Adapter integration:**
- Loaded via `_load_ip_adapter()` at the start of `edit_images()` (deferred from `__init__` so DDIM inversion runs clean)
- Reference image optionally segmented with LangSAM before use (see `--pipeline.ip_langsam_obj`)
- `_build_combined_attn_procs()`: `attn1` → `CrossViewAttnProcessor`, `attn2` → IP-Adapter processors

**Sequential reference view editing** (`edit_reference_views_sequential()`):
- Reference views are edited one-by-one: `ref_0` attends to all original ref latents, subsequent refs attend to already-edited ones
- Produces edited ref latents that target views then attend to via cross-view attention

**Data flow in `edit_images()`:**
- Reference views (4 by default, selected via FVS) are prepended to each chunk batch
- Batch sent to pipe: `[ref_0..ref_3, target_0..target_k]`
- After diffusion, only `[num_ref_views:]` images are kept and stored back

**Important defaults:**
- `diffusion_ckpt`: `runwayml/stable-diffusion-v1-5`
- `controlnet_ckpt`: `lllyasviel/sd-controlnet-depth`
- `ref_view_num`: 4 reference frames (FVS by default)
- `chunk_size`: 1
- 40 total training views by default
- Debug images: `{--pipeline.debug_dir}/{experiment_name}/` (default `outputs/debug_edited_images/{experiment_name}/`)
- **Render cache** (this branch only): `--pipeline.cache_dir` saves per-view `depth`, `z_0`, `rgb`, `mask` `.npy` files as `{idx:04d}_{type}.npy` so subsequent runs skip DDIM inversion
