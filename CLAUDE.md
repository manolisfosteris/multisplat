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
conda create -n multisplat python=3.10
conda activate multisplat
conda install cuda-nvcc cuda-cudart-dev cuda-cccl -c nvidia/label/cuda-11.8.0  # nvcc for gsplat JIT; torch ships its own CUDA runtime
conda install ffmpeg -c conda-forge   # video encoding for ns-multisplat-render camera-path
pip install torch==2.10.0 torchvision==0.25.0
pip install nerfstudio==1.0.0
pip install gsplat==0.1.3
pip install "git+https://github.com/luca-medeiros/lang-segment-anything.git@918043ed4666eea04da88aa179eb8d27ef4b1a1d"  # BEFORE requirements.txt
pip install -r requirements.txt   # pins transformers/huggingface_hub/diffusers back down
pip install -e .
```

Gotchas (all learned from a clean rebuild):
- No `tiny-cuda-nn` (splatfacto/gsplat-based); `groundingdino`/`segment-anything` not needed (pinned lang-sam uses SAM-2).
- Full `cuda` metapackage clobbers on recent conda → install the minimal `cuda-nvcc cuda-cudart-dev cuda-cccl`.
- lang-sam must precede `requirements.txt` (it pulls a newer transformers/huggingface_hub that breaks diffusers 0.26.0).
- `huggingface_hub==0.25.2` (in requirements.txt): diffusers 0.26.0 needs the `cached_download` symbol removed in hub 0.26.

Verify: `ns-train -h` — `multisplat` must appear in the method list.

## Key Commands

**Train a base 3DGS model (prerequisite):**
```bash
ns-train splatfacto --output-dir unedited_models --experiment-name my_scene nerfstudio-data --data data/my_scene
```

**MultiSplat editing (full pipeline):**
```bash
ns-train multisplat --load-checkpoint {path/to/step-000029999.ckpt} --experiment-name EXPERIMENT --output-dir outputs --pipeline.datamanager.data {path/to/data} --pipeline.edit_prompt "YOUR EDIT PROMPT" --pipeline.reverse_prompt "DESCRIPTION OF UNEDITED SCENE" --pipeline.guidance_scale 5 --pipeline.chunk_size 1 --pipeline.langsam_obj 'OBJECT' --pipeline.ip_adapter_image_path {path/to/ref.webp} --pipeline.ip_adapter_scale 0.6
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
- Debug images: written to `outputs/debug_edited_images/{experiment_name}/` **only in Self-Referential Mode** (`auto_ip_from_refs`), because `_auto_select_ip_from_refs` reads the edited refs back from disk to score them with ImageReward. The folder is deleted at the end of `edit_images`. Multimodal Mode writes nothing.
