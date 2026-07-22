<h1 align="center">[MultiSplat] — Multimodal Text and Image-Guided 3D Gaussian Splatting Scene Editing</h1>


<p align="center"><strong>🚧 Work in progress, this repository is under active development.</strong></p>



<p align="center">
  <img src="./assets/Multimodal vs Text-Only.png" alt="Multimodal vs text-only comparison" width="95%">
</p>

<p align="center"><sub>Same prompt, two conditioning modes. Adding a reference image locks the visual style down, and the result stays coherent across every view of the 3D scene.</sub></p>



## Overview

At a high level, consistent text and image-guided 3D editing in our method comes together from three parts:

- **Cross-view attention.** In the diffusion editor's UNet, each view's self-attention (`attn1`) is swapped for attention over a shared set of reference frames — every patch attends to the reference views, not just its own. That coupling is what keeps the separately-edited 2D views mutually consistent, so they fuse into a clean 3D scene instead of flickering.
- **Reference-image style control.** An IP-Adapter runs on the diffusion editor's cross-attention (`attn2`) while cross-view attention runs on `attn1` — the two mechanisms coexist without interference. A single reference image locks the target style.
- **Sequential reference-view editing.** Reference frames are edited one at a time — ref 0 sets the style; ref 1 attends to the already-edited ref 0; ref 2 attends to edited refs 0–1; and so on. Every subsequent target view then attends to a coherent set of edited references, giving markedly tighter multi-view consistency than editing all references in one shot.

Two operating modes ship out of the box:

- **Multimodal Mode** — you provide the reference image.
- **Self-Referential Mode** — no external image; the system scores its own intermediate edits with [ImageReward](https://github.com/THUDM/ImageReward), picks the strongest one, and reuses it as the IP-Adapter input. The scene bootstraps its own style anchor.

---

## Multimodal editing in action

Feed the system any reference image and a matching text prompt — the entire 3D scene is edited to match both.

<p align="center">
  <img src="./assets/Multimodal Results.png" alt="Multimodal editing examples: panda in the forest and bronze bust statue" width="95%">
</p>

<p align="center"><sub>A single reference image is enough to pin down species, material, and lighting the text prompt can only gesture at.</sub></p>

---

## Multi-view consistency

Sequential reference-view editing keeps the style locked across every rendered viewpoint.

<p align="center">
  <img src="./assets/Multimodal Experiments.png" alt="Picasso and Van Gogh style edits across six views" width="95%">
</p>

<p align="center">
  <img src="./assets/Multimodal Experiments2.png" alt="Jade horse and terracotta warrior style edits across six views" width="95%">
</p>

<p align="center"><sub>Six rendered viewpoints per row from the edited 3DGS scenes — Picasso, Van Gogh, jade horse, and terracotta warrior. All are 3D renders of the fine-tuned scene.</sub></p>

---

## How it works

<p align="center">
  <img src="./assets/pipeline.png" alt="Pipeline overview" width="95%">
</p>

1. **Render** RGB + depth for all training views from the pretrained 3DGS scene and DDIM-invert them to noisy latents.
2. **Sample references** — pick 4 reference views by **Farthest View Sampling** (greedy spatial + angular diversity). Wider geometric coverage feeds cross-view attention.
3. **Segment** (optional) — [LangSAM](https://github.com/luca-medeiros/lang-segment-anything) extracts an object mask per view so edits are composited over the original background.
4. **Sequentially edit references** with cross-view attention. Each new ref attends to the already-edited ones, DDIM-re-inverted between steps.
5. **Pick the IP-Adapter input** — *Self-Referential Mode*: score edited refs with ImageReward and use the best one. *Multimodal Mode*: use the user-provided image directly. LangSAM (optional) segments the chosen reference.
6. **Edit target views** in chunks with SD 1.5 + depth-ControlNet + IP-Adapter, attending to the edited reference latents on `attn1`.
7. **Fit** `splatfacto` on the edited views for 500 more steps.

---

## Installation

Reproduced on Linux (RHEL 9 / Ubuntu 22.04), Python 3.10, NVIDIA V100 (32 GB) / RTX A5000, driver ≥ 535. The environment pairs a conda **CUDA 11.8** toolkit — used only to provide `nvcc` for gsplat's runtime kernel compilation — with **PyTorch's own bundled CUDA 12.8** runtime. You do *not* need a matching system-wide CUDA install.

### Conda

```bash
git clone https://github.com/manolisfosteris/multisplat.git
cd multisplat

conda create -n multisplat python=3.10
conda activate multisplat

# nvcc + CUDA headers for gsplat's JIT kernel compilation (torch ships its own
# CUDA runtime). The full `cuda` metapackage hits a libcusparse file-clobber on
# recent conda, so install just the compile toolchain gsplat needs:
conda install cuda-nvcc cuda-cudart-dev cuda-cccl -c nvidia/label/cuda-11.8.0

# ffmpeg — required to encode rendered videos (ns-multisplat-render camera-path)
conda install ffmpeg -c conda-forge

# PyTorch — CUDA 12.8 wheels straight from PyPI (no extra index needed)
pip install torch==2.10.0 torchvision==0.25.0

# NeRFStudio + gsplat
pip install nerfstudio==1.0.0
pip install gsplat==0.1.3

# Lang-SAM — text-prompted segmentation (pulls in SAM-2 + the grounding model).
# Install BEFORE requirements.txt: lang-sam drags in a newer transformers /
# huggingface_hub that would break diffusers 0.26.0; requirements.txt pins them back.
pip install "git+https://github.com/luca-medeiros/lang-segment-anything.git@918043ed4666eea04da88aa179eb8d27ef4b1a1d"

# Diffusion + editing stack — pins transformers / huggingface_hub / diffusers / image-reward
pip install -r requirements.txt

# This project
pip install -e .
```

Verify: `ns-train -h` (you should see `multisplat` in the method list).

Notes:
- **No `tiny-cuda-nn`.** MultiSplat is `splatfacto`/`gsplat`-based; tiny-cuda-nn is only needed by NeRF-MLP methods (e.g. `nerfacto`) and is not a dependency here.
- **First-run compile.** gsplat 0.1.3 JIT-compiles its CUDA kernels on first use — that's why the conda CUDA 11.8 toolkit (`nvcc`) is installed. The first `ns-train multisplat` invocation will pause briefly the first time to build them.
- **lang-sam** is pinned to an exact commit for reproducibility; that commit's segmentation backend is SAM-2 (`sam2`), so `groundingdino` / `segment-anything` are *not* required.
- **`huggingface_hub` is pinned to `0.25.2`** (see `requirements.txt`). diffusers 0.26.0 imports `cached_download`, which huggingface_hub removed in 0.26.0 — a newer hub makes `import diffusers` crash. 0.25.2 is the last release that still exports it and remains compatible with `transformers==4.44.2`.

---

## Data

MultiSplat operates on scenes formatted the same way as NeRFStudio (`nerfstudio-data`). A scene folder needs:

```
data/my_scene/
  images/               # RGB training images
  transforms.json       # NeRFStudio camera format (COLMAP or manual)
  sparse_pc.ply         # optional but recommended: sparse point cloud from COLMAP
  camera_paths/         # optional: JSON files for render-path videos
```

The six upstream demo scenes (`bear`, `dinosaur`, `face`, `fangzhou`, `garden`, `stone_horse`) are included pre-processed under `data/`. To use your own scene, run COLMAP through `ns-process-data` or the equivalent.

### IP-Adapter reference images

For **Multimodal Mode** you also supply a single reference image that pins down the target style. A handful of the reference images used in our experiments ship under:

```
data/IP-Adapter Images/
  panda.webp
  dragon_with_wings.jpeg
  picasso.jpg
  terracota.jpeg
  van_gogh.jpg
```

Point `--pipeline.ip_adapter_image_path` at any of these (or your own image) — e.g. `"data/IP-Adapter Images/panda.webp"`.

---

## Usage

### 0 — Base 3DGS model (prerequisite)

Train a `splatfacto` model on your scene. This is a one-time cost per scene; the editor works from the pretrained checkpoint.

```bash
ns-train splatfacto --output-dir unedited_models --experiment-name my_scene nerfstudio-data --data data/my_scene
```

### Multimodal Mode — text prompt + reference image

```bash
ns-train multisplat --load-checkpoint unedited_models/my_scene/splatfacto/{TIMESTAMP}/nerfstudio_models/step-000029999.ckpt --experiment-name my_scene_edit --output-dir outputs --pipeline.datamanager.data data/my_scene --pipeline.edit_prompt "a photo of a bronze bust statue of a man" --pipeline.reverse_prompt "a photo of a man" --pipeline.guidance_scale 5 --pipeline.chunk_size 1 --pipeline.langsam_obj "man" --pipeline.ip_adapter_image_path "assets/ip_references/bronze_bust.jpeg" --pipeline.ip_adapter_scale 0.6
```

**Segmenting the reference image (optional).** Before the reference goes into the IP-Adapter, LangSAM can cut the subject out of its background so only the subject drives the style — controlled by `--pipeline.ip_langsam_obj`. It defaults to the scene's `langsam_obj`, so set it explicitly whenever the reference holds a different object than the scene (e.g. `--pipeline.ip_langsam_obj "panda"` when steering a bear scene with a panda reference), or pass `"none"` to skip it and use the whole image.

### Self-Referential Mode — no external image

```bash
ns-train multisplat --load-checkpoint unedited_models/bear/splatfacto/{TIMESTAMP}/nerfstudio_models/step-000029999.ckpt --experiment-name polar_bear --output-dir outputs/bear --pipeline.datamanager.data data/bear --pipeline.edit_prompt "a photo of a polar bear in the forest" --pipeline.reverse_prompt "a photo of a bear in the forest" --pipeline.guidance_scale 5 --pipeline.chunk_size 1 --pipeline.langsam_obj "bear" --pipeline.auto_ip_from_refs True --pipeline.ip_adapter_scale 0.6
```

### Render the edited scene

```bash
# Dataset viewpoints
ns-multisplat-render dataset --load-config outputs/.../config.yml --output_path render/my_scene_edit

# Camera-path video
ns-multisplat-render camera-path --load-config outputs/.../config.yml --camera-path-filename data/my_scene/camera_paths/render-path.json --output_path render/my_scene_edit.mp4
```

### Evaluate

```bash
python metrics/evaluate.py --edited_dir render/my_scene_edit/rgb --original_dir render/original_my_scene/rgb --edit_prompt "..." --reverse_prompt "..."
```

Reports `clip_score` (edited↔prompt), `clip_dir` (directional CLIP similarity), and `clip_img` (edited↔original image similarity).

### Key CLI flags

| Flag | Meaning |
|---|---|
| `--pipeline.ip_adapter_image_path` | Reference image for IP-Adapter |
| `--pipeline.ip_adapter_scale`      | IP-Adapter influence (0 = text only, 1 = image only) |
| `--pipeline.auto_ip_from_refs`     | Auto-pick best edited ref as IP-Adapter input (Self-Referential Mode) |
| `--pipeline.langsam_obj`           | LangSAM target for the scene object |
| `--pipeline.ip_langsam_obj`        | LangSAM target for the reference image (`"none"` disables) |
| `--pipeline.ref_view_selection`    | `"fvs"` (default) or `"random"` |
| `--pipeline.fvs_alpha`             | FVS angular-distance weight (default 1.0) |


## Repository tour

| File | Role |
|---|---|
| `multisplat/pipeline.py`    | Orchestrates render → invert → edit → train. Hosts `select_reference_views_fvs`, `_load_ip_adapter`, `_auto_select_ip_from_refs`, `_build_combined_attn_procs`, `edit_reference_views_sequential`, `edit_images`. |
| `multisplat/utils.py`       | `CrossViewAttnProcessor` — custom diffusers attention processor that replaces UNet `attn1` self-attention with cross-view attention over the reference frames. |
| `multisplat/trainer.py`     | Extends NeRFStudio's `Trainer` with a short 500-step fine-tuning loop after editing. |
| `multisplat/datamanager.py` | Subsamples 40 training views; stores per-view depth / z₀ latent / mask / unedited / edited image. |
| `multisplat/model.py`       | `MultiSplatModel` — extends `SplatfactoModel` with LPIPS + L1 losses. |
| `multisplat/config.py`      | Registers `"multisplat"` as a NeRFStudio method. |
| `metrics/evaluate.py`       | CLIP evaluation (`clip_score`, `clip_dir`, `clip_img`). |

---

---

## Acknowledgments

MultiSplat extends the work of [GaussCtrl](https://github.com/ActiveVisionLab/gaussctrl) (Wu et al., ECCV 2024).

```bibtex
@inproceedings{wu2024gaussctrl,
  title     = {{GaussCtrl}: Multi-View Consistent Text-Driven {3D} {G}aussian Splatting Editing},
  author    = {Wu, Jing and Bian, Jia-Wang and Li, Xinghui and Wang, Guangrun and Reid, Ian and Torr, Philip and Prisacariu, Victor Adrian},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2024}
}
```

Additional third-party components:

- **NeRFStudio** — training loop, camera / dataparser abstractions ([Tancik et al., SIGGRAPH 2023](https://docs.nerf.studio/)).
- **gsplat** — differentiable Gaussian rasterization ([Ye et al., 2024](https://github.com/nerfstudio-project/gsplat)).
- **IP-Adapter** — image-prompt cross-attention for Stable Diffusion ([Ye et al., 2023](https://github.com/tencent-ailab/IP-Adapter)).
- **ImageReward** — self-referential reward scoring for reference selection ([Xu et al., NeurIPS 2023](https://github.com/THUDM/ImageReward)).
- **NeRF Director** — Farthest View Sampling algorithm ([Xiao et al., CVPR 2024](https://arxiv.org/abs/2406.08839)).
- **lang-segment-anything** — text-prompted masking of scene and reference images ([Medeiros](https://github.com/luca-medeiros/lang-segment-anything)).

---

## License

_TODO — license to be added._
