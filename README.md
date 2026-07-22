<h1 align="center">MultiSplat</h1>

<p align="center">
  <em>Multimodal text- and image-guided 3D Gaussian Splatting scene editing — multi-view consistent, no per-frame flicker.</em>
</p>

<p align="center">
  <a href="#citation">Thesis PDF (TODO)</a> ·
  <a href="https://github.com/manolisfosteris/multisplat/tree/Seq_Ref_IP_Adapter">Seq_Ref_IP_Adapter branch</a> ·
  <a href="#license">License</a>
</p>

<p align="center">
  <img src="./assets/Multimodal vs Text-Only.png" alt="Multimodal vs text-only comparison" width="95%">
</p>

<p align="center"><sub>Same prompt, two conditioning modes. Adding a reference image locks the visual style down, and the result stays coherent across every view of the 3D scene.</sub></p>

---

## Abstract

Text-driven 3D Gaussian Splatting (3DGS) editors are powerful but ambiguous: a single prompt like *"a polar bear in a forest"* can describe wildly different-looking scenes, and multi-view consistency degrades quickly when reference views drift from each other. **MultiSplat** couples an IP-Adapter for style control with a *sequential* reference-view editing loop on top of the GaussCtrl framework. A single reference image (or an automatically self-selected one) pins down the visual style; reference frames are edited one at a time, each attending to the already-edited earlier ones, so the target views see a coherent set of references and multi-view consistency stays tight. Reference views are chosen with Farthest View Sampling for wider geometric coverage. All contributions run on the standard Stable Diffusion 1.5 + depth-ControlNet + `splatfacto` stack — no retraining of the diffusion model, no auxiliary networks beyond the off-the-shelf IP-Adapter and ImageReward scorer.

---

## Overview

Two ideas make text- and image-guided 3D editing practical:

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
7. **Fit** `splatfacto` on the edited views for 500 more steps with L1 + LPIPS.

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

Point `--pipeline.ip_adapter_image_path` at any of these (or your own image) — e.g. `"data/IP-Adapter Images/panda.webp"`. These are just examples of the kind of style anchors the IP-Adapter accepts; any RGB image works. (Mind the space in the folder name — quote the path.)

---

## Usage

### STEP 0 — Train the base 3DGS model (prerequisite)

MultiSplat does not build a 3D scene from scratch — it *edits* one that already exists. So before any editing you need a pretrained 3D Gaussian Splatting model of your scene. This step trains that model with NeRFStudio's stock `splatfacto` method (plain Gaussian Splatting, no editing). It's a **one-time cost per scene**: once you have the checkpoint, you can run as many edits on it as you like.

```bash
ns-train splatfacto --output-dir unedited_models --experiment-name my_scene nerfstudio-data --data data/my_scene
```

**What this command does:** it optimizes a set of 3D Gaussians to reproduce your input photographs, for `splatfacto`'s default 30,000 steps, and periodically saves checkpoints. When it finishes you have a `.ckpt` file that the MultiSplat editor loads in the next step. (On the very first run in a fresh env, the first step pauses a minute or two to JIT-compile gsplat's CUDA kernels — this is normal and only happens once.)

**The flags, one by one:**

| Part | What it does |
|---|---|
| `ns-train splatfacto` | The NeRFStudio training entry point, told to use the `splatfacto` method (vanilla 3D Gaussian Splatting). This produces the *unedited* scene. |
| `--output-dir unedited_models` | Root folder for results, **created in your current working directory** if it doesn't exist. Everything from this run is written underneath it. |
| `--experiment-name my_scene` | Names the run. It becomes a **subfolder** inside the output dir, so results land in `unedited_models/my_scene/…`. Use a name that identifies the scene (e.g. `bear`). |
| `nerfstudio-data --data data/my_scene` | Selects the **dataparser** (`nerfstudio-data`) and points it at your scene folder via its positional `--data` argument. This is where the training images and camera poses are read from. |

> **Order matters:** `--data` belongs to the `nerfstudio-data` dataparser, so it must come *after* the word `nerfstudio-data`, not before it. Flags before `nerfstudio-data` configure the trainer; the argument after it configures the dataparser.

**What kind of data `--data` expects:** a scene folder in **NeRFStudio format** — RGB training images plus the camera pose for each image (intrinsics + extrinsics) in a `transforms.json`, optionally a sparse point cloud to initialize the Gaussians:

```
data/my_scene/
  images/           # the RGB photos of the scene, taken from many viewpoints
  transforms.json   # camera intrinsics + per-image extrinsic pose (COLMAP or manual)
  sparse_pc.ply     # optional: sparse point cloud (COLMAP) used to seed the Gaussians
```

The six demo scenes shipped under `data/` (`bear`, `dinosaur`, `face`, `fangzhou`, `garden`, `stone_horse`) are already in this format — just point `--data` at one of them. For your own footage, generate this layout by running COLMAP through `ns-process-data` (see the [Data](#data) section above).

**Where the checkpoint ends up:** NeRFStudio inserts a timestamp folder for each run, so the final checkpoint the editor needs is:

```
unedited_models/my_scene/splatfacto/{TIMESTAMP}/nerfstudio_models/step-000029999.ckpt
```

Copy that exact path — it's what you pass to `--load-checkpoint` in the editing steps below.

### Multimodal Mode — text prompt + reference image

```bash
ns-train multisplat --load-checkpoint unedited_models/my_scene/splatfacto/{TIMESTAMP}/nerfstudio_models/step-000029999.ckpt --experiment-name my_scene_edit --output-dir outputs --pipeline.datamanager.data data/my_scene --pipeline.edit_prompt "a photo of a bronze bust statue of a man" --pipeline.reverse_prompt "a photo of a man" --pipeline.guidance_scale 5 --pipeline.chunk_size 1 --pipeline.langsam_obj "man" --pipeline.ip_adapter_image_path "assets/ip_references/bronze_bust.jpeg" --pipeline.ip_adapter_scale 0.6
```

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
| `--pipeline.debug_dir`             | Where per-view edited images are written (default `outputs/debug_edited_images`) |

---

## Results

CLIP-based metrics on the standard GaussCtrl demo scenes. Baseline is the upstream GaussCtrl editor at the same random seed; `↑` means higher is better.

| Scene | Edit prompt | Method | clip_score ↑ | clip_dir ↑ | clip_img ↑ |
|---|---|---|---|---|---|
| bear | polar bear in the forest | GaussCtrl                     | TODO | TODO | TODO |
| bear | polar bear in the forest | MultiSplat (Multimodal)       | TODO | TODO | TODO |
| bear | polar bear in the forest | MultiSplat (Self-Referential) | TODO | TODO | TODO |
| face | bronze bust of a man     | GaussCtrl                     | TODO | TODO | TODO |
| face | bronze bust of a man     | MultiSplat (Multimodal)       | TODO | TODO | TODO |

Ablations on the contribution of each component (sequential-ref editing, IP-Adapter, FVS, cross-view attention) are reported in the thesis (TODO: link).

---

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

## Citation

If you use MultiSplat, please cite:

```bibtex
@mastersthesis{fosteris2026multisplat,
  title  = {MultiSplat: Multimodal Text and Image-Guided 3D Gaussian Splatting Editing},
  author = {Fosteris, Manolis},
  school = {TODO},
  year   = {2026},
  note   = {TODO: thesis URL}
}
```

---

## Acknowledgments

MultiSplat builds directly on [GaussCtrl](https://github.com/ActiveVisionLab/gaussctrl) (Wu et al., ECCV 2024) — the render→invert→edit→retrain pipeline, the cross-view attention formulation, and the demo scenes come from that work.

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

BSD 3-Clause — see [`LICENSE.txt`](LICENSE.txt). Original GaussCtrl copyright is preserved verbatim; MultiSplat additions carry a separate copyright line.
