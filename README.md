<h1 align="center">GaussCtrl-SeqRef-IPA</h1>
<h3 align="center">Multimodal Text- and Image-Guided 3D Gaussian Splatting Editing</h3>

<p align="center">
  <em>Edit any 3D Gaussian Splatting scene with a text prompt <strong>and</strong> a reference image — multi-view consistent, no per-frame flicker.</em>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2403.08733"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv"></a>
  <a href="LICENSE.txt"><img alt="License" src="https://img.shields.io/badge/License-BSD-green"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8-blue?logo=python">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1.2-orange?logo=pytorch">
</p>

<p align="center">
  <img src="./assets/Multimodal vs Text-Only.png" alt="Multimodal vs text-only comparison" width="95%">
</p>

<p align="center"><sub>Same prompt, two conditioning modes. Text alone drifts — <em>"bronze bust statue"</em> paints a golden-yellow blur. Adding a reference image locks the visual style down, and the result stays coherent across every view of the 3D scene.</sub></p>

---

## Overview

3D scene editing with a text prompt is powerful but often ambiguous: the same words describe wildly different-looking objects. This project extends the GaussCtrl 3DGS editor with two ideas that together make text- and image-guided 3D editing practical:

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

Sequential reference-view editing keeps the style locked across every rendered viewpoint — no flickering identity between frames.

<p align="center">
  <img src="./assets/Multimodal Experiments.png" alt="Picasso and Van Gogh style edits across six views" width="95%">
</p>

<p align="center">
  <img src="./assets/Multimodal Experiments2.png" alt="Jade horse and terracotta warrior style edits across six views" width="95%">
</p>

<p align="center"><sub>Six rendered viewpoints per row from the edited 3DGS scenes — Picasso, Van Gogh, jade horse, and terracotta warrior. All are 3D renders of the fine-tuned scene, not 2D diffusion output.</sub></p>

---

## How it works

<p align="center">
  <img src="./assets/pipeline.png" alt="Pipeline overview" width="95%">
</p>

1. **Render** RGB + depth for all training views from the pretrained 3DGS scene and DDIM-invert them to noisy latents.
2. **Sample references** — pick 4 reference views by **Farthest View Sampling** (greedy spatial + angular diversity, from *NeRF Director*, Xiao et al., CVPR 2024) rather than uniform-random sampling. Wider geometric coverage feeds cross-view attention.
3. **Segment** (optional) — [LangSAM](https://github.com/luca-medeiros/lang-segment-anything) extracts an object mask per view so edits are composited over the original background.
4. **Sequentially edit references** with cross-view attention. Each new ref attends to the already-edited ones, DDIM-re-inverted between steps.
5. **Pick the IP-Adapter input** — *Self-Referential Mode*: score edited refs with ImageReward and use the best one. *Multimodal Mode*: use the user-provided image directly. LangSAM (optional) segments the chosen reference.
6. **Edit target views** in chunks with SD 1.5 + depth-ControlNet + IP-Adapter, attending to the edited reference latents on `attn1`.
7. **Fit** `splatfacto` on the edited views for 500 more steps with L1 + LPIPS.

---

## Installation

Tested on CUDA 11.8 + Ubuntu 22.04 + NeRFStudio 1.0.0 + NVIDIA V100 / RTX A5000 (24 GB).

### Conda

```bash
git clone https://github.com/manolisfosteris/gaussctrl.git
cd gaussctrl

conda create -n gaussctrl python=3.8
conda activate gaussctrl
conda install cuda -c nvidia/label/cuda-11.8.0

# Torch + tiny-cuda-nn
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# NeRFStudio + gsplat
pip install nerfstudio==1.0.0
pip install gsplat==0.1.3
pip install -r requirements.txt
pip install "huggingface_hub<0.24"

# Lang-SAM (specific branch, no-deps)
cd ..
git clone https://github.com/luca-medeiros/lang-segment-anything && cd lang-segment-anything
git checkout fix-no_detection
pip install --no-deps --no-build-isolation .
pip install groundingdino-py segment-anything

# This project
cd ../gaussctrl
pip install -e .
```

Verify: `ns-train -h`

If `tiny-cuda-nn` fails to build, see the [official build-from-source notes](https://github.com/NVlabs/tiny-cuda-nn/?tab=readme-ov-file#compilation-windows--linux). NeRFStudio v1.0.0 with gsplat v0.1.3 is the recommended pairing.

### Docker (fallback)

```bash
docker pull jingwu2121/gaussctrl:latest
docker run -it --gpus all --shm-size=16g -p 7007:7007 -v /path/to/gaussctrl:/workspace/gaussctrl jingwu2121/gaussctrl:latest /bin/bash
```

Inside the container:

```bash
conda activate gaussctrl
cd /workspace/gaussctrl
```

---

## Usage

### 0 — Base 3DGS model (prerequisite)

Train a `splatfacto` model on your scene:

```bash
ns-train splatfacto --output-dir unedited_models --experiment-name my_scene nerfstudio-data --data data/my_scene
```

### Multimodal Mode — text prompt + reference image

```bash
ns-train gaussctrl --load-checkpoint unedited_models/my_scene/splatfacto/{TIMESTAMP}/nerfstudio_models/step-000029999.ckpt --experiment-name my_scene_edit --output-dir outputs --pipeline.datamanager.data data/my_scene --pipeline.edit_prompt "a photo of a bronze bust statue of a man" --pipeline.reverse_prompt "a photo of a man" --pipeline.guidance_scale 5 --pipeline.chunk_size 1 --pipeline.langsam_obj "man" --pipeline.ip_adapter_image_path "assets/ip_references/bronze_bust.jpeg" --pipeline.ip_adapter_scale 0.6
```

### Self-Referential Mode — no external image

```bash
ns-train gaussctrl --load-checkpoint unedited_models/bear/splatfacto/{TIMESTAMP}/nerfstudio_models/step-000029999.ckpt --experiment-name polar_bear --output-dir outputs/bear --pipeline.datamanager.data data/bear --pipeline.edit_prompt "a photo of a polar bear in the forest" --pipeline.reverse_prompt "a photo of a bear in the forest" --pipeline.guidance_scale 5 --pipeline.chunk_size 1 --pipeline.langsam_obj "bear" --pipeline.auto_ip_from_refs True --pipeline.ip_adapter_scale 0.6
```

### Render the edited scene

```bash
# Dataset viewpoints
ns-gaussctrl-render dataset --load-config outputs/.../config.yml --output_path render/my_scene_edit

# Camera-path video
ns-gaussctrl-render camera-path --load-config outputs/.../config.yml --camera-path-filename data/my_scene/camera_paths/render-path.json --output_path render/my_scene_edit.mp4
```

### Evaluate

```bash
python metrics/evaluate.py --edited_dir render/my_scene_edit/rgb --original_dir render/original_my_scene/rgb --edit_prompt "..." --reverse_prompt "..."
```

### Key CLI flags

| Flag | Meaning |
|---|---|
| `--pipeline.ip_adapter_image_path` | Reference image for IP-Adapter |
| `--pipeline.ip_adapter_scale`      | IP-Adapter influence (0 = text only, 1 = image only) |
| `--pipeline.auto_ip_from_refs`     | Auto-pick best edited ref as IP-Adapter input |
| `--pipeline.langsam_obj`           | LangSAM target for the scene object |
| `--pipeline.ip_langsam_obj`        | LangSAM target for the reference image (`"none"` disables) |
| `--pipeline.ref_view_selection`    | `"fvs"` (default) or `"random"` |
| `--pipeline.fvs_alpha`             | FVS angular-distance weight (default 1.0) |

---

## Repository tour

| File | Role |
|---|---|
| `gaussctrl/gc_pipeline.py`    | Orchestrates render → invert → edit → train. Hosts `select_reference_views_fvs`, `_load_ip_adapter`, `_auto_select_ip_from_refs`, `_build_combined_attn_procs`, `edit_reference_views_sequential`, `edit_images`. |
| `gaussctrl/utils.py`          | `CrossViewAttnProcessor` — custom diffusers attention processor that replaces UNet `attn1` self-attention with cross-view attention over the reference frames. |
| `gaussctrl/gc_trainer.py`     | Extends NeRFStudio's `Trainer` with a short 500-step fine-tuning loop after editing. |
| `gaussctrl/gc_datamanager.py` | Subsamples 40 training views; stores per-view depth / z₀ latent / mask / unedited / edited image. |
| `gaussctrl/gc_config.py`      | Registers `"gaussctrl"` as a NeRFStudio method. |
| `metrics/evaluate.py`         | CLIP evaluation (`clip_score`, `clip_dir`, `clip_img`). |

---

## Citation

Please cite the original GaussCtrl paper if you use this work:

```bibtex
@article{gaussctrl2024,
  author  = {Wu, Jing and Bian, Jia-Wang and Li, Xinghui and Wang, Guangrun and Reid, Ian and Torr, Philip and Prisacariu, Victor},
  title   = {{GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing}},
  journal = {ECCV},
  year    = {2024}
}
```

Built with [NeRFStudio](https://docs.nerf.studio/), [gsplat](https://github.com/nerfstudio-project/gsplat), [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), [ImageReward](https://github.com/THUDM/ImageReward), and [lang-segment-anything](https://github.com/luca-medeiros/lang-segment-anything). Reference-view selection follows *NeRF Director* (Xiao et al., CVPR 2024).

## License

BSD — see [`LICENSE.txt`](LICENSE.txt).
