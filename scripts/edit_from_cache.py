"""
Standalone image editing script — runs the diffusion editing step only,
loading pre-rendered data from a cache directory (no 3DGS model needed).

Usage:
    python scripts/edit_from_cache.py \
        --cache_dir /data/.../cache/stable_diffusion_1.5/face \
        --output_dir /data/.../outputs/debug_edited_images/joker_IP_no_cross_att \
        --edit_prompt "a photo of joker" \
        --ip_adapter_image_path /data/.../joker.webp \
        --ip_adapter_scale 0.6 \
        --langsam_obj face
"""

import argparse
import os
import numpy as np
import torch
import torchvision
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.schedulers import DDIMScheduler


def depth2disparity(depth):
    """depth: numpy array [1, H, W] -> disparity [1, 3, H, W]"""
    disparity = 1 / (depth + 1e-5)
    disparity_map = disparity / np.max(disparity)
    disparity_map = np.concatenate([disparity_map, disparity_map, disparity_map], axis=0)
    return disparity_map[None]


def load_cache(cache_dir):
    """Load all views from cache. Returns list of dicts with depth, z0, rgb, mask."""
    views = []
    cam_idx = 0
    while True:
        z0_path = os.path.join(cache_dir, f"{cam_idx:04d}_z0.npy")
        if not os.path.exists(z0_path):
            break
        view = {
            'image_idx': cam_idx,
            'depth_image': np.load(os.path.join(cache_dir, f"{cam_idx:04d}_depth.npy")),
            'z_0_image': np.load(z0_path),
            'unedited_image': torch.from_numpy(np.load(os.path.join(cache_dir, f"{cam_idx:04d}_rgb.npy"))),
        }
        mask_path = os.path.join(cache_dir, f"{cam_idx:04d}_mask.npy")
        if os.path.exists(mask_path):
            view['mask_image'] = np.load(mask_path)
        views.append(view)
        cam_idx += 1
    print(f"Loaded {len(views)} views from cache.")
    return views


def load_ip_adapter(pipe, ip_adapter_image_path, ip_adapter_scale, langsam_obj):
    """Load IP-Adapter and optionally segment the reference image with LangSAM."""
    print(f"Loading IP-Adapter (scale={ip_adapter_scale})...")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")
    pipe.set_ip_adapter_scale(ip_adapter_scale)

    ip_image = Image.open(ip_adapter_image_path).convert("RGB")

    if langsam_obj:
        from lang_sam import LangSAM
        print(f"Segmenting '{langsam_obj}' from IP-Adapter reference image...")
        langsam = LangSAM()
        results = langsam.predict([ip_image], [langsam_obj])
        result_masks = results[0]["masks"]
        if len(result_masks) > 0:
            mask = result_masks[0]
            ip_array = np.array(ip_image)
            ip_array[mask == 0] = 255
            ip_image = Image.fromarray(ip_array)
            print("Segmentation successful.")
        else:
            print(f"Warning: LangSAM found no '{langsam_obj}', using full image.")

    print("IP-Adapter loaded.")
    return ip_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--edit_prompt", required=True)
    parser.add_argument("--ip_adapter_image_path", required=True)
    parser.add_argument("--ip_adapter_scale", type=float, default=0.6)
    parser.add_argument("--langsam_obj", default="")
    parser.add_argument("--diffusion_ckpt", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0"

    # Load diffusion pipeline
    print("Loading ControlNet and diffusion pipeline...")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.diffusion_ckpt, controlnet=controlnet,
        safety_checker=None, requires_safety_checker=False,
    ).to(device).to(torch.float16)
    pipe.scheduler = DDIMScheduler.from_pretrained(args.diffusion_ckpt, subfolder="scheduler")

    # Load IP-Adapter
    ip_image = load_ip_adapter(pipe, args.ip_adapter_image_path, args.ip_adapter_scale, args.langsam_obj)

    # Load cached views
    train_data = load_cache(args.cache_dir)

    # Prompts
    added = "best quality, extremely detailed"
    positive_prompt = args.edit_prompt + ", " + added
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    # Edit in chunks
    print("Starting editing...")
    for idx in range(0, len(train_data), args.chunk_size):
        chunk = train_data[idx: idx + args.chunk_size]
        indices = [v['image_idx'] for v in chunk]
        print(f"Editing views: {indices}")

        depth_images = [depth2disparity(v['depth_image']) for v in chunk]
        disparities = np.concatenate(depth_images, axis=0)
        disparities_torch = torch.from_numpy(disparities.copy()).to(torch.float16).to(device)

        z0s = np.concatenate([v['z_0_image'] for v in chunk], axis=0)
        latents_torch = torch.from_numpy(z0s.copy()).to(torch.float16).to(device)

        pipe_kwargs = dict(
            prompt=[positive_prompt] * len(chunk),
            negative_prompt=[negative_prompt] * len(chunk),
            latents=latents_torch,
            image=disparities_torch,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=1.0,
            eta=0.0,
            output_type='pt',
            ip_adapter_image=ip_image,
        )

        all_edited = pipe(**pipe_kwargs).images
        chunk_edited = all_edited.cpu()

        for local_idx, edited_image in enumerate(chunk_edited):
            global_idx = indices[local_idx]
            v = chunk[local_idx]

            result = edited_image
            if 'mask_image' in v:
                mask = torch.from_numpy(v['mask_image'])
                bg_mask = 1 - mask
                unedited = v['unedited_image'].permute(2, 0, 1)
                result = edited_image * mask[None] + unedited * bg_mask[None]

            torchvision.utils.save_image(result, os.path.join(args.output_dir, f"edited_{global_idx:04d}.png"))

    print(f"Done. Edited images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
