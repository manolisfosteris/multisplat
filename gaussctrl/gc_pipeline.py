# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GaussCtrl Pipeline and trainer"""

import os
from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional, Type, List
from rich.progress import Console
from copy import deepcopy
import numpy as np 
from PIL import Image
import mediapy as media
from lang_sam import LangSAM

import torch, random
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler
from gaussctrl.gc_datamanager import (
    GaussCtrlDataManagerConfig,
)
from diffusers.models.attention_processor import AttnProcessor
from gaussctrl import utils
from nerfstudio.viewer_legacy.server.utils import three_js_perspective_camera_focal_length
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils import colormaps

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler
import torchvision
import os

CONSOLE = Console(width=120)


def select_reference_views_fvs(cameras, num_refs: int, alpha: float) -> List[int]:
    """Select reference views using Farthest View Sampling (FVS).

    Greedily picks the most spatially and angularly diverse views.
    Based on Algorithm 1 from "NeRF Director" (Xiao et al., CVPR 2024).
    """
    is_list = isinstance(cameras, list)
    n = len(cameras)

    # Extract camera positions and viewing directions
    positions = torch.zeros(n, 3)
    directions = torch.zeros(n, 3)
    for i in range(n):
        cam = cameras[i] if is_list else cameras[i : i + 1]
        c2w = cam.camera_to_worlds
        positions[i] = c2w[0, :3, 3]
        d = -c2w[0, :3, 2]
        directions[i] = d / d.norm()

    # Pairwise spatial distance
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # [n, n, 3]
    d_spatial = diff.norm(dim=-1)  # [n, n]

    # Pairwise angular distance
    cos_sim = torch.clamp(torch.mm(directions, directions.t()), -1.0, 1.0)
    d_photo = torch.acos(cos_sim)  # [n, n]

    # Combined distance
    dist = d_spatial + alpha * d_photo  # [n, n]

    # Greedy FVS: start from view 0
    selected = [0]
    min_dist_to_S = dist[0].clone()  # min distance from each view to S

    for _ in range(num_refs - 1):
        # Mask out already selected
        min_dist_to_S[selected] = -1.0
        # Pick view with max min-distance to S
        v_star = int(torch.argmax(min_dist_to_S).item())
        selected.append(v_star)
        # Update min distances
        min_dist_to_S = torch.minimum(min_dist_to_S, dist[v_star])

    return sorted(selected)


@dataclass
class GaussCtrlPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: GaussCtrlPipeline)
    """target class to instantiate"""
    datamanager: GaussCtrlDataManagerConfig = GaussCtrlDataManagerConfig()
    """specifies the datamanager config"""
    render_rate: int = 500
    """how many gauss steps for gauss training"""
    cache_dir: str = ""
    """If set, save/load render_reverse outputs to/from this directory to skip re-rendering on subsequent runs"""
    edit_prompt: str = ""
    """Positive Prompt"""
    reverse_prompt: str = "" 
    """DDIM Inversion Prompt"""
    langsam_obj: str = ""
    """The object to be edited"""
    ip_langsam_obj: str = ""
    """Object to segment in IP-Adapter reference image (defaults to langsam_obj if empty, 'none' to disable)"""
    guidance_scale: float = 5
    """Classifier Free Guidance"""
    num_inference_steps: int = 20
    """Inference steps"""
    chunk_size: int = 5
    """Batch size for image editing, feel free to reduce to fit your GPU"""
    ref_view_num: int = 4
    """Number of reference frames"""
    diffusion_ckpt: str = 'runwayml/stable-diffusion-v1-5'
    """Diffusion checkpoints"""
    ip_adapter_image_path: str = ""
    """Path to reference image for IP-Adapter style guidance (empty = disabled)"""
    ip_adapter_scale: float = 0.6
    """IP-Adapter influence weight (0.0=text-only, 1.0=image-only)"""
    auto_ip_from_refs: bool = False
    """Auto-select best edited reference view as IP-Adapter input (requires ip_adapter_image_path to be empty)"""
    ref_view_selection: str = "fvs"
    """Reference view selection strategy: 'random' or 'fvs'"""
    fvs_alpha: float = 1.0
    """Scaling factor for photogrammetric (angular) distance in FVS"""


class GaussCtrlPipeline(VanillaPipeline):
    """GaussCtrl pipeline"""

    config: GaussCtrlPipelineConfig

    def __init__(
        self,
        config: GaussCtrlPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        self.test_mode = test_mode
        self.experiment_name = "default"
        self.langsam = LangSAM()
        
        self.edit_prompt = self.config.edit_prompt
        self.reverse_prompt = self.config.reverse_prompt
        self.pipe_device = 'cuda:0'
        self.ddim_scheduler = DDIMScheduler.from_pretrained(self.config.diffusion_ckpt, subfolder="scheduler")
        self.ddim_inverser = DDIMInverseScheduler.from_pretrained(self.config.diffusion_ckpt, subfolder="scheduler")
        
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(self.config.diffusion_ckpt, controlnet=controlnet, safety_checker=None, requires_safety_checker=False).to(self.device).to(torch.float16)
        self.pipe.to(self.pipe_device)

        # IP-Adapter: deferred to _load_ip_adapter() (called before edit_images, after render_reverse)
        self.ip_adapter_image = None
        self.ip_adapter_attn_procs = None

        added_prompt = 'best quality, extremely detailed'
        self.positive_prompt = self.edit_prompt + ', ' + added_prompt
        self.positive_reverse_prompt = self.reverse_prompt + ', ' + added_prompt
        self.negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
        
        view_num = len(self.datamanager.cameras)
        if self.config.ref_view_selection == "fvs":
            self.ref_indices = select_reference_views_fvs(
                self.datamanager.cameras, self.config.ref_view_num, self.config.fvs_alpha
            )
            CONSOLE.print(f"FVS selected reference views: {self.ref_indices}", style="bold green")
        else:
            anchors = [(view_num * i) // self.config.ref_view_num for i in range(self.config.ref_view_num)] + [view_num]
            random.seed(13789)
            self.ref_indices = [random.randint(anchor, anchors[idx+1]) for idx, anchor in enumerate(anchors[:-1])]
            CONSOLE.print(f"Random selected reference views: {self.ref_indices}", style="bold yellow")
        self.num_ref_views = len(self.ref_indices)

        self.num_inference_steps = self.config.num_inference_steps
        self.guidance_scale = self.config.guidance_scale
        self.controlnet_conditioning_scale = 1.0
        self.eta = 0.0
        self.chunk_size = self.config.chunk_size

    def _load_ip_adapter(self):
        """Load IP-Adapter weights and reference image. Called after render_reverse."""
        if not self.config.ip_adapter_image_path:
            return
        CONSOLE.print(f"Loading IP-Adapter with scale={self.config.ip_adapter_scale}", style="bold green")
        #loading the IP adapter
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")
        self.pipe.set_ip_adapter_scale(self.config.ip_adapter_scale)
        ip_image = Image.open(self.config.ip_adapter_image_path).convert("RGB")
        ip_seg_obj = self.config.ip_langsam_obj if self.config.ip_langsam_obj else self.config.langsam_obj
        if ip_seg_obj == "none":
            ip_seg_obj = ""
        if ip_seg_obj:
            CONSOLE.print(f"Segmenting '{ip_seg_obj}' from IP-Adapter reference image", style="bold green")
            results = self.langsam.predict([ip_image], [ip_seg_obj])
            result_masks = results[0]["masks"]
            if len(result_masks) > 0:
                mask = result_masks[0]
                ip_array = np.array(ip_image)
                ip_array[mask == 0] = 255  # white background
                ip_image = Image.fromarray(ip_array)
                ip_image.save("/data/leuven/385/vsc38511/outputs/debug_edited_images/ip_adapter_segmented.png")
                CONSOLE.print("IP-Adapter reference image segmented successfully", style="bold green")
            else:
                CONSOLE.print(f"Warning: LangSAM found no '{ip_seg_obj}' in IP-Adapter image, using full image", style="bold yellow")
        self.ip_adapter_image = ip_image
        self.ip_adapter_attn_procs = dict(self.pipe.unet.attn_processors)
        CONSOLE.print("IP-Adapter loaded successfully", style="bold green")

    def _auto_select_ip_from_refs(self, ref_save_dir: str):
        """Score edited reference views with ImageReward and use the best one as IP-Adapter input."""
        CONSOLE.print("Auto-selecting best edited reference view for IP-Adapter...", style="bold green")

        # Collect edited ref paths
        ref_paths = []
        for k, ref_idx in enumerate(self.ref_indices):
            path = f"{ref_save_dir}/ref_{k:02d}_idx{ref_idx:04d}_edited.png"
            ref_paths.append(path)

        # Score with ImageReward
        import ImageReward as RM
        reward_model = RM.load("ImageReward-v1.0", download_root="/scratch/leuven/385/vsc38511/.cache/ImageReward")

        best_score = float('-inf')
        best_path = ref_paths[0]
        for path in ref_paths:
            with torch.cuda.amp.autocast(enabled=False):
                score = reward_model.score(self.edit_prompt, path)
            CONSOLE.print(f"  ImageReward | {os.path.basename(path)}: {score:.4f}", style="dim")
            if score > best_score:
                best_score = score
                best_path = path

        # Free VRAM
        del reward_model
        torch.cuda.empty_cache()

        CONSOLE.print(f"Selected {os.path.basename(best_path)} (score={best_score:.4f}) as IP-Adapter reference", style="bold green")

        # Load IP-Adapter weights
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin")
        self.pipe.set_ip_adapter_scale(self.config.ip_adapter_scale)

        # Load and optionally segment the best image
        ip_image = Image.open(best_path).convert("RGB")
        ip_seg_obj = self.config.ip_langsam_obj if self.config.ip_langsam_obj else self.config.langsam_obj
        if ip_seg_obj == "none":
            ip_seg_obj = ""
        if ip_seg_obj:
            CONSOLE.print(f"Segmenting '{ip_seg_obj}' from auto-selected IP-Adapter image", style="bold green")
            results = self.langsam.predict([ip_image], [ip_seg_obj])
            result_masks = results[0]["masks"]
            if len(result_masks) > 0:
                mask = result_masks[0]
                ip_array = np.array(ip_image)
                ip_array[mask == 0] = 255
                ip_image = Image.fromarray(ip_array)
            else:
                CONSOLE.print(f"Warning: LangSAM found no '{ip_seg_obj}', using full image", style="bold yellow")

        self.ip_adapter_image = ip_image
        self.ip_adapter_attn_procs = dict(self.pipe.unet.attn_processors)
        ip_image.save(f"{ref_save_dir}/ip_adapter_input.png")
        CONSOLE.print(f"IP-Adapter input saved to {ref_save_dir}/ip_adapter_input.png", style="bold green")
        CONSOLE.print("IP-Adapter loaded from auto-selected reference", style="bold green")

    def _build_combined_attn_procs(self, self_attn_coeff, num_refs):
        """
        Build a combined attention processor dict that installs both CrossViewAttnProcessor
        and IP-Adapter processors simultaneously.

        In each UNet transformer block there are two attention layers:
          - attn1 (self-attention): replaced with CrossViewAttnProcessor to enforce
            multi-view consistency across reference and target frames.
          - attn2 (cross-attention to text/image): kept as-is from ip_adapter_attn_procs,
            so IP-Adapter style guidance from the reference image is preserved.

        Returns a dict suitable for pipe.unet.set_attn_processor().
        """                                                                                                                                                                                            
                     
        cross_view = utils.CrossViewAttnProcessor(self_attn_coeff=self_attn_coeff, unet_chunk_size=2, num_refs=num_refs)
        procs = {}
        for name, proc in self.ip_adapter_attn_procs.items():
            if name.endswith("attn1.processor"):
                procs[name] = cross_view
            else:
                procs[name] = proc
        return procs

    def render_reverse(self):
        '''Render rgb, depth and reverse rgb images back to latents'''
        if self.config.cache_dir and self._load_render_cache():
            return

        for cam_idx in range(len(self.datamanager.cameras)):
            CONSOLE.print(f"Rendering view {cam_idx}", style="bold yellow")
            current_cam = self.datamanager.cameras[cam_idx].to(self.device)
            if current_cam.metadata is None:
                current_cam.metadata = {}
            current_cam.metadata["cam_idx"] = cam_idx
            rendered_image = self._model.get_outputs_for_camera(current_cam)

            rendered_rgb = rendered_image['rgb'].to(torch.float16) # [512 512 3] 0-1
            rendered_depth = rendered_image['depth'].to(torch.float16) # [512 512 1]

            # reverse the images to noises
            self.pipe.unet.set_attn_processor(processor=AttnProcessor())
            self.pipe.controlnet.set_attn_processor(processor=AttnProcessor()) 
            init_latent = self.image2latent(rendered_rgb)
            disparity = self.depth2disparity_torch(rendered_depth[:,:,0][None]) 
            
            self.pipe.scheduler = self.ddim_inverser
            latent, _ = self.pipe(prompt=self.positive_reverse_prompt, num_inference_steps=self.num_inference_steps, latents=init_latent, image=disparity, return_dict=False, guidance_scale=0, output_type='latent')

            # LangSAM is optional
            if self.config.langsam_obj != "":
                langsam_obj = self.config.langsam_obj
                langsam_rgb_pil = Image.fromarray((rendered_rgb.cpu().numpy() * 255).astype(np.uint8))
                # Fosteris 05/03/2026: new lang_sam API expects lists; passing a bare string causes it to
                # iterate over characters (e.g. "bear" -> ["b","e","a","r"]), breaking batching.
                results = self.langsam.predict([langsam_rgb_pil], [langsam_obj])
                result_masks = results[0]["masks"]  # numpy array, new API returns list[dict]
                mask_npy = result_masks[0] * 1 if len(result_masks) > 0 else None

            if self.config.langsam_obj != "":
                self.update_datasets(cam_idx, rendered_rgb.cpu(), rendered_depth, latent, mask_npy)
            else:
                self.update_datasets(cam_idx, rendered_rgb.cpu(), rendered_depth, latent, None)

        if self.config.cache_dir:
            self._save_render_cache()

    def _save_render_cache(self):
        '''Save render_reverse outputs to disk for reuse.'''
        os.makedirs(self.config.cache_dir, exist_ok=True)
        for cam_idx, data in enumerate(self.datamanager.train_data):
            np.save(os.path.join(self.config.cache_dir, f"{cam_idx:04d}_depth.npy"), data['depth_image'])
            np.save(os.path.join(self.config.cache_dir, f"{cam_idx:04d}_z0.npy"), data['z_0_image'])
            np.save(os.path.join(self.config.cache_dir, f"{cam_idx:04d}_rgb.npy"), data['unedited_image'].numpy())
            if 'mask_image' in data:
                np.save(os.path.join(self.config.cache_dir, f"{cam_idx:04d}_mask.npy"), data['mask_image'])
        CONSOLE.print(f"Saved render cache to {self.config.cache_dir}", style="bold green")

    def _load_render_cache(self):
        '''Load render_reverse outputs from disk. Returns True if successful.'''
        first = os.path.join(self.config.cache_dir, "0000_z0.npy")
        if not os.path.exists(first):
            CONSOLE.print(f"No render cache found at {self.config.cache_dir}, running render_reverse.", style="bold yellow")
            return False
        CONSOLE.print(f"Loading render cache from {self.config.cache_dir}", style="bold green")
        num_views = len(self.datamanager.train_data)
        for cam_idx in range(num_views):
            self.datamanager.train_data[cam_idx]['depth_image'] = np.load(os.path.join(self.config.cache_dir, f"{cam_idx:04d}_depth.npy"))
            self.datamanager.train_data[cam_idx]['z_0_image'] = np.load(os.path.join(self.config.cache_dir, f"{cam_idx:04d}_z0.npy"))
            self.datamanager.train_data[cam_idx]['unedited_image'] = torch.from_numpy(np.load(os.path.join(self.config.cache_dir, f"{cam_idx:04d}_rgb.npy")))
            mask_path = os.path.join(self.config.cache_dir, f"{cam_idx:04d}_mask.npy")
            if os.path.exists(mask_path):
                self.datamanager.train_data[cam_idx]['mask_image'] = np.load(mask_path)
        return True

    def edit_reference_views_sequential(self, save_dir):
        '''Edit reference views sequentially: ref_0 is edited with all 4 original ref latents
        for multi-view quality, then refs 1-3 attend to previously edited refs.
        Returns stacked z_0 latents and disparities for all edited reference views.'''
        self.pipe.scheduler = self.ddim_scheduler

        edited_latents = []    # list of [1, 4, 64, 64] float16 tensors
        edited_disparities = []  # list of [1, 3, 512, 512] float16 tensors

        # Preload all ref z0s and disps (original unedited) for use in ref_0's joint edit
        all_ref_z0s = []
        all_ref_disps = []
        for ri in self.ref_indices:
            rd = self.datamanager.train_data[ri]
            all_ref_disps.append(torch.from_numpy(
                self.depth2disparity(rd['depth_image']).copy()
            ).to(torch.float16).to(self.pipe_device))
            all_ref_z0s.append(torch.from_numpy(
                rd['z_0_image'].copy()
            ).to(torch.float16).to(self.pipe_device))

        for k, ref_idx in enumerate(self.ref_indices):
            CONSOLE.print(f"Sequential ref editing {k+1}/{self.num_ref_views} (view {ref_idx})", style="bold yellow")
            ref_data = deepcopy(self.datamanager.train_data[ref_idx])
            ref_disp = all_ref_disps[k]
            ref_z0 = all_ref_z0s[k]

            if k == 0:
                num_refs_k = self.num_ref_views
            else:
                num_refs_k = k

            # Set attention processors: combined (CrossView + IP-Adapter) or CrossView only
            if self.ip_adapter_image is not None:
                self.pipe.unet.set_attn_processor(self._build_combined_attn_procs(self_attn_coeff=0.6, num_refs=num_refs_k))
            else:
                self.pipe.unet.set_attn_processor(
                    processor=utils.CrossViewAttnProcessor(self_attn_coeff=0.6, unet_chunk_size=2, num_refs=num_refs_k))
            self.pipe.controlnet.set_attn_processor(
                processor=utils.CrossViewAttnProcessor(self_attn_coeff=0, unet_chunk_size=2, num_refs=num_refs_k))

            if k == 0:
                batch_latents = torch.cat(all_ref_z0s, dim=0)   # [num_refs, 4, 64, 64]
                batch_disp = torch.cat(all_ref_disps, dim=0)    # [num_refs, 3, 512, 512]
                num_prompts = self.num_ref_views
                keep_idx = 0  # ref_0 is the first in the batch
            else:
                prev_latents = torch.cat(edited_latents, dim=0)    # [k, 4, 64, 64]
                prev_disps = torch.cat(edited_disparities, dim=0)  # [k, 3, 512, 512]
                batch_latents = torch.cat([prev_latents, ref_z0], dim=0)  # [k+1, 4, 64, 64]
                batch_disp = torch.cat([prev_disps, ref_disp], dim=0)    # [k+1, 3, 512, 512]
                num_prompts = k + 1
                keep_idx = -1  # current ref is the last in the batch

            pipe_kwargs = dict(
                prompt=[self.positive_prompt] * num_prompts,
                negative_prompt=[self.negative_prompts] * num_prompts,
                latents=batch_latents,
                image=batch_disp,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                eta=self.eta,
                output_type='pt',
            )
            if self.ip_adapter_image is not None:
                pipe_kwargs['ip_adapter_image'] = self.ip_adapter_image
            result_images = self.pipe(**pipe_kwargs).images

            edited_img = result_images[keep_idx].cpu()  # [C, H, W], keep only the current ref

            # Apply LangSAM mask if available (keep unedited background, same as target views)
            if 'mask_image' in ref_data:
                mask = torch.from_numpy(ref_data['mask_image'])  # [H, W]
                bg_mask = 1 - mask
                unedited_ref = ref_data['unedited_image'].permute(2, 0, 1)  # [C, H, W]
                edited_img = edited_img * mask[None] + unedited_ref * bg_mask[None]

            torchvision.utils.save_image(edited_img, f"{save_dir}/ref_{k:02d}_idx{ref_idx:04d}_edited.png")

            # DDIM-invert the edited image to get a proper noisy latent at t=T
            edited_img_hwc = edited_img.to(torch.float16).permute(1, 2, 0).to(self.pipe_device)
            vae_latent = self.image2latent(edited_img_hwc)  # clean latent at t=0
            self.pipe.scheduler = self.ddim_inverser
            inversion_kwargs = dict(
                prompt=self.positive_reverse_prompt,
                num_inference_steps=self.num_inference_steps,
                latents=vae_latent,
                image=ref_disp,
                return_dict=False,
                guidance_scale=0,
                output_type='latent',
            )
            if self.ip_adapter_image is not None:
                inversion_kwargs['ip_adapter_image'] = self.ip_adapter_image
            new_z0, _ = self.pipe(**inversion_kwargs)
            new_z0 = new_z0.to(torch.float16)  # [1, 4, 64, 64]
            self.pipe.scheduler = self.ddim_scheduler  # restore for next edit

            edited_latents.append(new_z0)
            edited_disparities.append(ref_disp)

        ref_z0_torch = torch.cat(edited_latents, dim=0)      # [num_refs, 4, 64, 64]
        ref_disp_torch = torch.cat(edited_disparities, dim=0)  # [num_refs, 3, 512, 512]
        return ref_z0_torch, ref_disp_torch

    def edit_images(self, base_dir=None):
        '''Edit images with ControlNet and AttnAlign'''
        self._load_ip_adapter()
        self.pipe.scheduler = self.ddim_scheduler

        print("#############################")
        CONSOLE.print("Start Editing: ", style="bold yellow")
        CONSOLE.print(f"Reference views are {[j+1 for j in self.ref_indices]}", style="bold yellow")
        print("#############################")

        ref_save_dir = f"/data/leuven/385/vsc38511/outputs/debug_edited_images/{base_dir}" if base_dir is not None else "/data/leuven/385/vsc38511/outputs/debug_edited_images"
        os.makedirs(ref_save_dir, exist_ok=True)

        # Sequentially edit reference views for consistency
        ref_z0_torch, ref_disparity_torch = self.edit_reference_views_sequential(ref_save_dir)

        # Auto-select best edited ref as IP-Adapter input if configured
        if self.ip_adapter_image is None and self.config.auto_ip_from_refs:
            self._auto_select_ip_from_refs(ref_save_dir)

        # Reset processor for target view editing
        if self.ip_adapter_image is not None:
            self.pipe.unet.set_attn_processor(self._build_combined_attn_procs(self_attn_coeff=0.6, num_refs=self.num_ref_views))
        else:
            self.pipe.unet.set_attn_processor(
                processor=utils.CrossViewAttnProcessor(self_attn_coeff=0.6, unet_chunk_size=2))
        self.pipe.controlnet.set_attn_processor(
            processor=utils.CrossViewAttnProcessor(self_attn_coeff=0, unet_chunk_size=2))
        CONSOLE.print("Done sequential ref editing, starting target view editing", style="bold blue")

        # Edit images in chunk
        for idx in range(0, len(self.datamanager.train_data), self.chunk_size):
            chunked_data = self.datamanager.train_data[idx: idx+self.chunk_size]
            
            indices = [current_data['image_idx'] for current_data in chunked_data]
            mask_images = [current_data['mask_image'] for current_data in chunked_data if 'mask_image' in current_data.keys()] 
            unedited_images = [current_data['unedited_image'] for current_data in chunked_data]
            CONSOLE.print(f"Generating view: {indices}", style="bold yellow")

            depth_images = [self.depth2disparity(current_data['depth_image']) for current_data in chunked_data]
            disparities = np.concatenate(depth_images, axis=0)
            disparities_torch = torch.from_numpy(disparities.copy()).to(torch.float16).to(self.pipe_device)

            z_0_images = [current_data['z_0_image'] for current_data in chunked_data] # list of np array
            z0s = np.concatenate(z_0_images, axis=0)
            latents_torch = torch.from_numpy(z0s.copy()).to(torch.float16).to(self.pipe_device)

            disp_ctrl_chunk = torch.concatenate((ref_disparity_torch, disparities_torch), dim=0)
            latents_chunk = torch.concatenate((ref_z0_torch, latents_torch), dim=0)
            
            pipe_kwargs = dict(
                prompt=[self.positive_prompt] * (self.num_ref_views+len(chunked_data)),
                negative_prompt=[self.negative_prompts] * (self.num_ref_views+len(chunked_data)),
                latents=latents_chunk,
                image=disp_ctrl_chunk,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                eta=self.eta,
                output_type='pt',
            )
            if self.ip_adapter_image is not None:
                pipe_kwargs['ip_adapter_image'] = self.ip_adapter_image
            all_edited = self.pipe(**pipe_kwargs).images
            chunk_edited = all_edited[self.num_ref_views:].cpu()

            # Save reference view reconstructions for the first chunk to verify round-trip fidelity
            if idx == 0:
                ref_recon = all_edited[:self.num_ref_views].cpu()
                for ri, recon_img in enumerate(ref_recon):
                    torchvision.utils.save_image(recon_img, f"{ref_save_dir}/ref_{ri:02d}_idx{self.ref_indices[ri]:04d}_reconstructed.png")

            # Insert edited images back to train data for training
            for local_idx, edited_image in enumerate(chunk_edited):
                global_idx = indices[local_idx]

                bg_cntrl_edited_image = edited_image
                if mask_images != []:
                    mask = torch.from_numpy(mask_images[local_idx])
                    bg_mask = 1 - mask

                    unedited_image = unedited_images[local_idx].permute(2,0,1)
                    bg_cntrl_edited_image = edited_image * mask[None] + unedited_image * bg_mask[None] 

                self.datamanager.train_data[global_idx]["image"] = bg_cntrl_edited_image.permute(1,2,0).to(torch.float32) # [512 512 3]
                torchvision.utils.save_image(bg_cntrl_edited_image, f"{ref_save_dir}/edited_{global_idx:04d}.png")
        print("#############################")
        CONSOLE.print("Done Editing", style="bold yellow")
        print("#############################")

    @torch.no_grad()
    def image2latent(self, image):
        """Encode images to latents"""
        image = image * 2 - 1
        image = image.permute(2, 0, 1).unsqueeze(0) # torch.Size([1, 3, 512, 512]) -1~1
        latents = self.pipe.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    def depth2disparity(self, depth):
        """
        Args: depth numpy array [1 512 512]
        Return: disparity
        """
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / np.max(disparity) # 0.00233~1
        disparity_map = np.concatenate([disparity_map, disparity_map, disparity_map], axis=0)
        return disparity_map[None]
    
    def depth2disparity_torch(self, depth):
        """
        Args: depth torch tensor
        Return: disparity
        """
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / torch.max(disparity) # 0.00233~1
        disparity_map = torch.concatenate([disparity_map, disparity_map, disparity_map], dim=0)
        return disparity_map[None]

    def update_datasets(self, cam_idx, unedited_image, depth, latent, mask):
        """Save mid results"""
        self.datamanager.train_data[cam_idx]["unedited_image"] = unedited_image 
        self.datamanager.train_data[cam_idx]["depth_image"] = depth.permute(2,0,1).cpu().to(torch.float32).numpy()
        self.datamanager.train_data[cam_idx]["z_0_image"] = latent.cpu().to(torch.float32).numpy()
        if mask is not None:
            self.datamanager.train_data[cam_idx]["mask_image"] = mask 

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step) # camera, data
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
