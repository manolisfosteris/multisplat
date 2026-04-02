"""
Standalone 3DGS retraining script — skips render_reverse and edit_images,
loads pre-edited images from disk and retrains the Gaussian Splatting model.

Intended workflow:
  1. Run edit_from_cache.py on P100 (diffusion editing, no gsplat needed)
  2. Run this script on V100 (3DGS retraining, requires gsplat)

Usage:
    python scripts/retrain_from_edited.py \
        --load_checkpoint /data/.../nerfstudio_models/step-000029999.ckpt \
        --edited_images_dir /data/.../outputs/debug_edited_images/joker_IP_no_cross_att \
        --data /data/.../gaussctrl-fork/data/face \
        --cache_dir /data/.../cache/stable_diffusion_1.5/face \
        --experiment_name joker_IP_no_cross_att \
        --output_dir /data/.../outputs/face
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig

from gaussctrl.gc_datamanager import GaussCtrlDataManagerConfig, GaussCtrlDataManager
from gaussctrl.gc_model import GaussCtrlModelConfig
from gaussctrl.gc_pipeline import GaussCtrlPipeline, GaussCtrlPipelineConfig
from gaussctrl.gc_trainer import GaussCtrlTrainerConfig
from gaussctrl.gc_dataparser_ns import GaussCtrlDataParserConfig
from gaussctrl.gc_dataset import GCDataset


def build_config(args):
    config = GaussCtrlTrainerConfig(
        method_name="gaussctrl",
        experiment_name=args.experiment_name,
        output_dir=Path(args.output_dir),
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=250,
        max_num_iterations=1000,
        steps_per_eval_all_images=1000,
        save_only_latest_checkpoint=True,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100},
        pipeline=GaussCtrlPipelineConfig(
            datamanager=GaussCtrlDataManagerConfig(
                _target=GaussCtrlDataManager[GCDataset],
                dataparser=GaussCtrlDataParserConfig(load_3D_points=True),
                data=Path(args.data),
            ),
            model=GaussCtrlModelConfig(),
            cache_dir=args.cache_dir,
            render_rate=args.render_rate,
            # Diffusion params unused (editing is skipped), but required by pipeline init
            edit_prompt="",
            reverse_prompt="",
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=30000),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15, quit_on_train_completion=True),
        vis="viewer",
    )
    config.set_timestamp()
    config.load_checkpoint = Path(args.load_checkpoint)
    return config


def make_load_edited_fn(edited_images_dir):
    """Returns a replacement for pipeline.edit_images that loads from disk."""
    edited_dir = Path(edited_images_dir)
    to_tensor = transforms.ToTensor()

    def load_edited_images(self, base_dir=None):
        print(f"Loading pre-edited images from {edited_dir}")
        loaded, missing = 0, 0
        for data in self.datamanager.train_data:
            idx = data['image_idx']
            img_path = edited_dir / f"edited_{idx:04d}.png"
            if img_path.exists():
                img = to_tensor(Image.open(img_path).convert("RGB"))
                data['image'] = img.permute(1, 2, 0).to(torch.float32)
                loaded += 1
            else:
                print(f"  Warning: no edited image for idx {idx}, keeping unedited")
                missing += 1
        print(f"Loaded {loaded} edited images ({missing} missing).")

    return load_edited_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint", required=True, help="Path to pretrained splatfacto .ckpt")
    parser.add_argument("--edited_images_dir", required=True, help="Directory with edited_XXXX.png files")
    parser.add_argument("--data", required=True, help="Path to scene data directory")
    parser.add_argument("--cache_dir", default="", help="Render cache directory (skip re-rendering if set)")
    parser.add_argument("--experiment_name", required=True, help="Name for this experiment")
    parser.add_argument("--output_dir", required=True, help="Output directory for checkpoints and config")
    parser.add_argument("--render_rate", type=int, default=500, help="Number of 3DGS training steps")
    args = parser.parse_args()

    # Patch edit_images on the class before the trainer instantiates the pipeline
    GaussCtrlPipeline.edit_images = make_load_edited_fn(args.edited_images_dir)

    config = build_config(args)
    config.save_config()
    trainer = config._target(config=config, local_rank=0, world_size=1)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
