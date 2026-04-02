"""
Evaluate GaussCtrl editing results using CLIP metrics.

Usage:
    python metrics/evaluate.py \
        --edited_dir render/my_experiment \
        --original_dir render/original \
        --edit_prompt "a photo of a polar bear in the forest" \
        --reverse_prompt "a photo of a bear statue in the forest"

If --original_dir is not provided, only clip_score is computed (no directional similarity or image similarity).
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from clip_metrics import ClipSimilarity


def load_image(path):
    return transforms.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edited_dir", required=True, type=Path, help="Directory with edited rendered images")
    parser.add_argument("--original_dir", type=Path, default=None, help="Directory with original (unedited) rendered images")
    parser.add_argument("--edit_prompt", required=True, type=str, help="Text prompt describing the edit")
    parser.add_argument("--reverse_prompt", type=str, default=None, help="Text prompt describing the original scene")
    parser.add_argument("--clip_model", type=str, default="ViT-L/14", help="CLIP model variant")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    metric = ClipSimilarity(name=args.clip_model).to(args.device)

    edited_paths = sorted(args.edited_dir.glob("*.png")) + sorted(args.edited_dir.glob("*.jpg"))
    if not edited_paths:
        raise ValueError(f"No images found in {args.edited_dir}")

    scores = {"clip_score": []}
    if args.original_dir is not None:
        scores["clip_dir"] = []
        scores["clip_img"] = []

    for img_path in edited_paths:
        edited = load_image(img_path).to(args.device)

        if args.original_dir is not None:
            orig_path = args.original_dir / img_path.name
            if not orig_path.exists():
                print(f"Warning: no matching original for {img_path.name}, skipping")
                continue
            original = load_image(orig_path).to(args.device)
            sim_0, sim_1, sim_dir, sim_img = metric(original, edited, args.reverse_prompt, args.edit_prompt)
            scores["clip_score"].append(sim_1.item())
            scores["clip_dir"].append(sim_dir.item())
            scores["clip_img"].append(sim_img.item())
        else:
            text_features = metric.encode_text([args.edit_prompt])
            image_features = metric.encode_image(edited)
            import torch.nn.functional as F
            sim_1 = F.cosine_similarity(image_features, text_features)
            scores["clip_score"].append(sim_1.item())

    print(f"\nResults over {len(scores['clip_score'])} images:")
    for k, vals in scores.items():
        print(f"  {k}: {sum(vals) / len(vals):.4f}")


if __name__ == "__main__":
    main()
