# Experiment Log

## Naming convention
- `{scene}_{method}_{descriptor}` — e.g. `bear_original`, `bear_epipolar_v1`
- Methods: `original` | `epipolar` | `sequential_ref`

## Run commands
```bash
# Original
ns-train gaussctrl \
  --load-checkpoint {ckpt} \
  --experiment-name {name} \
  --output-dir outputs \
  --pipeline.datamanager.data {data} \
  --pipeline.edit_prompt "{prompt}" \
  --pipeline.reverse_prompt "{reverse_prompt}" \
  --pipeline.guidance_scale 5 \
  --pipeline.chunk_size 3

# Render dataset views
ns-gaussctrl-render dataset --load-config outputs/{name}/gaussctrl/{timestamp}/config.yml --output_path render/{name}

# Render video
ns-gaussctrl-render camera-path --load-config outputs/{name}/gaussctrl/{timestamp}/config.yml --camera-path-filename data/{scene}/camera_paths/render-path.json --output_path render/{name}.mp4
```

## Experiments

| experiment_name           | date       | scene | method           | edit_prompt                             | guidance_scale | chunk_size | notes                                                                    |
|---------------------------|------------|-------|------------------|-----------------------------------------|----------------|------------|--------------------------------------------------------------------------|
| panda_main                | 2026-03-07 | bear  | original         | "a photo of a panda in the forest"      | 5              | 1          | langsam_obj=bear                                                         |
| panda_seq_v4              | 2026-03-09 | bear  | sequential_ref   | "a photo of a panda in the forest"      | 5              | 1          | langsam_obj=bear                                                         |
| polar_bear_seq_v4         | 2026-03-10 | bear  | sequential_ref   | "a photo of a polar bear in the forest" | 5              | 1          | langsam_obj=bear                                                         |
| polar_bear_main           | 2026-03-12 | bear  | original         | "a photo of a polar bear in the forest" | 5              | 1          | langsam_obj=bear                                                         |
| panda_stable_1_5_ip_cross | 2026-03-23 | bear  | ip_adapter+cross | "a photo of a panda in the forest"      | 5              | 1          | SD1.5, ip_adapter_scale=0.6, ip_image=panda_mage.webp, langsam_obj=bear |
