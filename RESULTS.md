# Results

## Scene: bear

### Edit: bear → panda

| experiment                | date       | method           | clip_score | clip_dir | clip_img | notes                             |
|---------------------------|------------|------------------|------------|----------|----------|-----------------------------------|
| panda_seq_v4              | 2026-03-09 | sequential_ref   | 0.2463     | 0.1971   | 0.7275   | guidance=5, chunk=1, langsam=bear |
| panda_main                | 2026-03-07 | original         | 0.2394     | 0.1739   | 0.7592   | guidance=5, chunk=1, langsam=bear |
| panda_stable_1_5_ip_cross | 2026-03-23 | ip_adapter+cross | 0.2461     | 0.2049   | 0.7248   | SD1.5, ip_scale=0.6, langsam=bear |

### Edit: bear → polar bear

| experiment        | date       | method         | clip_score | clip_dir | clip_img | notes                             |
|-------------------|------------|----------------|------------|----------|----------|-----------------------------------|
| polar_bear_seq_v4 | 2026-03-10 | sequential_ref | 0.2546     | 0.1588   | 0.7719   | guidance=5, chunk=1, langsam=bear |
| polar_bear_main   | 2026-03-12 | original       | 0.2549     | 0.1578   | 0.7779   | guidance=5, chunk=1, langsam=bear |
