# Frozen-input E/J integrated smoke

**12/12 COMPLETE.** Reconstructed tuning-only inputs; not paper-formula reproduction or efficacy.

| Dataset | Task | Method | PSNR | LPIPS VGG | Sampling seconds |
|---|---|---|---:|---:|---:|
| AFHQ-Dog | sr4 | E | 24.633777 | 0.384666 | 6.684 |
| AFHQ-Dog | sr4 | J | 27.200338 | 0.280467 | 69.789 |
| AFHQ-Dog | motion61 | E | 22.335058 | 0.387025 | 6.261 |
| AFHQ-Dog | motion61 | J | 24.035866 | 0.361414 | 65.398 |
| AFHQ-Dog | mask64 | E | 23.237680 | 0.238577 | 6.391 |
| AFHQ-Dog | mask64 | J | 26.088280 | 0.244588 | 66.180 |
| FFHQ | sr4 | E | 30.230237 | 0.330713 | 5.952 |
| FFHQ | sr4 | J | 31.651728 | 0.245611 | 65.774 |
| FFHQ | motion61 | E | 27.295619 | 0.356841 | 6.128 |
| FFHQ | motion61 | J | 28.542325 | 0.332624 | 66.669 |
| FFHQ | mask64 | E | 31.977055 | 0.171162 | 5.837 |
| FFHQ | mask64 | J | 33.749727 | 0.147207 | 66.904 |

Total sampling: 437.967s; total phase execution including both model loads: 556.231s.

Counts: {"predict_forward": 1500, "transformer_forward": 1500, "observed_block_forward_including_recompute": 153900, "vae_decode": 2712, "autograd_calls": 2700}.

PSNR RGB[0,1] and LPIPS VGG v0.1,256px[-1,1]; measured on float predictions before PNG quantization.

Profile: untouched released sampler, flow_map1, norm/pixel, no early stop. E50steps/5inner iterations/eta6; J200steps/1inner iteration/normalized gradient/eta5(AFHQ) or3(FFHQ).

All twelve image/record hashes, input payload identities, per-cell counters and phase identities verified. Guard remained active during sampling; only seeded initial latent was created before it.

Full launch remains disabled. Missing/failed outputs: none. No Modal or paid services used for this run.

Runner: `e84e3d0e1eec542442a4ae48df50442f467e65db`. The public JSON adds sealed settings and input identities and replaces the hardware UUID with a fingerprint. The raw company report SHA256 is `f17000d595778164371476197073b3cbd3c83137b2622d25a408908a70a481ca`.
