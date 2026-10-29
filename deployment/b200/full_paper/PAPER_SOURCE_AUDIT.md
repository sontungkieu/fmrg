# Full-paper audit — 29 October 2026

Audit complete; full execution remains blocked. `CONTRACT.json` is the machine-readable specification. This audit pins arXiv2604.27147v3 and upstream `be485ba40f1d1c163b95ad61112aca6baa13ed22`. It does not certify that the released CLI reproduces the paper formulas. Neither a full experiment nor a GPU evaluator run was launched in this preparation.

## Sources and evidence

[Original paper, v3](https://arxiv.org/html/2604.27147v3), PDF58pages, SHA256 `640ff820d4b7dea270ebf8687ff29dbc476d9c08d63af0294f6b564eb8de2281`. Printed/PDF pages40,41,54,58 were also visually inspected; Eq.134's square is visible. PDF is mirrored with attribution under its CC BY4.0 license. Source file byte hashes are in CONTRACT.json; GitHub line references below use the pinned scientific commit. A deployment Git commit is a distinct operational identity.

## Paper specification versus executable release

| Topic | Paper | Released source and consequence |
|---|---|---|
| Inverse loss | E.3 p40 Eq.134: negative squared pixel observation residual | `fluxfm_sampler_ip.py:127-147`; `solve_inverse_problem.py:164-167` defaults norm/latent. `sum` + `pixel` expresses the squared observation loss; latent means pseudo-inverse weighted residual, not VAE latent distance. Changing defaults alone does not fix time semantics. |
| Time | E.2 p40: data-time0→1 | `predict_vector` uses noise-time sigma1→0; t=1−sigma. |
| E strength | E.2: eta*t_cur*(1−t_next) | IP:590 and reward:918/1084 use eta*sigma_cur*(1−sigma_next); converted paper expression is eta*(1−sigma_cur)*sigma_next. These are different. |
| Early stop | E.2: end guidance then one unguided flow-map jump | IP:511-516 truncates to [:k]+[0] but still guides last interval. Reward:1006/1142+ has another branch and unguided_steps; integer k is not a data-time threshold. |
| J optimizer | E.2: gradient normalized and rescaled by velocity norm, eta*delta_t | `generate_geneval.py:185` forces Adam. `reward_consistency_xt`:404-431 returns before normalization at453; --normalize_grad does not make this branch paper-J. LR decays with step index. |
| Data | E.3: each dataset50tuning +1000eval | No exact image IDs/split hashes in inspected paper/release. Source preprocess is RGB, direct256square BICUBIC, tensor→[-1,1] (`solve_inverse_problem.py:294-298,346-352`). LPIPS metric resize is a separate bilinear224 transform. |
| Observation | E.3: 4xavgpool,61x61motion blur(.5),64x64center mask, Gaussian sigma.03 | Source noise draws use global RNG after model init, batched/resumed execution changes consumption. Freeze kernel/noise seed/measurement bytes before tuning; a seed flag alone is insufficient. |
| GenEval | E.5: Mask2Former ResNet50 | Pinned official GenEval defaults Swin-S. Both checkpoints are staged; paper-R50 and official-default-Swin scores need separate labels. R50 needs explicit config AND model basename. |
| VLM | E.6 p58: sigmoid(logitYes−logitNo) after exact question | Skywork model card describes a scalar value head; these are different rewards. No matching differentiable runner or renoise constant c is provided by the inspected release. |

## Settings confirmed from paper

TableE.2 p41 (all three inverse tasks, both datasets): E50steps/NFE100,eta6,nopt5; J200steps/NFE400,nopt1,eta5AFHQ-Dog/eta3FFHQ. E stops at data-time.67; J does not early-stop (TableE.1). CFG3.5;256px rank32 LoRA. High-NFE advances use flow_map2. Table settings are specifications, not executable CLI presets with verified equivalence.

E.5/TableE.17 p54:512px; ensemble5HPS+1ImageReward+.01CLIP+.05PickScore; K3 warmup selection. E100/eta.7/nopt3, J20/eta3/nopt1, J100/eta5/nopt1, all t_stop.25. J20 reuses endpoint prediction; E100/J100 use flow_map2. GenEval553prompts×4seeds, aggregate across seeds then mean six category accuracies. Released sweep seeds0,1,2,3 are pinned separately; paper does not identify all randomness. Baseline paper settings include FMTT7particles/2clones/20steps/lookahead4/scale4.5 (NFE1400), ReNO50optim+8sampling/lr5 (~NFE58), BoN32×4steps(NFE128). Their complete baseline environments remain outside this prepared bundle; do not claim a complete baseline runtime.

E.4 p52: style uses CLIP ViT-B/16 layer2 Gram features (not VGG19),256px,16reference styles with3prompts/style, NFEbudget100. Reference images, exact48paired prompts and chosen config grid are unresolved. E.6 p58: Skywork7B,20guidedsteps,eta1,K4; stochastic renoise c is not numerically specified. E.7 timings use L40S/H100, not B200; the old J pilot's~5s/image is not a cost estimate for400NFE+all evaluators.

## CPU control-flow and metric checks

`evidence/control-flow-trace.json` uses actual pinned sampler control flow with synthetic velocity/reward, not FLUX weights. It records every predict_vector call and the actual sigma grid. Released sweep tags21/100/99 produce20/99/97 calls in the three tested branches; effective data-time cutoffs are approximately.12977/.06782/.06771, not.25. The count includes warmup and selection forwards along this path, but not checkpoint recomputation or neural backward cost. Never report the directory tag as measured compute. The trace records VAE scale/pipeline init identity when refreshed; source comments are not authoritative configuration.

CPU LPIPS identity0, perturbed.0066795311868190765, input-gradient norm.008097774349153042; two finite2048-D FID feature samples. This is an offline synthetic code canary; no dataset FID score, B200 ensemble gradient or GenEval native kernel was tested. `torch-fidelity==0.3.0` is a hash-pinned pure-Python overlay; the frozen generation venv is unchanged.

Released LPIPS computes VGG on resized224 tensors[0,1], normalize=False. Standard[-1,1]/native-resolution LPIPS is a separate sensitivity profile; do not silently mix the numbers. FID/KID pre-resize299bilinear/antialias then uint8; KID subset_size=min(50,N), default100subsets, no explicit seed in aggregator. `aggregate_metrics.py` skips malformed rows and intersects filenames; full evaluation must reject incomplete identities/counts before using it. Bootstrapping must respect image/prompt clusters and recompute dataset metrics; mean differences do not establish distribution equivalence.

## Assets and unresolved work

The new immutable asset manifest contains30files/10,153,895,280bytes: full HPSv2.1, ImageReward, PickScore, BERTtokenizer, CLIPB16(style)/L14(GenEval), VGG16/LPIPScalibration/FIDInception, both detectors, pinned source archives and paper. Existing core FLUX/LoRAs/CLIP-H assets are reused. Model download success is not evaluator qualification.

Skywork README is public but16required config/tokenizer/weight files return GatedRepoError for the authorized HF account. No terms were accepted and no alternative mirror bypass was attempted. GenEval needs separate native qualification: original pins Python3.9/Torch1.12.1+cu113, mmcv-full1.7.1, mmdet2.28.2, open-clip2.20, timm.9.2. Those native binaries are absent from the cu128/Torch2.7 generation runtime; changing versions to support B200 requires a separately recorded compatibility decision and kernel canary. A supported separate evaluator GPU/CPU path is possible only after authorization/qualification; it is not silently substituted.

`dataset-contract.json` is an explicit empty author-exact split contract; no arbitrary dataset mirror/selection was fabricated. Tuning objective, complete grid, seed derivation, baseline environment, full ablation/figure matrix, VLM adapter and company allocation remain explicit blockers in CONTRACT.json. Full-launch permission stays false until a new reviewed contract and authorization.

## Screening closure

`company-screening-reported.json` preserves the user's company report as USER_REPORTED, not independently audited raw receipts. Verdict: replay screening completed, not bitwise, distribution equivalence unconfirmed. No company log file was written by this preparation. Any reconstructed provenance must be labeled reconstructed. Screening is not efficacy and does not validate a universal3NFE headline.

## Next execution contract (planning only)

Start with evaluator canaries and frozen data/sampler manifests; keep release-v1 and paper-formula-v1 results separate. The core inverse FMRG comparison alone is2datasets×3tasks×2methods×1000=12,000outputs per chosen run seed, plus50held-out tuning images/dataset and all tuning/selection work. Three main reward configs require553×4×3=6,636outputs before baselines. These are partial table workloads, not the complete set of paper ablations. Derive estimates from a bounded representative technical timing run after semantics are resolved; sum actual warmup/discarded particles/forwards/backwards/recompute/decoder/evaluator costs, with transfer/disk/deadline limits. No hours estimate is certified from the old screening.
