# Inverse practical-paper formula: matched smoke

**12 pairs /24 outputs COMPLETE.** Two reconstructed tuning images,one seed; no full launch or efficacy claim.

| Dataset | Task | Method | PSNR released | PSNR adapter | Delta | LPIPS256 released | LPIPS256 adapter | CLI LPIPS released | CLI LPIPS adapter |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| AFHQ-Dog | sr4 | E | 27.317730 | 27.089378 | -0.228352 | 0.288645 | 0.311587 | 0.213327 | 0.239606 |
| AFHQ-Dog | sr4 | J | 27.256775 | 27.246620 | -0.010155 | 0.277689 | 0.280084 | 0.199933 | 0.200893 |
| AFHQ-Dog | motion61 | E | 22.925564 | 21.892254 | -1.033310 | 0.433248 | 0.449796 | 0.358819 | 0.382210 |
| AFHQ-Dog | motion61 | J | 24.143996 | 24.152851 | +0.008855 | 0.375988 | 0.366276 | 0.308589 | 0.300543 |
| AFHQ-Dog | mask64 | E | 24.742086 | 25.889146 | +1.147060 | 0.283358 | 0.264819 | 0.198519 | 0.173612 |
| AFHQ-Dog | mask64 | J | 27.399497 | 26.587180 | -0.812317 | 0.250092 | 0.253497 | 0.155119 | 0.159227 |
| FFHQ | sr4 | E | 32.386028 | 31.798889 | -0.587139 | 0.234213 | 0.260150 | 0.139833 | 0.167271 |
| FFHQ | sr4 | J | 32.099474 | 32.033672 | -0.065802 | 0.193051 | 0.200705 | 0.107815 | 0.112494 |
| FFHQ | motion61 | E | 20.098517 | 17.277277 | -2.821241 | 0.534268 | 0.567308 | 0.491455 | 0.521997 |
| FFHQ | motion61 | J | 29.398886 | 29.398897 | +0.000011 | 0.257901 | 0.259293 | 0.175223 | 0.177756 |
| FFHQ | mask64 | E | 27.959465 | 29.644365 | +1.684900 | 0.286881 | 0.239235 | 0.189453 | 0.139659 |
| FFHQ | mask64 | J | 34.646394 | 34.565180 | -0.081214 | 0.140707 | 0.139034 | 0.067882 | 0.068557 |

J is an A/A control of the unchanged normalized inverse-J source: {"pairs": 6, "bitwise_equal": 0, "max_abs_prediction_diff": 1.046875, "max_abs_delta_psnr": 0.8123170205938983, "equivalence_claim": false}.

Sampling seconds: {"released": 453.5924711744301, "paper": 449.03025478310883}; phase execution total 1017.571s.

Measured calls: {"released": {"predict_forward": 2868, "transformer_forward": 2868, "observed_block_forward_including_recompute": 231876, "vae_decode": 2382, "autograd_calls": 2370}, "paper": {"predict_forward": 2868, "transformer_forward": 2868, "observed_block_forward_including_recompute": 231876, "vae_decode": 2352, "autograd_calls": 2340}}.

Both tracks use squared/pixel loss,flow_map2,paper class prompts and shared latent/RNG/input bytes. E uses50 nominal grid intervals,eta6,nopt5; J200/eta5(AFHQ) or3(FFHQ)/nopt1.

The reconstructed cutoff policy preserves the shifted full grid and inserts data-time0.67. Released E matches the integer guided count but still guides its jump to data; adapter E finishes with a separate unguided map.

E.2 practical gain per clean-endpoint inner update; no extra dt/nopt. Full shifted scheduler grid clipped with exact .67 boundary. Generic Algorithms5/6 ordering not certified.

All24 float/file/payload/PNG identities; shared latent+RNG+input; independent PSNR <=1e-4dB; E gain/stop trace; actual calls verified. GPU float LPIPS not re-evaluated; separate literal CLI PNG/PIL protocol independently rescored on CPU.

GPU LPIPS256/float224 use[-1,1]. Additional CLI LPIPS is independently rescored on CPU with the actual released PNG/PIL224/ToTensor[0,1] recipe,no Normalize. These protocols must not be mixed. No parameter was retuned after viewing the smoke.

Full reproduction remains unqualified; see the interpretation contract and report limitations.

## Collection and next handoff

Collected through the reassigned Terminal 8 on 2026-11-02, on the same company
instance. No sampling was repeated. The CPU collector completed with wrapper
exit 0, runner `ok:true`, payload exit 0. No Modal call or paid service was used.
The cumulative paid cap remains USD 10; this is not a fresh provider billing read.

All 24 stored float/latent containers and payloads, record/PNG hashes, shared
latent/RNG identities, output quantization, call counts and E gain/cutoff traces
passed the collector. Maximum independently recomputed float PSNR discrepancy:
**0.000001103086965770217 dB**, below the frozen 0.0001 dB tolerance. The entire
6,300-observation collection was not rehashed by this output collector.

The public JSON replaces the GPU UUID with its SHA256. Original company report
SHA256: `3c0721426c854b196b0354be39e3f6cfbf3f59bec3f418b450975280c85f3cae`
(38,035 bytes). Public metrics and all 24 artifact identities are retained.

E improves PSNR in 2/6 cells and decreases it in 4/6. This is descriptive only:
two images and one seed do not establish efficacy. J is an unchanged-source A/A
control, and its 0/6 byte-equal pairs prevent an equivalence claim or attribution
of all E differences to the adapter. No settings were retuned after inspection.

Do not replay the completed smoke or launch a matrix. Preserve the separate
[practical interpretation contract](../../experiments/PAPER_FORMULA_CONTRACT.md).
Full reproduction remains blocked on author-exact data (current splits are
reconstructed), practical/generic formula and author-grid interpretation,
paper tuning/seeds/baselines/compute and full allocation. Reward/VLM integrated
guidance, renoise-c, style references/prompts and GenEval numerical equivalence
remain outside this inverse-only qualification. `full_launch_allowed=false`.

On the company node, the full contact sheet and raw report remain in
`fmrg_all/runs/fmrg-paper-compare-20261101-v1-report/`; the two sampling phases
remain in sibling `...-v1-first/` and `...-v1-rest/` directories. Source/runtime,
frozen observations, prior attempts and all original outputs are unchanged.
