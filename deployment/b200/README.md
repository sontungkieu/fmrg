# Portable B200 setup and company replay

The [reconstructed inverse E/J campaign handoff](experiments/INVERSE_CAMPAIGN_HANDOFF.md)
and [sealed contract](experiments/INVERSE_RECONSTRUCTED_CONTRACT.json) define the
new user-authorized 12,000-output evaluation scope. They replace the old global
blockers for this scope with explicit reconstructed-data/practical-formula
conventions; launch still requires the matching batch qualification and bounded
company allocation. Reward/VLM/style and all-paper execution remain separate.
The [2026-11-02 launch checkpoint](results/inverse-campaign-20261102/REPORT.md)
records the E-only6,000-output stage on4 company B200s after128-output full-config
qualification, independent artifact verification and CPU scorer canary. J is
prepared separately for the additional cluster; launched is not completed.

The [inverse practical-paper adapter contract](experiments/PAPER_FORMULA_CONTRACT.md)
defines a separate E.2/E.3 reconstruction and a matched 12-pair smoke against
the untouched released inverse sampler. Its tests cover analytic gradients,
time conversion and a separate unguided cutoff completion. Practical/generic
algorithm ambiguities are explicit; this does not enable full execution.

The [matched comparison report](results/paper-compare-20261101/REPORT.md),
collected on 2026-11-02, verifies all 12 pairs / 24 saved outputs without
repeating sampling. Independent CPU PSNR and released-CLI LPIPS rescoring,
artifact hashes and E gain/cutoff traces passed. J A/A controls are not bitwise
equal (maximum absolute PSNR delta 0.812317 dB); this technical comparison
does not establish equivalence, efficacy or full-paper reproduction.

The [company frozen-input smoke report](results/frozen-smoke-20261101/REPORT.md)
records 12/12 completed E/J cells on 2026-11-01, with image/input hashes,
resolved settings, measured call counts and PSNR/LPIPS. This is an integrated
smoke on two reconstructed tuning images, not full paper reproduction.

The bounded [frozen-input E/J smoke runner](experiments/frozen_smoke.py) runs
one first cell, then the remaining 11 only after that cell completes. Both phases
share a 30-minute deadline and write to new output directories. It preserves the
released sampler, precreates seeded latent noise outside the observation guard,
uses verified frozen tensors/states and records images, PSNR,256px LPIPS VGG,
actual call counts and sigma grids. Its explicit norm/pixel loss and flow_map1
profile are smoke settings, not the paper-formula contract or an efficacy result.
Use only the authorized existing company GPU via the guarded bridge; no full launch.
The [collector](experiments/collect_frozen_smoke.py) checks all 12 cell identities,
image/record hashes, counters and phase completion before creating the report and
contact sheet. Metrics use floating-point predictions before PNG quantization;
the preview is separate from metric computation.

The [2026-11-01 GenEval/VLM/reconstructed-data update](geneval/PREPARATION_HANDOFF.md)
adds a separate evaluator overlay with Modal B200 technical qualification,
authorized Skywork weights and explicitly reconstructed image splits. It keeps
full execution disabled; company evaluator qualification is tracked separately
in its runtime and canary receipts.

The 2026-10-29 preparation is in
[full_paper/FULL_PAPER_HANDOFF.md](full_paper/FULL_PAPER_HANDOFF.md): audited
paper/source differences, immutable HF evaluator downloads, offline setup and
canaries. [CONTRACT.json](full_paper/CONTRACT.json) keeps full launch disabled.
The company screening is closed as user-reported replay completion, without
bitwise parity or established distribution equivalence. Do not rerun it simply
because this preparation has a newer handoff.

Start with [AGENT_PROMPT.md](AGENT_PROMPT.md) for a copy-paste assignment and
[HANDOFF_SETUP_AGENT.md](HANDOFF_SETUP_AGENT.md) for immutable HF downloads,
checksums and the setup command. The public distribution is
[codemaivanngu/fmrg-b200-portable](https://huggingface.co/datasets/codemaivanngu/fmrg-b200-portable).

The runtime and assets are hosted on HF; this Git repository contains small
source, documentation and evidence only. Company downloads must use the approved
HF proxy. No GitHub download or dependency installation is required on the node.

The scientific source inside the runtime remains upstream commit
`be485ba40f1d1c163b95ad61112aca6baa13ed22`. A newer Git commit containing this
deployment directory is a separate operational identity. The runtime's source
hash gate checks `/opt/fmrg/source`, not a modified Git checkout or this folder.

`source-manifest.json` records the original 13 files in the HF source export;
those bytes are preserved here. The manifest does not cover itself or the added
prompt, handoff, documentation and receipts; use the Git commit for those files.
`runtime/setup_public.sh` is the exact HF helper and installs at `/opt/fmrg`.
It runs imports/kernel/tiny/full technical gates and writes a setup summary.
It does not submit jobs to a company queue or run scientific experiments.

After technical qualification, the offline runners are
`experiments/gaussian.py`, `pilot_ip.py` (I/J) and `pilot_reward.py` (R).
They need the restored runtime wrapper and data paths described in the handoff.
I/J/R execute complete pilot grids, not single-cell canaries. G is CPU work.
Reference results are in [results/shortlist-20261026](results/shortlist-20261026).
Raw images/metrics are also available from the immutable HF results archive.

These are screening results, not full paper reproduction. Read
[PAPER_REPRODUCTION_GATES.md](PAPER_REPRODUCTION_GATES.md) before expanding the
benchmark. Company host/allocation/queue/proxy must be resolved from the current
environment; no historical hostname or GPU lease is supplied as authorization.
