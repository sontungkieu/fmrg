# Inverse E/J campaign: launch checkpoint — 2026-11-02

**E stage launched:6,000 evaluation outputs on the existing4 B200 GPUs0–3.**
At the recorded snapshot all4 workers were alive and loading the model; no
evaluation image had yet been independently collected. This is a launch
checkpoint, not completed evaluation or reproduced paper scores.

The whole campaign is12,000 outputs (E6,000 plus J6,000), using the fixed
reconstructed1,000 evaluation IDs per dataset,3 tasks and seed0. The E stage is
authorized separately from the pending J stage. J awaits the additional
cluster route/allocation. Reward/VLM/style and all-paper launch remain outside
this manifest. See the [sealed contract](../../experiments/INVERSE_RECONSTRUCTED_CONTRACT.json)
and [execution handoff](../../experiments/INVERSE_CAMPAIGN_HANDOFF.md).

Before E launch, all8,406 input files (3,628,076,264 bytes) passed hash checks,
including2,100 restored PNGs and6,300 frozen observations.9/9 CPU tests passed.
Full-config tuning qualification produced128/128 outputs on4 GPUs: E50/J200,
all3 operators and both prompts. The separate CPU scorer canary passed on16
tuning outputs and emitted no dataset score. An independent verifier then
checked all128 stored containers/payloads, BF16 latent bytes, PNG quantization,
grids and actual call counts. These data were not used for tuning parameters.

Full-step batch16 timing: E2.581–2.689seconds/image, J19.767–19.963seconds/image
across the qualified groups; maximum allocated38.6004GiB. These are startup-free
batch timings, not a guarantee for every evaluation image. E nominal100NFEs
uses78 actual transformer invocations per image under the explicit cutoff
convention; J nominal400 uses400. GPU batch arithmetic is not claimed bitwise
equivalent to single-image execution.

The E supervisor runs detached. Its new server directory is
`fmrg_all/runs/fmrg-inverse-E-evaluation-20261102-v1/`; transition evidence is in
`fmrg_all/runs/fmrg-inverse-advance-20261102-v1/`. The hard deadline remains
**10:29:14 Asia/Saigon on02/11/2026**, not reset by preparation failures. Completed
cells survive disconnect/deadline; do not replay a launch. Monitor the existing
supervisor/worker logs and exact artifacts. Metric evaluation must reject any
group missing one of its1,000 expected IDs.

Only the inverse E-specific launch manifest has `full_launch_allowed=true`.
Prior handoffs and technical canaries retain false. No Modal or paid-service
call was made; the cumulative paid cap remains USD10. See
[launch-E.json](launch-E.json) for hashes and retained failure classifications.
