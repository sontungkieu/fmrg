# GenEval runtime and reconstructed inputs

Read [PREPARATION_HANDOFF.md](PREPARATION_HANDOFF.md) for immutable HF downloads,
checksums, restore and company canary commands. The overlay inherits the original
core runtime at `/opt/fmrg`; it installs separately at `/opt/fmrg-geneval`.

The adapted Torch2.7/cu128/MMCV1.7.1 runtime passed native forward/backward/NMS,
R50/Swin-S detector and official color-classifier canaries on Modal B200. Model
keys/shapes match both checkpoints. OpenAI CLIP keeps its original loader and
QuickGELU; loading/evaluation denies TCP connections. These are technical checks,
not a paper score, company qualification or numerical parity with Torch1.12.

The user-authorized reconstructed AFHQ-Dog/FFHQ split has 50 tuning and 1000
evaluation images per dataset, pinned sources/hashes and original FFHQ attribution.
Skywork weights are mirrored after authorized gated access; its paper reward
adapter remains pending. See [CONTRACT_UPDATE.json](CONTRACT_UPDATE.json) for the
current boundaries. Full launch remains disabled.

`build_overlay.py`, `preflight.py`, `runtime_probe.py`, `prepare_additions.py` and
`publish_update.py` are remote preparation sources. Do not execute installation
or upload workers on the company node. The company toolkit contains the offline
adapter, canary, restore/download helpers and evidence checks. Operational Modal
launchers and credentials are excluded from this repository/toolkit.

`strict_evidence.py` normalizes official result identities and rejects incomplete,
duplicate, failed or mismatched cells before six-category GenEval aggregation.
Its inverse split validator distinguishes image-only preparation from a full
measurement freeze. `tuning-plan-draft.json` is a proposed independent protocol,
not recovered author settings or authorization to run it.

Run the standard-library helper tests with an existing Python interpreter:

```sh
python -B -m unittest -v test_preparation.py
```

The native C++17 build patch is in `patches/`; no numerical kernel was changed.
Code/source/runtime/asset identities remain separate. Keep a new receipt/output
per company attempt and retain failures explicitly.
