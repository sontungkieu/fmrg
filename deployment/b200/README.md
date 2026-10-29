# Portable B200 setup and company replay

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
