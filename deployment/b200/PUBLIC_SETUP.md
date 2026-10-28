# FMRG portable runtime and pilot assets

This bundle supports offline setup on an allocated NVIDIA B200 Linux x86_64 node.
It contains a dependency runtime at `/opt/fmrg`, separately pinned model/input
assets, pilot source, and pilot outputs. No company node has been qualified yet.

Use the immutable publication manifest and setup helper linked in `HANDOFF.md`.
Download through the operator's approved HF proxy, verify the helper and manifest
SHA-256, then call `setup_public.sh MANIFEST_URL MANIFEST_SHA DOWNLOAD_DIR
NEW_DATA_DIR NEW_RUN_DIR`. These must be approved absolute paths; the download
directory and parent directories must exist. Setup requires write access to
`/opt`, refuses an existing `/opt/fmrg` or data/output destination, and runs the
imports/kernel/tiny/full technical gates. Keep scheduler-provided GPU visibility.
Do not run outside the allocation or install dependencies from PyPI on the node.
Allow roughly 100 GiB free for downloaded archives and extraction; full datasets
and experiment outputs need additional storage. Verify actual capacity first.

The public runtime repacks the qualified dependency archive, removing cloud
launchers and local infrastructure paths in its tools directory. Dependency,
interpreter and upstream source file bytes are preserved. Its archive SHA differs
from the historical qualification archive; see `manifests/archive-audit.json`.
The source upstream is `jrrhuang/fmrg` at
`be485ba40f1d1c163b95ad61112aca6baa13ed22`. The runtime uses Python 3.11.14,
Torch 2.7.1+cu128, CUDA 12.8 and SDPA. Ubuntu22/Modal B200 was qualified;
Ubuntu24 and the company node still require their own receipts.
Host prerequisites include `curl`, `tar`, `sha256sum`, Python 3 for validation,
an NVIDIA driver supporting CUDA 12.8, and Ubuntu `libgl1`/`libglib2.0-0` support
libraries (the prior clean-restore container supplied these). Missing host
libraries must be resolved through the operator's approved host/container setup;
the helper does not run apt or change the system driver.

Assets retain their original immutable bytes. Their embedded manifest calls the
original export private; publication was subsequently explicitly authorized on
2026-10-28. This label is historical and is not the current HF visibility.
The bundle includes FLUX.1-dev, flow-map LoRAs, CLIP ViT-H, three input PNGs and
six frozen measurements. It excludes full AFHQ/FFHQ datasets and the full paper
reward/evaluation ensemble. Public availability does not replace upstream terms:
the original FLUX license is included in the assets and under `licenses/`.
The flow-map upstream has no separate license metadata. Dependencies retain
their individual license files inside the runtime. Model authors retain rights.
Rights to use FLUX.1-dev and its derivatives are granted directly by Black
Forest Labs to recipients under the included license.

> The FLUX.1 [dev] Model is licensed by Black Forest Labs Inc. under the FLUX.1
> [dev] Non-Commercial License. Copyright Black Forest Labs Inc.
> IN NO EVENT SHALL BLACK FOREST LABS INC. BE LIABLE FOR ANY CLAIM, DAMAGES OR
> OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
> FROM, OUT OF OR IN CONNECTION WITH USE OF THIS MODEL.

Pilot results: G has 288 configs and eight oracle checks; I/J/R have 72/96/24
outputs. R is CLIP-only. These are small screening runs, not full reproduction
of the paper tables. All generated output images are AI-generated. See
`results/PILOT_RESULTS.md` and the raw results archive for scope and exact metrics.
J is not certified bitwise reproducible; the prior clean-restore smoke differed
by up to 7/255 per PNG channel. The reward counter `autograd_calls` hooks
`torch.autograd.grad` only; zero does not mean no `Tensor.backward()` execution.
Resolve loss, schedule, data split/tuning and evaluator contracts before a full
scientific run. The helper launches technical gates only, never scientific jobs.
