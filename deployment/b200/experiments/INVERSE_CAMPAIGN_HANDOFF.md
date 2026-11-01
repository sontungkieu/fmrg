# Reconstructed inverse E/J campaign — 2026-11-02

## Material Passport

Task: reproduce the published high-NFE inverse E/J configurations on the
user-approved reconstructed split. Evidence: paper v3 Appendix E.2/E.3,
pinned scientific source `be485ba40f1d1c163b95ad61112aca6baa13ed22`,
10 formula CPU tests and the independently verified 12-pair B200 smoke.
This contract does not cover reward/VLM/style or all-paper reproduction.

## Scientific decisions now sealed

[INVERSE_RECONSTRUCTED_CONTRACT.json](INVERSE_RECONSTRUCTED_CONTRACT.json)
is the machine-readable specification. It transfers the published E/J settings;
there is no new search, selection cost or post-smoke retuning. The 50-image
tuning partitions are used only for technical qualification. All 1,000 evaluation
IDs per dataset remain fixed. Author-exact IDs are replaced by the explicitly
authorized reconstructed design, not claimed recovered.

The practical adapter remains byte-identical to the qualified implementation.
E updates the clean-latent endpoint using its practical gain at each inner
update; J delegates to the unchanged inverse branch. The shifted full schedule,
inserted cutoff and separate unguided E completion are explicit conventions.
The practical appendix and generic post-flow pseudocode are not asserted
equivalent. See [the interpretation audit](PAPER_FORMULA_CONTRACT.md) and
[the paper](https://arxiv.org/html/2604.27147v3#A5.SS2).

One track produces 12,000 outputs: 2 datasets × 3 tasks × 2 methods × 1,000 IDs,
seed0. It does not duplicate the released comparison track or reproduce the
low-NFE curves. Same seed0 latent bytes per image are intentional; observation
noise is the separately frozen per-image/task measurement.

## Execution and acceptance

Use only the existing company GPUs0–3. Initial authorization is ten hours,
starting 2026-11-01 17:29:14 UTC; hard deadline 2026-11-02 03:29:14 UTC
(10:29:14 Asia/Saigon). Additional GPUs and extensions need concrete allocation.
No Modal or paid-service call is allowed; the cumulative USD10 cap is retained.

The tuning-only J8 probe passed batch1/4/8/16, with batch16 peak38.57GiB and
an extrapolated J200 cost20.42 seconds/image. These are throughput estimates,
not evaluation results. Full-config batch qualification must cover E50/J200,
all three operators and both class prompts before evaluation launch. Batch
size16 is an execution profile, not a bitwise-equivalence claim.

The initial existing four-GPU group runs the E-only stage (6,000 outputs).
The J-only stage (6,000 outputs) is prepared separately for the additional
four-GPU group when its access/allocation is supplied. Method filters precede
sharding and the supervisor checks the exact expected stage IDs. Neither stage
alone is reported as the complete12,000-output campaign.

`inverse_campaign.py` validates split identities and disjointness, freezes
12,000 content-addressed cells, verifies operator/source pins and copies the
actual qualified reference grids. It does not generate observations or images.

`run_inverse_campaign.py` rechecks input, observation file/payload, operator
state/config and model/source identities. It records actual grid/gradient/call
counts, RNG/latent hashes and output hashes. Each completed cell contains PNG,
float prediction, latent and a final `record.json`. Failed and partial cells
are preserved. A resume may reuse only exact contract/runner/cell identities
with verified artifacts; ambiguous duplicate attempts or corrupt/partial cells
are rejected. No automatic retry changes batching, seeds or settings.

`start_inverse_campaign.py` starts a durable supervisor and four independently
bounded workers. The supervisor records PIDs, exit codes and completion state;
`inspect_inverse_campaign.py` reads status without interrupting/replaying work.
Closing the bridge does not stop these detached workers. Workers stop at the
allocation deadline and retain completed cells. `GENERATION_COMPLETE` alone
does not mean metrics or scientific evaluation are complete.

`score_inverse_campaign.py` rejects missing/duplicate IDs before scoring each
complete 1,000-image group. It computes released-CLI PNG PSNR/SSIM/LPIPS and
separate native256 LPIPS, dataset FID/KID, and deterministic image bootstrap
CIs. CPU/GPU scoring backend is recorded. Dataset FID is not averaged over
images and no FID CI/equivalence claim is made. The explicit `--canary` path
uses only16 tuning outputs and never emits a dataset score.

The original handoffs/contracts retain their historical `full_launch_allowed=false`.
Only a new inverse-specific allocation manifest with a matching full-config
qualification receipt may set it true for this bounded campaign. That flag
does not authorize reward/VLM/style, new allocation, a new search or all-paper
launch. Newer user scope/allocation instructions govern the new campaign.

## Evidence and failures to retain

Preparation first caught the manifest's real split label `evaluation` rather
than `eval`; this was corrected without changing IDs/data. Another prelaunch
hash check rejected a Windows-CRLF hash of an LF-transferred allocation file.
Both failures occurred before neural qualification/launch and their bridge
receipts are retained. Hashes are computed over the actual transported bytes.

The first full-config qualification then found only the two smoke PNGs had
been extracted at the old split root; it stopped before any sampling (0 outputs).
The full archive was already staged and verified. Recovery extracts all2,100
images into a new root without changing hashes/IDs, and preflights every image
and observation file before loading the model. Old split paths and failed
qualification logs are preserved; the allocation deadline is not reset.

Before reporting final success: verify the expected unique12,000 cells, raw
artifacts and exact1,000 IDs in every group; then collect scorer outputs and
report mean/CI, actual compute and all missing/failed cells. Reconstructed
data, seed0, practical/generic ambiguity, adapted runtime and absent baselines
remain limits on comparison with published scores, not hidden provenance.
