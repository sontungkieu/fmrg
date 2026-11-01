# Inverse paper-practical adapter v1

Scope: reconstructed-data technical comparison, not a recovered author runner or
full-paper result. Full launch remains disabled. Scientific source stays
`be485ba40f1d1c163b95ad61112aca6baa13ed22` and is never modified.

Primary evidence: [paper v3](https://arxiv.org/html/2604.27147v3), E.2/E.3,
Table E.1/E.2 and Eq.134. `paper_inverse_adapter.py` implements the following
explicit practical realization:

- Decode and degrade in pixel space; sum squared residuals as a positive loss.
- Convert noise time to data time with `t=1-sigma`.
- E optimizes the clean-latent endpoint (Table E.1), using per-inner-update gain
  `eta*(1-sigma_cur)*sigma_next`. Each of the five inner updates receives this
  gain. The resulting clean-latent displacement corrects the base flow step.
- The inverse J source already uses a unit gradient, velocity-norm rescaling and
  `eta*delta_t`. Delegate to that unchanged source with squared/pixel loss,
  flow_map2, nopt1 and no early stop. The reward sampler's Adam branch is outside
  this adapter's scope.
- Preserve the released shifted full scheduler grid, insert the exact E cutoff
  at data-time0.67, then perform a separate unguided map to data. J uses the full
  grid. Every actual node, effective cutoff and call count is recorded.
- Use the same frozen y/operator states, initial BF16 latent bytes, seed0 and
  paper class prompt in both tracks: dog for AFHQ and person for FFHQ.

The smoke compares this adapter to the untouched released sampler configured
with the same E50/J200, eta6/5/3, nopt5/1, squared/pixel loss and flow_map2.
For E, the released integer early-stop count is matched to the adapter's number
of guided intervals; its final interval still jumps to data under guidance.
That is an intended observed difference, not a silent cutoff equivalence claim.
J is an A/A control because its inverse formula already matches E.2.
The earlier norm/pixel/flow_map1 smoke is a separate historical profile and is
not used as a matched baseline.

## Interpretation limits

The practical appendix describes gains per update, whereas the generic
Algorithms5/6 and Eq.18 describe post-flow lookahead and explicit dt/nopt.
This implementation follows the practical clean-endpoint realization and
applies the E.2 gain per inner update, without an additional dt/nopt. It does
not claim a literal implementation of every generic pseudocode operation.
Author grid nodes, cutoff discretization and this practical/generic mapping
are not recovered from an executable author contract. The chosen policy is
versioned here; source-style E endpoint optimization and gain-after-inner-loop
are distinct from this gain-inside-inner-loop realization.

This comparison changes known paper settings together, not one variable at a
time. It is not a causal ablation or an efficacy test. No retuning follows
viewing these smoke results. Actual compute replaces NFE labels in the report.
Both float predictions and PNGs are saved. LPIPS256 and the source-style224
bilinear metric are named separately. Author-exact data, tuning, baselines,
reward/VLM guidance and full allocation remain separate contracts.
