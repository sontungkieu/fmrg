# Conditions before full paper reproduction

The Modal pilot is not reproduction of the paper's full image benchmark. It
uses three inverse-problem images and two reward prompts. Freeze the following
contracts before launching a large benchmark, and retain the resolved versions
in every run receipt.

1. **Data and tuning.** Pin image IDs, hashes, preprocessing and separate tuning
   and evaluation splits for AFHQ-Dog and FFHQ. The paper describes 50 held-out
   tuning images and 1000 evaluation images per dataset. Pilot inputs do not
   replace these sets. Keep measurement noise sigma 0.03 and record operator,
   noise seeds and frozen measurement hashes.
2. **Loss.** Audit the equation and executable source separately: Eq.134 gives
   squared pixel residual; the source CLI defaults to a norm and its latent loss
   uses a pseudo-inverse-weighted residual. Pilot I uses pixel/norm, strength10,
   n_opt3. Do not silently substitute that contract for tuned paper tables or
   change the pilot hyperparameters to match old outputs.
3. **Time and compute.** Paper data-time increases 0 to1; the sampler's noise
   time decreases 1 to0 and early_stop is an integer index. Resolve cutoffs on
   the actual grid. Do not translate t_stop0.67 as 0.67 times steps without
   checking this mapping. Count model forwards, checkpoint recomputation,
   decoder, backward, reward and K3 selection work as well as reported NFE.
4. **Candidate table settings.** The historical TableE.2 transcription is
   provisional: E steps50/NFE100, eta6/n_opt5; J steps200/NFE400/n_opt1, eta5 for
   AFHQ or eta3 for FFHQ, and higher-NFE flow_map2. Check the pinned paper and
   source/loss/schedule contract before treating this as executable reproduction.
   Baselines need identical splits and comparable tuning budgets.
5. **Evaluators.** Stage and pin full reward ensemble, VLM/GenEval, LPIPS and FID
   dependencies/weights through the approved HF channel. R is CLIP-only; package
   imports do not prove full reward weights/backward work. Do not let evaluators
   download from unapproved hosts during a job. Validate range/preprocessing and
   use sufficient samples for the requested metrics.
6. **Interpretation.** J has no bitwise parity certificate: the previous
   clean-restore smoke differed by up to7/255 in PNG channels. Predeclare
   comparison tolerances and report differences. R's autograd_calls counter
   tracks autograd.grad only, whereas the official reward sampler uses
   Tensor.backward. Zero in that field is not absence of backpropagation.
   K3 has additional selection cost; it is not compute-matched to K1.
7. **Gaussian objective.** Eight continuous oracle checks and 288 discrete
   configs are diagnostics. Include integrated control energy when comparing
   the exact-control objective, and keep effective discrete cutoffs. Low-NFE
   Euler does not exactly reproduce the continuous terminal distribution.
8. **Execution and reporting.** Confirm allocation, deadlines and queue policy,
   budget the full matrix, and canary missing components before a large launch.
   Require expected unique outputs, raw metrics, resolved config and all
   identities. Missing/failed cells stay explicit, never zero-filled. Report
   technical qualification, screening, and paper-level results separately.

The pinned HF handoff is the download/setup source of truth. A company agent
must obtain the paper through an allowed route if an exact claim above needs
rechecking; absence of that route does not authorize a network fallback.
