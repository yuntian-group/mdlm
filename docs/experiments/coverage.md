# Curation coverage

Scope: 43 local commits above recovery baseline
`99891f9441d9aef7082a088963ae2970bc992563`, plus the 40 modified/untracked files
captured when curation began. Together these affect 107 distinct paths.

All implementation, configuration and test paths are retained. Machine details
were replaced by local environment variables and descriptive run aliases.
Operational status notes are consolidated below; redundant old patches are
superseded by the new commit series. No weights or private runtime logs are included.

The original checkout and its uncommitted work were left unchanged. New commits
group experiment development in chronological order, with related uncommitted
architecture support placed alongside the experiment that needs it. Original
commit IDs and dates are not presented as newly executed runs.

| Original path | Curated disposition |
|---|---|
| `AGENTS.md` | Retained reproducibility guidance; personal notification recipient removed. |
| `configs/lr_scheduler/linear_decay_continuation.yaml` | Retained; only privacy/portability edits where needed. |
| `configs/model/contextual-forest-endpoints-small.yaml` | Retained; only privacy/portability edits where needed. |
| `configs/model/contextual-forest-tiny.yaml` | Retained; only privacy/portability edits where needed. |
| `diffusion.py` | Retained; only privacy/portability edits where needed. |
| `docs/ccf-all-1k-five.md` | Consolidated into `docs/experiments/10-five-sample-generation.md`. |
| `docs/ccf-dd-option-experiments.md` | Retained and edited to separate settings, evidence, limitations and private operations. |
| `docs/ccf-eight-steps.md` | Consolidated into `docs/experiments/13-eight-step-sweep.md`. |
| `docs/ccf-endpoint-factors.md` | Retained and edited to separate settings, evidence, limitations and private operations. |
| `docs/ccf-fast-generation.md` | Retained and edited to separate settings, evidence, limitations and private operations. |
| `docs/ccf-full-sample-profile.md` | Consolidated into `docs/ccf-fast-generation.md`. |
| `docs/ccf-low-steps-checkpoint-sweep.md` | Consolidated into `docs/experiments/12-full-low-step-sweep.md`. |
| `docs/ccf-low-steps-twenty.md` | Consolidated into `docs/experiments/11-basic-low-step-evaluation.md`. |
| `docs/ccf-lr-decay.md` | Retained and edited to separate settings, evidence, limitations and private operations. |
| `docs/ccf-original-five-trend.md` | Consolidated into `docs/experiments/10-five-sample-generation.md`. |
| `docs/ccf-parallel-traversal-noise-tests.md` | Consolidated into `docs/ccf-fast-generation.md`. |
| `docs/ccf-quality-diagnosis-20260916.md` | Retained and edited to separate settings, evidence, limitations and private operations. |
| `docs/ccf-r8-dd-2k-more-seeds.md` | Consolidated into `docs/ccf-quality-diagnosis-20260916.md`. |
| `docs/ccf-restart-recovery.md` | Retained and edited to separate settings, evidence, limitations and private operations. |
| `docs/ccf-sampling-speed-audit-v2.md` | Consolidated into `docs/ccf-sampling-speed-fix-v2.md`. |
| `docs/ccf-sampling-speed-audit-v3.md` | Consolidated into `docs/ccf-fast-generation.md`. |
| `docs/ccf-sampling-speed-fix-v2.md` | Retained and edited to separate settings, evidence, limitations and private operations. |
| `docs/ccf-sampling-speed-fix.md` | Retained and edited to separate settings, evidence, limitations and private operations. |
| `docs/ccf-sampling-v3-gpu-tests.md` | Consolidated into `docs/ccf-fast-generation.md`. |
| `docs/ccf-sampling-v4-level-tests.md` | Consolidated into `docs/ccf-fast-generation.md`. |
| `docs/ccf-selected-five.md` | Consolidated into `docs/experiments/10-five-sample-generation.md`. |
| `docs/ccf-separate-checkpoint-evaluation.md` | Consolidated into `docs/experiments/05-separate-rank-screens.md`. |
| `docs/ccf-separate-rank16.md` | Consolidated into `docs/experiments/05-separate-rank-screens.md`. |
| `docs/ccf-verified-v3-profile.md` | Consolidated into `docs/ccf-fast-generation.md`. |
| `docs/patches/ccf-all-1k-five.patch` | Duplicate patch omitted; its code is retained in the new series. |
| `docs/patches/ccf-eight-steps.patch` | Duplicate patch omitted; its code is retained in the new series. |
| `docs/patches/ccf-fast-generation-entry.patch` | Duplicate patch omitted; its code is retained in the new series. |
| `docs/patches/ccf-low-steps-checkpoint-sweep.patch` | Duplicate patch omitted; its code is retained in the new series. |
| `docs/patches/ccf-low-steps-twenty.patch` | Duplicate patch omitted; its code is retained in the new series. |
| `docs/patches/ccf-original-five-trend.patch` | Duplicate patch omitted; its code is retained in the new series. |
| `docs/patches/ccf-sampling-speed-v1.patch` | Duplicate patch omitted; its code is retained in the new series. |
| `docs/patches/ccf-sampling-speed-v2.patch` | Duplicate patch omitted; its code is retained in the new series. |
| `docs/patches/ccf-selected-five.patch` | Duplicate patch omitted; its code is retained in the new series. |
| `docs/third-party/pytorch-v2.2.2-LICENSE.txt` | Retained; only privacy/portability edits where needed. |
| `evaluation/generation_harness.py` | Retained; only privacy/portability edits where needed. |
| `lr_schedules.py` | Retained; only privacy/portability edits where needed. |
| `models/directional_forest.py` | Retained; only privacy/portability edits where needed. |
| `models/structured_decoder.py` | Retained; only privacy/portability edits where needed. |
| `scripts/audit_ccf_sampling_speed_v2.py` | Retained; only privacy/portability edits where needed. |
| `scripts/audit_ccf_sampling_v3.py` | Retained; only privacy/portability edits where needed. |
| `scripts/audit_ccf_sampling_v4.py` | Retained; only privacy/portability edits where needed. |
| `scripts/ccf_separate_resume.py` | Retained; only privacy/portability edits where needed. |
| `scripts/continue_four_ccf_matched_3k_to_6k.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/continue_four_ccf_matched_6k_to_10k.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/continue_four_ccf_matched_to_3k.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/continue_four_ccf_shared_6k_to_10k_lr_decay.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/diagnose_ccf_objective.py` | Retained; only privacy/portability edits where needed. |
| `scripts/eval_original_mdlm_owt_ppl.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/evaluate_ccf_eight_steps.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/evaluate_ccf_film_checkpoints.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/evaluate_ccf_low_steps_sweep.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/evaluate_ccf_low_steps_twenty.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/evaluate_ccf_lr_decay_10k.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/evaluate_ccf_r8_dd_2k_more_seeds.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/evaluate_ccf_selected_five.py` | Retained; only privacy/portability edits where needed. |
| `scripts/evaluate_ccf_selected_five.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/evaluate_ccf_separate_checkpoints.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/evaluate_four_ccf_6k_7k_paired.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/evaluate_four_ccf_7k_8k.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/evaluate_four_ccf_step500.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/evaluate_mdlm_factorized_runner.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/export_structured_adapter.py` | Retained; only privacy/portability edits where needed. |
| `scripts/generate_four_ccf_matched.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/generate_original_mdlm_hf_1k.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/plot_ccf_r8_dd_convergence.py` | Retained; only privacy/portability edits where needed. |
| `scripts/profile_ccf_full_sample.py` | Retained; only privacy/portability edits where needed. |
| `scripts/profile_ccf_full_sample_gpu.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/profile_ccf_level_gpu.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/profile_ccf_verified_v3_gpu.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/run_ccf_dd_option_7k.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/run_ccf_separate_7k.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/run_generation_fast.py` | Retained; only privacy/portability edits where needed. |
| `scripts/score_text_files.py` | Retained; only privacy/portability edits where needed. |
| `scripts/summarize_ccf_6k_7k_paired.py` | Retained; only privacy/portability edits where needed. |
| `scripts/train_ccf_dd_film_7k.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/train_ccf_dd_topk256_7k.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/train_ccf_separate_7k.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/train_ccf_separate_rank16_7k.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/train_four_ccf_matched_1k.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/verify_ccf_sampling_optimization.py` | Retained; only privacy/portability edits where needed. |
| `scripts/verify_ccf_sampling_speed_gpu.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/verify_ccf_sampling_v2.py` | Retained; only privacy/portability edits where needed. |
| `scripts/verify_ccf_sampling_v2_gpu.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/verify_ccf_sampling_v3_gpu.py` | Retained; only privacy/portability edits where needed. |
| `scripts/verify_ccf_sampling_v3_gpu.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/verify_ccf_sampling_v4_gpu.sh` | Retained; only privacy/portability edits where needed. |
| `scripts/verify_ccf_sampling_v5_gpu.sh` | Retained; only privacy/portability edits where needed. |
| `structured_objective.py` | Retained; only privacy/portability edits where needed. |
| `structured_utils.py` | Retained; only privacy/portability edits where needed. |
| `tests/test_ccf_phase_profile.py` | Retained; only privacy/portability edits where needed. |
| `tests/test_ccf_sampling_optimization.py` | Retained; only privacy/portability edits where needed. |
| `tests/test_ccf_sampling_v3_experiments.py` | Retained; only privacy/portability edits where needed. |
| `tests/test_ccf_sampling_v4_experiments.py` | Retained; only privacy/portability edits where needed. |
| `tests/test_ccf_selected_five.py` | Retained; only privacy/portability edits where needed. |
| `tests/test_ccf_separate_resume.py` | Retained; only privacy/portability edits where needed. |
| `tests/test_contextual_forest_adapter.py` | Retained; only privacy/portability edits where needed. |
| `tests/test_export_structured_adapter.py` | Retained; only privacy/portability edits where needed. |
| `tests/test_factor_conditioner.py` | Retained; only privacy/portability edits where needed. |
| `tests/test_factor_embedding_modes.py` | Retained; only privacy/portability edits where needed. |
| `tests/test_fast_generation_entry.py` | Retained; only privacy/portability edits where needed. |
| `tests/test_lr_schedules.py` | Retained; only privacy/portability edits where needed. |
| `tests/test_score_text_files.py` | Retained; only privacy/portability edits where needed. |

Portability changes load the unchanged V1 reference by Git blob, replace a local V2
snapshot directory with the chosen checkout, and export the cache/code roots.
Production architecture, objectives, sampler math and scoring policy match the
captured source work. Saved results were not regenerated by these packaging edits.
