# Learning-rate continuation and FiLM checkpoint evaluation

Historical September 16, 2026. Continue the same shared-R16 four-arm 6k
checkpoints to 10k total updates, restoring optimizer and scheduler state.
Only the LR schedule changes: 3e-4 at 6k decreases linearly to 3e-5 at 10k,
with no second warmup. Interruptions preserve the absolute-step schedule.

| Arm at 10k | GPT-2 PPL | Generated samples | Historical sampler |
|---|---:|---:|---|
| SS | 117.119 | 2 | O: unoptimized |
| FD | 94.810 | 2 | O: unoptimized |
| DF | 82.190 | 2 | O: unoptimized |
| DD | 103.579 | 2 | O: unoptimized |

Eight generated samples total, two seeds (91001/91002) per arm, length 1024,
batch 1, 1,000 reverse transitions plus cleanup. Scoring is token-weighted
GPT-2-large float32 PPL through the first nonleading EOS. These runs used an
isolated pre-speed-fix snapshot, even though V1 development overlapped in time.
A later checkout's default sampler does not relabel this historical result.

No matched two-sample constant-LR 10k baseline was generated for this screen;
the earlier one-sample baseline is not an equally precise control. Do not claim
a reliable schedule improvement from these measurements.

This milestone also preserves the FiLM 2k/4k/6k evaluation launcher. Its
one-sample results and unoptimized status are recorded in
`experiments/03-endpoints-and-conditioners.md`; the 7k result was reused, not
a second generation. Source paths are configurable/aliased for collaborator
use, with private email and machine restrictions removed.
