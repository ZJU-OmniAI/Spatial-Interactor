# Reproducibility checklist

## Paper data protocol

The released curriculum assignment was checked against the final 107,518-row
LSI-108K index:

| Split | Rows | Curriculum role |
| --- | ---: | --- |
| L1 | 15,109 | Full release |
| L2 | 69,487 | SFT |
| L3 | 22,922 | Long-horizon candidate pool |
| L1 + L2 | 84,596 | Full local-transition curriculum |
| Reported SFT subset | 82,596 | 13,109 L1 + 69,487 L2 |

SFT adds Public-80K: VSI 50,000, MindCube 10,000, and VSTI 20,000. The online
stage selects 10,912 RoomTour-L3 and 11,000 VSTI camera-movement and
camera-displacement records before a video-level split, yielding 21,495
training and 417 validation records.

## Fixed defaults

- SFT: one epoch, seed 42, global batch 64, peak learning rate `1e-5`;
- OPD/GRPO: one epoch, seed 1, global batch 16, learning rate `5e-7`;
- deterministic OPD data split seed: `20260712`;
- video input: 32 frames;
- rollout group size: 8;
- rollout and actor global batch: 16 questions;
- maximum response length: 384 tokens;
- reference-policy KL coefficient: 0.01;
- OPD coefficient: 0.05 for 100 steps, then linearly decayed over 1,100 steps;
- OPD support: teacher top 100 tokens at temperature 1;
- OPD mask: reasoning tokens only, ending before `Answer:`;
- task/format reward weights: 0.9/0.1.

The privileged teacher is the stop-gradient pre-update policy evaluated on the
same student-generated prefixes. It is not an EMA model and never receives the
reference answer. Pointwise vocabulary clipping is disabled. Vocabulary
contributions are summed before non-negative numerical stabilization, and the
optional upper cap is disabled.

## Release checks

Run the CPU checks from the repository root:

```bash
bash scripts/check_release.sh
```

To include the OPD objective tests, point `OPD_PYTHON` to an environment with
the tested PyTorch dependencies:

```bash
OPD_PYTHON=/absolute/path/to/easyr1-python bash scripts/check_release.sh
```

Before launching a cluster job, use `CHECK_ONLY=1` with either training script.
The launchers validate paths and print the effective protocol without starting
Ray or allocating model weights.

SFT writes trainer logs and checkpoints under `OUTPUT_DIR`. OPD/GRPO writes
TensorBoard scalars, checkpoints, validation metrics, and sampled rollout audits
under `OUTPUT_ROOT`. The rollout audit records generated reasoning, final answer,
task reward, format reward, and process loss for reward-hacking inspection.

Model checkpoints, source media, public benchmark data, and generated
annotations are distributed separately from the code repository.
