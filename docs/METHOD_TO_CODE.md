# Paper-to-code map

This page maps each method component in the paper to its executable
implementation. No model architecture is added or replaced.

| Paper component | Implementation |
| --- | --- |
| LSI-108K curriculum | `data_preparation/assign_curriculum_levels.py` |
| L1 + L2 and Public-80K SFT mixture | `data_preparation/build_sft_mixture.py` |
| Standard full-parameter SFT | `training/sft/train_sft.sh` |
| One question-independent trace per video | `data_generation/build_trace_manifest.py` |
| Frozen annotator and ordered segment descriptions | `data_generation/generate_privileged_traces.py` |
| Selected RoomTour-L3 + long-horizon VSTI | `data_preparation/build_opd_dataset.py` |
| Student on-policy rollouts with group size 8 | `training/opd/EasyR1/verl/trainer/ray_trainer.py` |
| Same-prefix privileged self-teacher | `training/opd/EasyR1/verl/workers/fsdp_workers.py` |
| CoT mask and teacher-to-student top-k forward KL | `training/opd/EasyR1/verl/trainer/opd.py` |
| Teacher/student token distributions on shared support | `training/opd/EasyR1/verl/workers/actor/dp_actor.py` |
| Verifiable final-answer reward | `training/opd/rewards/spatial_video_reward.py` |
| Joint GRPO, reference KL, and OPD launch | `training/opd/scripts/train_opd.sh` |
| Answer-reward-only GRPO ablation | `training/opd/scripts/train_grpo.sh` |

## Information flow

The plain training input is `x = (32-frame video, question)`. The offline trace
annotator receives only the 32 chronological frames, divided into four fixed
intervals; it never receives the question, answer, or reward target. The data
builder stores the plain prompt and a separate training-only privileged prompt.

At each update, the pre-update policy samples the student responses. The same
frozen pre-update parameters then score every sampled prefix twice: once with
the plain prompt and once with the privileged trace. The privileged branch is
detached and supplies a teacher distribution; it does not generate a separate
answer. Forward KL is applied only before the `Answer:` boundary. Final answer
tokens are supervised by verifiable GRPO rewards, not by OPD. Non-CoT
positions are zero-masked, and the process term follows the actor's valid-token
reduction.

The reference policy is an independent frozen SFT checkpoint used only for the
standard KL regularizer. It is not the privileged teacher. At inference, trace
annotation, the privileged branch, rewards, and the reference policy are all
removed; the model receives only the original video and question.

## Data protocol

- Full LSI-108K: L1 (15,109) + L2 (69,487) + L3 (22,922). The reported SFT
  run uses 13,109 single-transition L1 records and all L2 records, plus VSI
  (50,000), MindCube (10,000), and VSTI (20,000).
- OPD/GRPO: 10,712 RoomTour-L3 + 10,783 VSTI camera-movement and
  camera-displacement training records; 417 video-disjoint records are used
  for validation.
- OPD train/validation splitting is deterministic and video-disjoint.
- Prepared media paths are stored relative to `MEDIA_ROOT`; no machine path is
  written into the released training records.
