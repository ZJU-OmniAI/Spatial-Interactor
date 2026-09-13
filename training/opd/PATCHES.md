# EasyR1 modifications

The vendored EasyR1 tree is based on commit
`dd71bbd252694f5f850213eec15795b6b88d9fea`. The main Spatial-Interactor
changes are:

- `verl/trainer/config.py`: OPD, fixed-frame video, and rollout-audit options.
- `verl/utils/dataset.py`: plain and privileged prompts with shared media.
- `verl/trainer/opd.py`: same-prefix batches, CoT masks, top-k forward KL, and
  the process-weight schedule.
- `verl/trainer/ray_trainer.py`: on-policy teacher target collection and joint
  GRPO/OPD updates.
- `verl/workers/actor/dp_actor.py`: teacher top-k and student support scoring.
- `verl/workers/fsdp_workers.py`: latest pre-update same-policy teacher branch.
- `verl/trainer/rollout_audit.py`: sampled rollout and reward diagnostics.
- `verl/utils/dataset.py`, `verl/utils/vllm_utils.py`, and Qwen-VL model files:
  fixed 32-frame and Qwen2.5-VL/Qwen3-VL compatibility.

The process KL is summed over vocabulary before non-negative numerical stabilization.
`token_kl_clip` is `null` in the paper configuration.
No temporal module, adapter, or other model-architecture component is added.
