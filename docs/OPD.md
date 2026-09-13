# On-Policy Distillation

For an input `x = (video, question)`, the current policy samples `G=8`
responses. Each response contains a reasoning paragraph followed by a compact
answer. The student sees only `x`.

For every sampled response and every student prefix, two distributions are
evaluated on the same response history:

- plain branch: `x + prefix -> p_student`
- privileged branch: `x + trace + prefix -> p_teacher`

The teacher is a stop-gradient, latest pre-update snapshot. During the actor
update, gradients flow only through the current plain branch. The teacher is
not an EMA model and does not read the reference answer. Teacher forcing keeps
the two branches on exactly the same student-generated prefixes.

OPD computes teacher-to-student forward KL on the teacher's top-100 token
support. Only reasoning tokens before `Answer:` receive this process loss;
answer tokens and special tokens are masked. The masked process term uses the
same valid-response-token reduction as the rollout objective. The vocabulary
contributions are summed before non-negative numerical stabilization. The
optional upper cap is disabled in the paper configuration.

The optimization objective is:

```text
L_OPD = L_GRPO + 0.01 L_KL-reference + lambda_t L_process
```

`lambda_t` is 0.05 for the first 100 optimizer steps, then decays linearly to
zero over 1,100 steps. The independent frozen SFT reference policy is used only
for conventional policy KL and must not be confused with the privileged
teacher.

Final-answer rewards are verifiable:

- exact option reward for multiple-choice tasks;
- continuous relative-accuracy reward for numeric tasks;
- mean field accuracy for structured `straight/angle/path` tasks;
- total reward `0.9 * task_reward + 0.1 * format_reward`.

GRPO standardizes rewards within each rollout group. The default rollout batch
contains 16 questions and therefore 128 trajectories per optimizer step.
This continuous training reward is distinct from the threshold-aggregated MRA
used by official benchmark evaluation.

At inference, the annotator, trace, privileged branch, reward function, and
reference policy are removed. The trained model receives the same 32-frame
video and question interface as the SFT checkpoint.

## Important implementation files

- `training/opd/EasyR1/verl/trainer/opd.py`: CoT masks, paired scoring, KL,
  and coefficient schedule.
- `training/opd/EasyR1/verl/trainer/ray_trainer.py`: on-policy integration.
- `training/opd/EasyR1/verl/workers/fsdp_workers.py`: same-policy teacher
  target computation.
- `training/opd/rewards/spatial_video_reward.py`: verifiable rewards.
- `training/opd/EasyR1/verl/trainer/rollout_audit.py`: sampled rollout,
  reward, and process-loss diagnostics.
