# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    privileged_prompt_key: Optional[str] = None
    """optional enhanced prompt used only for OPD teacher re-scoring"""
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    video_nframes: Optional[int] = None
    """fixed uniform frame count; when set, takes precedence over video_fps"""
    video_max_frames: Optional[int] = None
    """maximum uniformly sampled frames when video_fps is used"""
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    assistant_prefill: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.format_prompt = get_abs_path(self.format_prompt, prompt="Format prompt file")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")


@dataclass
class OPDConfig:
    enabled: bool = False
    """enable privileged-context on-policy distillation"""
    loss_weight: float = 0.05
    """initial coefficient for the direct top-k distillation loss"""
    weight_hold_steps: int = 100
    """number of initial steps that keep loss_weight constant"""
    weight_decay_steps: int = 1100
    """linear decay window after weight_hold_steps"""
    top_k: int = 100
    """teacher-supported vocabulary size used for local FKL"""
    temperature: float = 1.0
    """temperature for teacher and student top-k distributions"""
    token_kl_clip: Optional[float] = None
    """optional upper cap applied after summing each non-negative token KL"""
    mask_special_tokens: bool = True
    """exclude tokenizer special tokens from teacher top-k support"""
    cot_open_tag: str = "<COT>"
    cot_close_tag: str = "</COT>"
    """only generated tokens strictly enclosed by these tags receive OPD loss"""
    allow_legacy_cot: bool = False
    """also accept complete COT: ... Answer: spans while migrating SFT output format"""
    response_starts_in_cot: bool = True
    """treat generated tokens before Answer: as COT when the prompt ends with a reasoning prefill"""
    require_nonempty_cot: bool = True
    """fail a batch when no complete COT span is present instead of using reward-only GRPO"""


@dataclass
class RolloutAuditConfig:
    enabled: bool = False
    """write reward-hacking diagnostics and sampled group-level HTML reports"""
    log_freq: int = 10
    """write sampled audit pages every this many optimizer steps"""
    groups_per_step: int = 2
    """number of suspicious or low-signal rollout groups to preserve per audit step"""
    include_privileged_context: bool = True
    """include training-only visual descriptions in collapsible audit sections"""
    max_response_chars: int = 8000
    """maximum decoded characters stored for each audited response"""
    output_dir: Optional[str] = None
    """audit output directory; defaults to rollout_audit under the checkpoint directory"""

    def post_init(self):
        if self.log_freq <= 0:
            raise ValueError("rollout_audit.log_freq must be positive.")
        if self.groups_per_step < 0:
            raise ValueError("rollout_audit.groups_per_step cannot be negative.")
        if self.max_response_chars < 256:
            raise ValueError("rollout_audit.max_response_chars must be at least 256.")


@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    """discount factor for ppo gae advantage estimator"""
    lam: float = 1.0
    """lambda value for ppo gae advantage estimator"""
    adv_estimator: str = "grpo"
    """advantage estimator, support `gae`, `grpo`, `reinforce_plus_plus`, `remax`, `rloo`"""
    disable_kl: bool = False
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_penalty: str = "kl"
    """kl penalty type, support `kl`, `abs`, `mse`, `low_var_kl`, `full`"""
    kl_coef: float = 1e-3
    """kl coefficient"""
    kl_type: str = "fixed"
    """kl controller type, support `fixed`, `adaptive`"""
    kl_horizon: float = 10000.0
    """kl horizon for adaptive kl controller"""
    kl_target: float = 0.1
    """target kl for adaptive kl controller"""
    online_filtering: bool = False
    """use online filtering"""
    filter_key: str = "overall"
    """reward key for filtering samples"""
    filter_low: float = 0.01
    """filter out low reward samples if online filtering"""
    filter_high: float = 0.99
    """filter out high reward samples if online filtering"""
    opd: OPDConfig = field(default_factory=OPDConfig)


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    rollout_audit: RolloutAuditConfig = field(default_factory=RolloutAuditConfig)
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    validate_at_end: bool = True
    """run one final validation pass after the training loop"""
    save_at_end: bool = True
    """save a final checkpoint when the last step was not already saved"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
