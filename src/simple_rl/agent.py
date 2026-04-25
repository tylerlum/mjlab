import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import gym
import numpy as np
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from torch import nn

from simple_rl.player import InferenceConfig
from simple_rl.utils import (
    asymmetric_critic,
    datasets,
    models,
    schedulers,
    torch_utils,
)
from simple_rl.utils.asymmetric_critic import AsymmetricCriticConfig
from simple_rl.utils.conditioning_utils import CONDITIONING_IDX_DIM
from simple_rl.utils.experience import ExperienceBuffer
from simple_rl.utils.network import NetworkConfig
from simple_rl.utils.optimizers import OptimizerName, build_optimizer
from simple_rl.utils.rewards_shaper import (
    DefaultRewardsShaper,
    RewardsShaperParams,
)
from simple_rl.utils.sapg_utils import filter_leader, shuffle_batch

# =============================================================================
# HARD-CODED COMPATIBILITY HACKS
# -----------------------------------------------------------------------------
# These are intentionally NOT config values.
#
# Rationale:
# - They encode repo-specific compatibility choices or bug-emulation toggles.
# - They are not meaningful task hyperparameters.
# - We do not want task YAMLs to silently depend on them.
#
# If one of these needs to change for parity/debugging, do it explicitly in code.
# =============================================================================

# BUG_FROM_EPO_REPO:
# Original EPO repo behavior after an evolutionary replacement:
#   1. reset `target_objective_known = False`
#   2. but DO NOT clear the set of finished env ids
#
# That means the next interval can immediately appear "fully observed" even though many
# `last_true_objectives` entries are still stale values from the previous interval.
# This looks like a real bug rather than an intended algorithmic choice.
#
# Keep False by default so the next interval must genuinely re-observe all envs.
# Flip to True only if exact original-repo bug emulation is needed for debugging.
BUG_FROM_EPO_REPO_KEEP_FINISHED_AGENTS_AFTER_REPLACEMENT = False


@dataclass
class SapgConfig:
    """Configuration for SAPG (Split and Aggregate Policy Gradients).

    M blocks of environments each receive a different entropy coefficient (linspace 0.5→0.0).
    A learned (M, conditioning_dim) embedding matrix distinguishes blocks; the integer
    conditioning index is appended to obs and looked up at forward time.

    Obs normalization note: simple_rl normalizes only the raw obs dims (excluding the integer
    conditioning index), so the per-block conditioning lookup is always exact.  rl_games uses
    the same design via coef_id_idx: only obs[:, :coef_id_idx] is normalized and the appended
    float block identifier (50.0, 40.0, ...) passes through unnormalized.  Both frameworks
    correctly exclude the block identifier from normalization.

    Set ppo.sapg to enable. For EPO, also set ppo.epo.
    """

    num_conditionings: int
    """M in the SAPG paper — number of blocks = number of distinct entropy levels.
    Envs are split into M equal groups; each group receives its own conditioning embedding
    and entropy coefficient.  Must evenly divide num_actors."""

    conditioning_dim: int
    """Dimension of the per-block learnable embedding appended to obs.
    Corresponds to rl_games' expl_reward_coef_embd_size (default 32).
    The policy network input grows by this many dims at inference time."""

    use_others_experience: bool = True
    """If True, each minibatch is augmented with re-labelled experience from the leader block
    (rl_games 'lf' / leader-follower mode).  Disabled for very small env counts where
    the augmented batch size would not divide evenly."""

    off_policy_ratio: int = 1
    """Number of extra off-policy blocks added per minibatch when use_others_experience=True.
    Each extra block increases the effective minibatch by batch_size / M.
    Value of 1 means one additional off-policy copy (2× block-0 data per update)."""

    # Block ordering (same convention as rl_games):
    #   block 0     = LEADER   — highest entropy coef = 0.5 * entropy_coef_scale  (most exploratory)
    #   block M-1   = FOLLOWER — entropy coef = 0.0                               (pure exploitation)
    #   blocks 1…M-2           — linearly interpolated between leader and follower
    #
    # Entropy coefficients: torch.linspace(0.5, 0.0, M) * entropy_coef_scale
    # E.g. M=6, scale=0.005 → coefs [0.0025, 0.002, 0.0015, 0.001, 0.0005, 0.0]
    use_entropy_bonus: bool = True
    """Whether to add a per-block entropy bonus to each block's policy loss.
    True (default): block k gets entropy coef = linspace(0.5, 0.0, M)[k] * entropy_coef_scale.
    False: all blocks are trained with no entropy bonus (equivalent to SAPG without exploration)."""

    entropy_coef_scale: float = 1.0
    """Scale multiplier applied to the linspace(0.5, 0.0, M) entropy coefficient schedule.
    Actual per-block coefs = linspace(0.5, 0.0, M) * entropy_coef_scale.
    Block 0 (leader) gets the maximum coef = 0.5 * entropy_coef_scale.
    Block M-1 (follower) always gets 0.0 regardless of scale.
    Pretrained policy uses 0.005 → leader coef = 0.0025."""

    entropy_in_returns: bool = False
    """If True, adds entropy * coef to the shaped rewards before GAE computation (intrinsic reward).
    If False (default, matches rl_games), entropy only enters via the loss term directly.
    Setting True makes the entropy bonus propagate through value targets, which can help
    exploration on sparse-reward tasks but may slow convergence on dense-reward ones."""


@dataclass
class EpoConfig:
    """Configuration for EPO (Evolutionary Policy Optimization).

    EPO extends SAPG with the original repo's embedding-selection logic:
    after a warm-up, evaluate blocks over fixed frame intervals, rank them by
    interval-best true objective, average the top-k parents, and replace the
    worst policies. Requires ppo.sapg to also be set (EPO is SAPG + evolution).
    """

    evolution_interval_frames: int = 20_000_000
    """Frame interval between evolutionary selection events.

    Matches the original EPO repo's hard-coded 20M frame cadence.
    """

    evolution_warmup_frames: int = 200_000_000
    """Delay evolution until at least this many frames have elapsed.

    Matches the original EPO repo's hard-coded 200M frame warm-up.
    """

    num_best_policies: int = 2
    """Number of top-ranked parent policies whose embeddings are averaged."""

    num_replace_policies: int = 1
    """Number of worst-ranked policies replaced at each evolutionary event.

    The original EPO repo always replaces exactly one policy per event.
    """

    # Legacy fields kept only so older local test code/config constructors still parse.
    evolution_frequency: Optional[int] = None
    evolution_kill_ratio: Optional[float] = None

    log_sigma: bool = False
    """Log mean action standard deviation (sigma) to TensorBoard each epoch.
    Useful for diagnosing whether per-block sigmas are collapsing or diverging."""

    log_off_policy_grads: bool = False
    """Log cosine similarity between on-policy and off-policy gradient directions each epoch.
    Expensive (requires two backward passes); only enable for debugging."""


@dataclass
class PpoConfig:
    """PPO training configuration. Required fields have no defaults; optional fields do."""

    # ── Required fields ───────────────────────────────────────────────────────
    num_actors: int
    learning_rate: float
    entropy_coef: float
    """Coefficient on the entropy bonus in the policy loss (encourages exploration).
    0.0 = no bonus; positive values push the policy toward higher-entropy actions."""

    horizon_length: int
    """Number of environment steps collected per environment before each gradient update.
    Total batch size = num_actors × horizon_length."""

    normalize_advantage: bool
    """Normalize advantages to mean=0, std=1 within each minibatch before the PPO update.
    Almost always beneficial for stability."""

    normalize_input: bool
    """Maintain a running mean/std of observations and normalize them online.
    Greatly helps when observation scales vary widely across dims."""

    grad_norm: float
    """Gradient norm clipping threshold. Gradients are rescaled to have L2 norm ≤ grad_norm."""

    critic_coef: float
    """Weight of the value (critic) loss in the combined loss: total = actor + critic_coef * critic."""

    gamma: float
    """Discount factor for future rewards. Typical value: 0.99."""

    tau: float
    """GAE-λ: controls the bias–variance trade-off in advantage estimation.
    τ=1.0 → Monte-Carlo returns (low bias, high variance).
    τ=0.0 → TD(0) (high bias, low variance).
    Typical value: 0.95."""

    reward_shaper: RewardsShaperParams
    mini_epochs: int
    """Number of gradient update passes over the collected rollout per epoch.
    Higher values extract more signal per sample but can cause policy divergence for LSTM."""

    e_clip: float
    """PPO clipping parameter ε. The policy ratio r = π_new/π_old is clipped to [1-ε, 1+ε].
    Smaller values → more conservative updates. Typical: 0.1–0.2."""

    # ── Fields with defaults ──────────────────────────────────────────────────
    multi_gpu: bool = False
    device: str = "cuda:0"
    optimizer: OptimizerName = "adam"
    """Optimizer used for the actor/policy network. Defaults to Adam to preserve current behavior."""

    optimizer_beta1: float = 0.9
    """First-moment coefficient for Adam/AdamW; momentum coefficient for Muon."""

    optimizer_beta2: float = 0.999
    """Second-moment coefficient for Adam/AdamW. Ignored by Muon."""

    optimizer_eps: float = 1e-8
    """Numerical stability epsilon passed to the optimizer."""

    weight_decay: float = 0.0
    asymmetric_critic: Optional[AsymmetricCriticConfig] = None
    truncate_grads: bool = False
    """If True, skip the gradient step when the clipped norm would exceed grad_norm.
    Conservative alternative to clipping: avoids corrupt updates entirely."""

    save_frequency: int = 0
    save_best_after: int = 100
    print_stats: bool = True
    max_epochs: int = -1
    max_frames: int = -1
    lr_schedule: Optional[Literal["adaptive", "linear"]] = None
    """Learning-rate schedule. None keeps the LR constant, "adaptive" matches rl_games'
    KL-based scheduler and requires kl_threshold, and "linear" decays over max_epochs/max_frames."""

    schedule_type: Literal["legacy", "standard"] = "legacy"
    """Controls when the LR scheduler is stepped.
    "legacy": LR updates happen inside each minibatch loop (per mini-epoch step).
    "standard": LR updates happen once per mini-epoch from the mean KL, matching rl_games configs."""

    kl_threshold: Optional[float] = None
    """Target KL divergence for adaptive LR schedule. If mean KL > threshold, LR is decreased;
    if KL < threshold/2, LR is increased. Only used when lr_schedule="adaptive"."""

    seq_length: int = 4
    """LSTM sequence length: the rollout is split into chunks of this length for BPTT.
    Should equal horizon_length for full-horizon backprop (the typical setting for LSTM policies).
    Shorter values trade gradient quality for memory."""

    zero_rnn_on_done: bool = True
    """Reset LSTM hidden state to zeros at episode boundaries (done=True).
    Almost always True; setting False lets the LSTM carry state across episode resets."""

    normalize_value: bool = False
    """Normalize value targets with a running mean/std before the critic update.
    Helps when reward magnitudes change over training."""

    games_to_track: int = 100
    """Rolling window size for the mean episode reward reported during training."""

    minibatch_size_per_env: int = 0
    minibatch_size: Optional[int] = None
    """Size of each gradient step. batch_size / minibatch_size minibatches are processed per
    mini-epoch. Smaller → more gradient steps per epoch but higher variance per step.
    Reference SAPG/EPO repos use 4 × num_envs (4 minibatches per epoch)."""

    mixed_precision: bool = False
    bounds_loss_coef: Optional[float] = None
    """Coefficient for an action-bounds regularization loss that penalizes actions outside
    [-1, 1] (before tanh squashing). None or 0.0 = disabled."""

    bound_loss_type: Literal["bound", "regularisation"] = "bound"
    value_bootstrap: bool = False
    """If True, bootstrap the value at episode truncations (timeout) rather than treating
    them as terminal states. Corrects value targets when episodes end due to time limits."""

    clip_actions: bool = True
    schedule_entropy: bool = False
    freeze_critic: bool = False

    # ── SAPG / EPO — both None means plain PPO ────────────────────────────────
    sapg: Optional[SapgConfig] = None
    """Set to a SapgConfig to enable SAPG (multi-block training with entropy diversity).
    Also required when using EPO."""

    epo: Optional[EpoConfig] = None
    """Set to an EpoConfig to add EPO evolutionary updates on top of SAPG.
    Requires sapg to also be set."""

    def to_inference_config(self) -> InferenceConfig:
        return InferenceConfig(
            normalize_input=self.normalize_input,
            clip_actions=self.clip_actions,
            device=self.device,
            normalize_value=self.normalize_value,
            conditioning_dim=self.sapg.conditioning_dim if self.sapg is not None else None,
            num_conditionings=self.sapg.num_conditionings if self.sapg is not None else None,
        )


def swap_and_flatten01(arr: torch.Tensor) -> torch.Tensor:
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


def print_statistics(
    print_stats: bool,
    curr_frames: float,
    step_time: float,
    step_inference_time: float,
    total_time: float,
    epoch_num: int,
    max_epochs: int,
    frame: float,
    max_frames: int,
) -> None:
    if print_stats:
        step_time = max(step_time, 1e-9)
        fps_step = curr_frames / step_time
        fps_step_inference = curr_frames / step_inference_time
        fps_total = curr_frames / total_time

        if max_epochs == -1 and max_frames == -1:
            print(
                f"fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}"
            )
        elif max_epochs == -1:
            print(
                f"fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}/{max_frames:.0f}"
            )
        elif max_frames == -1:
            print(
                f"fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}"
            )
        else:
            print(
                f"fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}/{max_frames:.0f}"
            )


class Agent:
    def __init__(
        self,
        experiment_dir: Path,
        ppo_config: PpoConfig,
        network_config: NetworkConfig,
        env: Any,
    ):
        self.cfg = ppo_config

        ## A2CBase ##
        # multi-gpu/multi-node data
        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        if self.cfg.multi_gpu:
            # local rank of the GPU in a node
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # global rank of the GPU
            self.global_rank = int(os.getenv("RANK", "0"))
            # total number of GPUs across all nodes
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))

            dist.init_process_group(
                "nccl", rank=self.global_rank, world_size=self.world_size
            )

            self.device_name = "cuda:" + str(self.local_rank)
            self.cfg.device = self.device_name
            if self.global_rank != 0:
                self.cfg.print_stats = False
                self.cfg.lr_schedule = None

        self.env = env
        self.env_info = self.env.get_env_info()
        self.value_size = self.env_info.get("value_size", 1)
        self.observation_space = self.env_info["observation_space"]
        self.num_agents = self.env_info.get("agents", 1)

        self.has_asymmetric_critic = self.cfg.asymmetric_critic is not None
        if self.has_asymmetric_critic:
            self.state_space = self.env_info.get("state_space", None)
            if isinstance(self.state_space, gym.spaces.Dict):
                raise NotImplementedError("Dict state space is not supported")
            else:
                self.state_shape = self.state_space.shape

        self.rnn_states: Optional[Tuple[torch.Tensor, ...]] = None

        # Setting learning rate scheduler
        if self.cfg.lr_schedule == "adaptive":
            assert self.cfg.kl_threshold is not None
            self.scheduler = schedulers.AdaptiveScheduler(
                kl_threshold=self.cfg.kl_threshold
            )

        elif self.cfg.lr_schedule == "linear":
            if self.cfg.max_epochs == -1 and self.cfg.max_frames == -1:
                print(
                    "Max epochs and max frames are not set. Linear learning rate schedule can't be used, switching to the contstant (identity) one."
                )
                self.scheduler = schedulers.IdentityScheduler()
            else:
                use_epochs = True
                max_steps = self.cfg.max_epochs

                if self.cfg.max_epochs == -1:
                    use_epochs = False
                    max_steps = self.cfg.max_frames

                self.scheduler = schedulers.LinearScheduler(
                    float(self.cfg.learning_rate),
                    max_steps=max_steps,
                    use_epochs=use_epochs,
                    apply_to_entropy=self.cfg.schedule_entropy,
                    start_entropy_coef=self.cfg.entropy_coef,
                )
        else:
            self.scheduler = schedulers.IdentityScheduler()

        self.rewards_shaper = DefaultRewardsShaper(self.cfg.reward_shaper)

        if isinstance(self.observation_space, gym.spaces.Dict):
            raise NotImplementedError("Dict observation space is not supported")
        else:
            self.obs_shape = self.observation_space.shape
            assert len(self.obs_shape) == 1, (
                f"Observation shape must be 1D, got {self.obs_shape}"
            )

        print("current training device:", self.device)
        self.game_rewards = torch_utils.AverageMeter(
            in_shape=self.value_size, max_size=self.cfg.games_to_track
        ).to(self.device)
        self.game_shaped_rewards = torch_utils.AverageMeter(
            in_shape=self.value_size, max_size=self.cfg.games_to_track
        ).to(self.device)
        self.game_lengths = torch_utils.AverageMeter(
            in_shape=1, max_size=self.cfg.games_to_track
        ).to(self.device)

        # ════════════════════════════════════════════════════════════════════════
        # ENV-SPECIFIC METERS — SimToolReal task metrics tracked at episode end.
        # Keys must match what the env puts in the `infos` dict (env.py extras).
        # Add / remove entries here when the task changes.
        # ════════════════════════════════════════════════════════════════════════
        self.game_successes = torch_utils.AverageMeter(
            in_shape=1, max_size=self.cfg.games_to_track
        ).to(self.device)
        self.game_closest_kp = torch_utils.AverageMeter(
            in_shape=1, max_size=self.cfg.games_to_track
        ).to(self.device)
        self.game_true_obj = torch_utils.AverageMeter(
            in_shape=1, max_size=self.cfg.games_to_track
        ).to(self.device)
        # ════════════════════════════════════════════════════════════════════════

        # ════════════════════════════════════════════════════════════════════════
        # GENERAL ENV-INFO STATE — mirrors RLGPUAlgoObserver in rlgames_utils.py.
        # _process_infos() is called every rollout step; _log_env_info() is called
        # once per epoch. Together they replicate the rl_games observer pattern so
        # WandB shows the same metric set for both training backends.
        # ════════════════════════════════════════════════════════════════════════
        from collections import deque as _deque  # noqa: F401 (used in _process_infos)
        self._deque = _deque
        self._ep_cumulative: dict = {}        # running per-env cumulative for episode_cumulative keys
        self._ep_cumulative_avg: dict = {}    # deque[games_to_track] per key, filled at episode end
        self._direct_info: dict = {}          # flat scalar dict from the most-recent infos step
        self._new_finished_episodes: bool = False
        # ════════════════════════════════════════════════════════════════════════

        self.obs_dict: Optional[Dict[str, Any]] = None

        self.batch_size_envs = self.cfg.horizon_length * self.cfg.num_actors
        self.batch_size = self.batch_size_envs * self.num_agents

        # either minibatch_size_per_env or minibatch_size should be present in a config
        # if both are present, minibatch_size is used
        # otherwise minibatch_size_per_env is used minibatch_size_per_env is used to calculate minibatch_size
        self.minibatch_size = (
            self.cfg.minibatch_size
            if self.cfg.minibatch_size is not None
            else self.cfg.num_actors * self.cfg.minibatch_size_per_env
        )
        assert self.minibatch_size > 0

        assert self.batch_size % self.minibatch_size == 0, (
            f"batch_size ({self.batch_size}) must be divisible by minibatch_size ({self.minibatch_size})"
        )

        # For SAPG with experience sharing, the training batch is augmented:
        #   augmented_batch = batch + (num_repeat-1) * block_size_steps
        # where block_size_steps = horizon * (num_actors / M). rl_games keeps
        # floor(augmented_batch / minibatch_size) minibatches and extends the final
        # minibatch to include the augmented tail.
        if (
            self.cfg.sapg is not None
            and self.cfg.sapg.use_others_experience
        ):
            M = self.cfg.sapg.num_conditionings
            assert self.batch_size % self.minibatch_size == 0, (
                f"For SAPG use_others_experience, batch_size ({self.batch_size}) must be "
                f"divisible by minibatch_size ({self.minibatch_size})."
            )
            num_repeat = min(M, self.cfg.sapg.off_policy_ratio + 1)
            self.augmented_batch_size = self.batch_size + (num_repeat - 1) * (self.batch_size // M)
        else:
            self.augmented_batch_size = self.batch_size

        self.num_minibatches = self.augmented_batch_size // self.minibatch_size
        assert self.num_minibatches > 0, f"{self.augmented_batch_size}, {self.minibatch_size}"

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.cfg.mixed_precision and self.device != "cpu"
        )

        self.current_lr = float(self.cfg.learning_rate)
        self.frame = 0
        self.update_time = 0
        self.mean_rewards = self.last_mean_rewards = -1000000000
        self.epoch_num = 0
        self.curr_frames = 0

        self.experiment_dir = experiment_dir
        self.nn_dir = self.experiment_dir / "nn"
        self.summaries_dir = self.experiment_dir / "summaries"

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.nn_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

        self.current_entropy_coef = self.cfg.entropy_coef

        if self.global_rank == 0:
            writer = SummaryWriter(str(self.summaries_dir))
            self.writer = writer
        else:
            self.writer = None

        self.is_tensor_obses = False

        if self.cfg.epo is not None:
            assert self.cfg.sapg is not None, "epo config requires sapg config to also be set"

        # SAPG/EPO: entropy intrinsic reward setup
        # uses_intrinsic_reward: controls per-block entropy in the LOSS (always True for SAPG)
        # uses_entropy_in_returns: controls whether entropy is also added to GAE returns.
        #   False (default): entropy only enters via the loss term (negative -entropy_loss).
        #   True: entropy is also added to shaped rewards before GAE (like an intrinsic bonus).
        #   rl_games stores intr_rewards=zeros for expl_type=entropy, so rl_games also uses
        #   False semantics (entropy only in loss).  Both default to this.  Set True to explore
        #   the alternative formulation.
        self.uses_intrinsic_reward = (
            self.cfg.sapg is not None
            and self.cfg.sapg.use_entropy_bonus
        )
        self.uses_entropy_in_returns = (
            self.uses_intrinsic_reward
            and self.cfg.sapg.entropy_in_returns
        )
        if self.uses_intrinsic_reward:
            M = self.cfg.sapg.num_conditionings
            assert self.cfg.num_actors % M == 0, (
                f"num_actors ({self.cfg.num_actors}) must be divisible by num_conditionings ({M})"
            )
            block_size = self.cfg.num_actors // M
            env_block_ids = (
                torch.arange(M).repeat_interleave(block_size).to(self.device)
            )
            # Per-env entropy coefficient, shape (num_actors,): used in GAE when entropy_in_returns=True
            self.intr_reward_coef = (
                torch.linspace(0.5, 0.0, M).to(self.device)[env_block_ids]
                * self.cfg.sapg.entropy_coef_scale
            )
            # Per-block entropy coefficient, shape (M,): used for per-sample lookup in loss
            self.intr_reward_coef_per_block = (
                torch.linspace(0.5, 0.0, M).to(self.device)
                * self.cfg.sapg.entropy_coef_scale
            )
        else:
            self.intr_reward_coef = None
            self.intr_reward_coef_per_block = None

        # EPO/SAPG: per-block episode reward tracking
        if self.cfg.sapg is not None:
            self.block_game_rewards = [
                torch_utils.AverageMeter(
                    in_shape=self.value_size, max_size=self.cfg.games_to_track
                ).to(self.device)
                for _ in range(self.cfg.sapg.num_conditionings)
            ]
        else:
            self.block_game_rewards = None

        if self.cfg.epo is not None:
            self._epo_interval_idx = -1
            self._epo_finished_agents: Set[int] = set()
            self._epo_last_true_objectives: Optional[List[float]] = None
            self._epo_target_objective_known = False
            self._epo_best_objective_curr_interval_per_block: Optional[List[float]] = None
        else:
            self._epo_interval_idx = -1
            self._epo_finished_agents = set()
            self._epo_last_true_objectives = None
            self._epo_target_objective_known = False
            self._epo_best_objective_curr_interval_per_block = None

        # Conditioning idxs: integer index per env, appended to obs/states
        if self.cfg.sapg is not None:
            M = self.cfg.sapg.num_conditionings
            if self.env.num_envs < M:
                conditioning_idxs = (
                    torch.arange(self.env.num_envs)
                    .to(self.device)
                    .reshape(self.env.num_envs, CONDITIONING_IDX_DIM)
                )
            else:
                assert self.env.num_envs % M == 0, (
                    f"Number of environments must be divisible by num_conditionings: "
                    f"{self.env.num_envs} % {M} != 0"
                )
                block_size = self.env.num_envs // M
                conditioning_idxs = (
                    torch.arange(M)
                    .repeat_interleave(block_size)
                    .to(self.device)
                    .reshape(self.env.num_envs, CONDITIONING_IDX_DIM)
                )
            self.conditioning_idxs = conditioning_idxs
        else:
            self.conditioning_idxs = None

        ## ContinuousA2CBase ##
        action_space = self.env_info["action_space"]
        self.actions_num = action_space.shape[0]

        self.actions_low = (
            torch.from_numpy(action_space.low.copy()).float().to(self.device)
        )
        self.actions_high = (
            torch.from_numpy(action_space.high.copy()).float().to(self.device)
        )

        ## A2CAgent ##
        self.model = models.ModelA2CContinuousLogStd(
            network_config=network_config,
            actions_num=self.actions_num,
            input_shape=self.obs_shape,
            normalize_value=self.cfg.normalize_value,
            normalize_input=self.cfg.normalize_input,
            value_size=self.env_info.get("value_size", 1),
            num_seqs=self.cfg.num_actors * self.num_agents,
            conditioning_dim=self.cfg.sapg.conditioning_dim if self.cfg.sapg is not None else None,
            num_conditionings=self.cfg.sapg.num_conditionings if self.cfg.sapg is not None else None,
        )

        self.model.to(self.device)

        self.states = None
        self.init_rnn_from_model(self.model)
        self.optimizer = build_optimizer(
            params=self.model.parameters(),
            optimizer_name=self.cfg.optimizer,
            learning_rate=float(self.current_lr),
            beta1=self.cfg.optimizer_beta1,
            beta2=self.cfg.optimizer_beta2,
            eps=self.cfg.optimizer_eps,
            weight_decay=self.cfg.weight_decay,
        )

        if self.has_asymmetric_critic:
            print("Adding Asymmetric Critic Network")
            assert self.cfg.asymmetric_critic is not None
            critic_cfg = replace(
                self.cfg.asymmetric_critic,
                optimizer=(
                    self.cfg.asymmetric_critic.optimizer
                    if self.cfg.asymmetric_critic.optimizer is not None
                    else self.cfg.optimizer
                ),
                optimizer_beta1=(
                    self.cfg.asymmetric_critic.optimizer_beta1
                    if self.cfg.asymmetric_critic.optimizer_beta1 is not None
                    else self.cfg.optimizer_beta1
                ),
                optimizer_beta2=(
                    self.cfg.asymmetric_critic.optimizer_beta2
                    if self.cfg.asymmetric_critic.optimizer_beta2 is not None
                    else self.cfg.optimizer_beta2
                ),
                optimizer_eps=(
                    self.cfg.asymmetric_critic.optimizer_eps
                    if self.cfg.asymmetric_critic.optimizer_eps is not None
                    else self.cfg.optimizer_eps
                ),
            )
            self.asymmetric_critic_net = asymmetric_critic.AsymmetricCritic(
                state_shape=self.state_shape,
                value_size=self.value_size,
                ppo_device=self.device,
                num_agents=self.num_agents,
                horizon_length=self.cfg.horizon_length,
                num_actors=self.cfg.num_actors,
                num_actions=self.actions_num,
                seq_length=self.cfg.seq_length,
                normalize_value=self.cfg.normalize_value,
                config=critic_cfg,
                writer=self.writer,
                max_epochs=self.cfg.max_epochs,
                multi_gpu=self.cfg.multi_gpu,
                zero_rnn_on_done=self.cfg.zero_rnn_on_done,
                batch_size=self.augmented_batch_size,
                conditioning_dim=self.cfg.sapg.conditioning_dim if self.cfg.sapg is not None else None,
                num_conditionings=self.cfg.sapg.num_conditionings if self.cfg.sapg is not None else None,
            ).to(self.device)

        self.dataset = datasets.PPODataset(
            batch_size=self.augmented_batch_size,
            minibatch_size=self.minibatch_size,
            is_rnn=self.is_rnn,
            device=self.device,
            seq_length=self.cfg.seq_length,
        )
        if self.cfg.normalize_value:
            self.value_mean_std = (
                self.asymmetric_critic_net.model.value_mean_std
                if self.has_asymmetric_critic
                else self.model.value_mean_std
            )

    def truncate_gradients_and_step(self) -> List[torch.Tensor]:
        if self.cfg.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))

            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(
                            param.grad.data
                        )
                        / self.world_size
                    )
                    offset += param.numel()
        else:
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))
            all_grads = torch.cat(all_grads_list)

        if self.cfg.truncate_grads:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return all_grads

    def write_stats(
        self,
        total_time: float,
        epoch_num: int,
        step_time: float,
        play_time: float,
        update_time: float,
        a_losses: List[torch.Tensor],
        c_losses: List[torch.Tensor],
        entropies: List[torch.Tensor],
        kls: List[torch.Tensor],
        current_lr: float,
        lr_mul: float,
        frame: int,
        scaled_time: float,
        scaled_play_time: float,
        curr_frames: float,
        csigmas: Optional[List[torch.Tensor]] = None,
    ) -> None:
        if self.writer is None:
            print("writer is None, skipping writing stats")
            return

        # do we need scaled time?
        self.writer.add_scalar(
            tag="performance/step_inference_rl_update_fps",
            scalar_value=curr_frames / scaled_time,
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="performance/step_inference_fps",
            scalar_value=curr_frames / scaled_play_time,
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="performance/step_fps",
            scalar_value=curr_frames / step_time,
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="performance/rl_update_time",
            scalar_value=update_time,
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="performance/step_inference_time",
            scalar_value=play_time,
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="performance/step_time", scalar_value=step_time, global_step=frame
        )
        self.writer.add_scalar(
            tag="losses/a_loss",
            scalar_value=torch.mean(torch.stack(a_losses)).item(),
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="losses/c_loss",
            scalar_value=torch.mean(torch.stack(c_losses)).item(),
            global_step=frame,
        )

        self.writer.add_scalar(
            tag="losses/entropy",
            scalar_value=torch.mean(torch.stack(entropies)).item(),
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="info/current_lr", scalar_value=current_lr * lr_mul, global_step=frame
        )
        self.writer.add_scalar(
            tag="info/lr_mul", scalar_value=lr_mul, global_step=frame
        )
        self.writer.add_scalar(
            tag="info/e_clip", scalar_value=self.cfg.e_clip * lr_mul, global_step=frame
        )
        self.writer.add_scalar(
            tag="info/kl",
            scalar_value=torch.mean(torch.stack(kls)).item(),
            global_step=frame,
        )
        self.writer.add_scalar(
            tag="info/epochs", scalar_value=epoch_num, global_step=frame
        )

        # EPO: action sigma logging
        if self.cfg.epo is not None and self.cfg.epo.log_sigma and csigmas is not None and len(csigmas) > 0:
            self.writer.add_scalar(
                tag="auxiliary_stats/sigma",
                scalar_value=torch.stack(csigmas).mean().item(),
                global_step=frame,
            )

        # EPO/SAPG: per-block reward logging
        if self.block_game_rewards is not None:
            for k, meter in enumerate(self.block_game_rewards):
                if meter.current_size > 0:
                    self.writer.add_scalar(
                        tag=f"block_rewards/block_{k}",
                        scalar_value=meter.get_mean()[0].item(),
                        global_step=frame,
                    )

        # SAPG: intrinsic reward logging
        if self.uses_intrinsic_reward and hasattr(self, "experience_buffer"):
            mb_intr_rewards = self.experience_buffer.tensor_dict.get(
                "intr_rewards", None
            )
            if mb_intr_rewards is not None:
                M = self.cfg.sapg.num_conditionings
                block_size = self.cfg.num_actors // M
                for k in range(M):
                    self.writer.add_scalar(
                        tag=f"intr_rewards/block_{k}",
                        scalar_value=mb_intr_rewards[
                            :, k * block_size : (k + 1) * block_size
                        ]
                        .mean()
                        .item(),
                        global_step=frame,
                    )
                mb_extr_rewards = self.experience_buffer.tensor_dict.get("rewards")
                if mb_extr_rewards is not None:
                    self.writer.add_scalar(
                        tag="intr_rewards/extr_rewards",
                        scalar_value=mb_extr_rewards.mean().item(),
                        global_step=frame,
                    )

    def set_eval(self) -> None:
        self.model.eval()

    def set_train(self) -> None:
        self.model.train()

    def update_lr(self, lr: float) -> None:
        if self.cfg.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, src=0)
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_action_values(self, obs_dict: Dict[str, Any]) -> dict:
        obs = obs_dict["obs"]
        self.model.eval()
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.rnn_states,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_asymmetric_critic:
                states = obs_dict["states"]
                input_dict = {
                    "is_train": False,
                    "states": states,
                }
                value = self.get_asymmetric_critic_value(input_dict)
                res_dict["values"] = value
        return res_dict

    def get_values(
        self,
        obs_dict: Dict[str, Any],
        rnn_states: Optional[Tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            if self.has_asymmetric_critic:
                states = obs_dict["states"]
                self.asymmetric_critic_net.eval()
                input_dict = {
                    "is_train": False,
                    "states": states,
                    "actions": None,
                    "is_done": self.dones,
                }
                value = self.get_asymmetric_critic_value(input_dict)
            else:
                self.model.eval()
                obs = obs_dict["obs"]
                input_dict = {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": obs,
                    "rnn_states": rnn_states,
                }
                result = self.model(input_dict)
                value = result["values"]
            return value

    @property
    def device(self) -> Union[str, torch.device]:
        return self.cfg.device

    def init_tensors(self) -> None:
        ## A2CBase ##
        batch_size = self.num_agents * self.cfg.num_actors
        self.experience_buffer = ExperienceBuffer(
            env_info=self.env_info,
            num_actors=self.cfg.num_actors,
            horizon_length=self.cfg.horizon_length,
            has_asymmetric_critic=self.has_asymmetric_critic,
            device=self.device,
            extra_obs_dim=CONDITIONING_IDX_DIM if self.cfg.sapg is not None else None,
            extra_states_dim=CONDITIONING_IDX_DIM
            if self.has_asymmetric_critic and self.cfg.sapg is not None
            else None,
            aux_tensor_dict={"intr_rewards": (1,)} if self.uses_entropy_in_returns else None,
        )

        _val_shape = (self.cfg.horizon_length, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(
            current_rewards_shape, dtype=torch.float32, device=self.device
        )
        self.current_shaped_rewards = torch.zeros(
            current_rewards_shape, dtype=torch.float32, device=self.device
        )
        self.current_lengths = torch.zeros(
            batch_size, dtype=torch.float32, device=self.device
        )
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)

        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.device) for s in self.rnn_states]

            total_agents = self.num_agents * self.cfg.num_actors
            num_seqs = self.cfg.horizon_length // self.cfg.seq_length
            assert (
                self.cfg.horizon_length * total_agents // self.num_minibatches
            ) % self.cfg.seq_length == 0
            self.mb_rnn_states = [
                torch.zeros(
                    (num_seqs, s.size()[0], total_agents, s.size()[2]),
                    dtype=torch.float32,
                    device=self.device,
                )
                for s in self.rnn_states
            ]

        ## ContinuousA2CBase ##
        self.update_list = ["actions", "neglogpacs", "values", "mus", "sigmas"]
        self.tensor_list = self.update_list + ["obses", "states", "dones"]

    def init_rnn_from_model(self, model) -> None:
        self.is_rnn = self.model.is_rnn()

    def cast_obs(self, obs) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        elif isinstance(obs, np.ndarray):
            assert obs.dtype != np.int8
            if obs.dtype == np.uint8:
                obs = torch.ByteTensor(obs).to(self.device)
            else:
                obs = torch.FloatTensor(obs).to(self.device)
        return obs

    def obs_to_dict_of_tensors(self, obs) -> Dict[str, Any]:
        obs_is_dict = isinstance(obs, dict)
        if obs_is_dict:
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        if not obs_is_dict or "obs" not in obs:
            upd_obs = {"obs": upd_obs}
        return upd_obs

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            upd_obs = {}
            for key, value in obs.items():
                upd_obs[key] = self._obs_to_tensors_internal(value)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs

    def preprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.cfg.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = torch_utils.rescale_actions(
                self.actions_low, self.actions_high, clamped_actions
            )
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions

    def env_step(self, actions: torch.Tensor) -> Tuple:
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.env.step(actions)

        obs_dict = self.obs_to_dict_of_tensors(obs)

        if self.conditioning_idxs is not None:
            obs_dict["obs"] = torch.cat(
                [obs_dict["obs"], self.conditioning_idxs], dim=1
            )
            if self.has_asymmetric_critic:
                obs_dict["states"] = torch.cat(
                    [obs_dict["states"], self.conditioning_idxs], dim=1
                )

        if self.is_tensor_obses:
            if self.value_size == 1:
                rewards = rewards.unsqueeze(1)
            return (
                obs_dict,
                rewards.to(self.device),
                dones.to(self.device),
                infos,
            )
        else:
            if self.value_size == 1:
                rewards = np.expand_dims(rewards, axis=1)
            return (
                obs_dict,
                torch.from_numpy(rewards).float().to(self.device),
                torch.from_numpy(dones).to(self.device),
                infos,
            )

    def env_reset(self):
        obs = self.env.reset()
        obs_dict = self.obs_to_dict_of_tensors(obs)
        if self.conditioning_idxs is not None:
            obs_dict["obs"] = torch.cat(
                [obs_dict["obs"], self.conditioning_idxs], dim=1
            )
            if self.has_asymmetric_critic:
                obs_dict["states"] = torch.cat(
                    [obs_dict["states"], self.conditioning_idxs], dim=1
                )
        return obs_dict

    def discount_values(
        self,
        fdones: torch.Tensor,
        last_extrinsic_values: torch.Tensor,
        mb_fdones: torch.Tensor,
        mb_extrinsic_values: torch.Tensor,
        mb_rewards: torch.Tensor,
    ) -> torch.Tensor:
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.cfg.horizon_length)):
            if t == self.cfg.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t + 1]
                nextvalues = mb_extrinsic_values[t + 1]
            nextnonterminal = nextnonterminal.unsqueeze(1)

            delta = (
                mb_rewards[t]
                + self.cfg.gamma * nextvalues * nextnonterminal
                - mb_extrinsic_values[t]
            )
            mb_advs[t] = lastgaelam = (
                delta + self.cfg.gamma * self.cfg.tau * nextnonterminal * lastgaelam
            )
        return mb_advs

    def clear_stats(self) -> None:
        self.game_rewards.clear()
        self.game_shaped_rewards.clear()
        self.game_lengths.clear()
        self.mean_rewards = self.last_mean_rewards = -100500
        if self.block_game_rewards is not None:
            for meter in self.block_game_rewards:
                meter.clear()

        # ════════════════════════════════════════════════════════════════════════
        # SIMTOOLREAL-SPECIFIC CLEAR
        # -----------------------------------------------------------------------------
        # These meters are NOT general PPO framework state. They exist only because the
        # current SimToolReal env exports these specific per-episode task metrics:
        #   - successes
        #   - closest_keypoint_max_dist
        #   - true_objective
        #
        # If another env wants framework-level logging, add a new env-adapter path rather than
        # silently reusing these SimToolReal names.
        # Keep this block in sync with:
        #   1. the meter init block above
        #   2. the logging block in the train loop
        # ════════════════════════════════════════════════════════════════════════
        self.game_successes.clear()
        self.game_closest_kp.clear()
        self.game_true_obj.clear()
        # ════════════════════════════════════════════════════════════════════════

        # ════════════════════════════════════════════════════════════════════════
        # GENERAL ENV-INFO CLEAR — reset epoch-level state each epoch.
        # _ep_cumulative: NOT cleared — per-env running sums must survive epoch
        #   boundaries so mid-episode accumulation is preserved.
        # _ep_cumulative_avg: NOT cleared — deques persist across epochs (same
        #   as game_rewards AverageMeter), giving a rolling window of last N eps.
        # _direct_info: cleared — comes from the most-recent rollout step; stale
        #   after the epoch ends, refreshed on the first play_steps call.
        # _new_finished_episodes: reset so _log_env_info only logs when there is
        #   genuinely new data this epoch.
        # ════════════════════════════════════════════════════════════════════════
        self._direct_info.clear()
        self._new_finished_episodes = False
        # ════════════════════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════════════════════════
    # GENERAL ENV-INFO PROCESSING — mirrors RLGPUAlgoObserver.process_infos().
    # Called every rollout step from play_steps(). Accumulates episode_cumulative
    # data and updates _direct_info with scalar extras from the infos dict.
    # ════════════════════════════════════════════════════════════════════════════
    def _process_infos(self, infos: dict, env_done_indices: "torch.Tensor") -> None:
        # 1. episode_cumulative: accumulate per-env, snapshot at episode end
        if "episode_cumulative" in infos:
            for key, value in infos["episode_cumulative"].items():
                if key not in self._ep_cumulative:
                    self._ep_cumulative[key] = torch.zeros_like(value)
                self._ep_cumulative[key] += value
            for done_idx in env_done_indices:
                self._new_finished_episodes = True
                idx = done_idx.item()
                for key in infos["episode_cumulative"]:
                    if key not in self._ep_cumulative_avg:
                        self._ep_cumulative_avg[key] = self._deque(
                            [], maxlen=self.cfg.games_to_track
                        )
                    self._ep_cumulative_avg[key].append(
                        self._ep_cumulative[key][idx].item()
                    )
                    self._ep_cumulative[key][idx] = 0.0

        # 2. Flatten all scalar extras → _direct_info (overwrites each step, same as rl_games)
        from isaacgymenvs.utils.utils import flatten_dict
        # Tensor keys whose per-env values are handled by AverageMeter or step-3 below
        _SKIP = {
            "successes", "closest_keypoint_max_dist", "true_objective",
            "time_outs", "episode_cumulative", "rewards_episode",
        }
        infos_flat = flatten_dict(infos, prefix="", separator="/")
        self._direct_info = {}
        for k, v in infos_flat.items():
            if k.split("/")[0] in _SKIP:
                continue
            if isinstance(v, (float, int)):
                self._direct_info[k] = v
            elif isinstance(v, torch.Tensor) and v.numel() == 1:
                self._direct_info[k] = v.item()

        # 3. SIMTOOLREAL-SPECIFIC tensor metrics
        # -----------------------------------------------------------------------------
        # These names come from the current SimToolReal env extras. They are not generic PPO
        # concepts. If another env needs structured task metrics, add a new explicit path instead
        # of piggybacking on these keys.
        for tag in ["successes", "closest_keypoint_max_dist", "true_objective"]:
            if tag in infos:
                t = infos[tag]
                if isinstance(t, torch.Tensor) and t.numel() > 1:
                    self._direct_info[f"{tag}_median"] = torch.median(t).item()
                    self._direct_info[f"{tag}_max"] = t.max().item()
                    if tag == "true_objective":
                        self._direct_info["true_objective_mean"] = t.mean().item()

        # 4. SIMTOOLREAL-SPECIFIC per-block success logging
        # -----------------------------------------------------------------------------
        # This assumes the env exports a dense `successes` tensor aligned with the SAPG/EPO block
        # layout. That is true for SimToolReal and for some debugging envs, but this is not a
        # general-purpose abstraction for arbitrary tasks.
        if "successes" in infos and self.block_game_rewards is not None:
            t = infos["successes"]
            if isinstance(t, torch.Tensor):
                M = len(self.block_game_rewards)
                block_size = t.shape[0] // M
                for k in range(M):
                    block_t = t[k * block_size : (k + 1) * block_size]
                    self._direct_info[f"successes_per_block/block_{k}"] = block_t.mean().item()

        self._update_epo_interval_stats(infos=infos, env_done_indices=env_done_indices)

    def _update_epo_interval_stats(
        self, infos: dict, env_done_indices: "torch.Tensor"
    ) -> None:
        """Track per-block best true objective within the current EPO interval.

        This is intentionally tied to envs that emit a `true_objective` tensor in `infos`.
        In practice that means SimToolReal today. The original EPO repo also keys its
        selection logic off `true_objective`, so we keep the same env-specific assumption here.

        EPO repo parity note:
        - The original `EPOObserver.process_infos()` has a fallback path when
          `true_objective` is absent:
              if self.best_embeddings:
                  self.target_objective_known = self.algo.game_rewards.current_size >= self.algo.games_to_track
                  if self.target_objective_known:
                      self.curr_target_objective_value = float(self.algo.mean_rewards)
        - We intentionally do NOT copy that fallback here.

        Why not:
        - For SimToolReal, `true_objective` is the metric we actually care about for EPO
          selection. Silently falling back to reward would make the evolutionary rule depend on
          task-specific reward shaping and could hide a broken env-info contract.
        - The original fallback is useful for broader task coverage, but for this repo it is
          safer to fail "quietly inactive" than to evolve on the wrong signal.

        If we ever want broader cross-task EPO parity, the first thing to consider is adding an
        explicit hard-coded fallback branch here that mirrors the original repo exactly. That
        would look roughly like:
            if "true_objective" not in infos:
                self._epo_target_objective_known = (
                    self.game_rewards.current_size >= self.cfg.games_to_track
                )
                if self._epo_target_objective_known:
                    # rank blocks from reward meters instead of true_objective
                    ...
                return

        For now, keep this path SimToolReal-specific and explicit.
        """
        if self.cfg.epo is None or self.cfg.sapg is None:
            return
        if "true_objective" not in infos:
            return
        true_objective = infos["true_objective"]
        if not isinstance(true_objective, torch.Tensor):
            return

        if self._epo_last_true_objectives is None:
            num_envs = int(true_objective.shape[0])
            self._epo_last_true_objectives = [float("-inf")] * num_envs
            self._epo_best_objective_curr_interval_per_block = [
                float("-inf")
            ] * self.cfg.sapg.num_conditionings

        done_indices_list = env_done_indices.tolist()
        self._epo_finished_agents.update(int(idx) for idx in done_indices_list)
        for done_idx in done_indices_list:
            self._epo_last_true_objectives[done_idx] = float(
                true_objective[done_idx].item()
            )

        block_size = len(self._epo_last_true_objectives) // self.cfg.sapg.num_conditionings
        self._epo_target_objective_known = (
            len(self._epo_finished_agents) >= len(self._epo_last_true_objectives)
        )
        if not self._epo_target_objective_known:
            return

        for block_idx in range(self.cfg.sapg.num_conditionings):
            start = block_idx * block_size
            end = (block_idx + 1) * block_size
            block_mean = float(np.mean(self._epo_last_true_objectives[start:end]))
            if (
                self._epo_best_objective_curr_interval_per_block is not None
                and block_mean
                > self._epo_best_objective_curr_interval_per_block[block_idx]
            ):
                self._epo_best_objective_curr_interval_per_block[block_idx] = block_mean

    def _maybe_run_evolutionary_update(self) -> None:
        """Match the original EPO repo's frame-based observer-triggered evolution."""
        if self.cfg.epo is None or self.cfg.sapg is None:
            return

        current_frame = self.frame
        interval_frames = self.cfg.epo.evolution_interval_frames
        if self._epo_interval_idx == -1:
            self._epo_interval_idx = current_frame // interval_frames

        interval_idx = current_frame // interval_frames
        if interval_idx <= self._epo_interval_idx:
            return
        if not self._epo_target_objective_known:
            print("EPO: waiting for more finished episodes before evolutionary update")
            return
        if current_frame < self.cfg.epo.evolution_warmup_frames:
            print("EPO: warm-up active, skipping evolutionary update")
            return

        self._run_evolutionary_update()
        self._epo_interval_idx = interval_idx

    # ════════════════════════════════════════════════════════════════════════════
    # GENERAL ENV-INFO LOGGING — mirrors RLGPUAlgoObserver.after_print_stats().
    # Called once per epoch from the train loop. Logs episode_cumulative metrics
    # and all scalar extras collected in _direct_info during the rollout.
    # ════════════════════════════════════════════════════════════════════════════
    def _log_env_info(self, frame: int, epoch_num: int, total_time: float) -> None:
        import numpy as np

        # episode_cumulative (only log when new episodes finished this epoch)
        if self._new_finished_episodes:
            for key, dq in self._ep_cumulative_avg.items():
                if dq:
                    self.writer.add_scalar(
                        f"episode_cumulative/{key}", np.mean(dq), frame
                    )
                    self.writer.add_scalar(
                        f"episode_cumulative_min/{key}_min", np.min(dq), frame
                    )
                    self.writer.add_scalar(
                        f"episode_cumulative_max/{key}_max", np.max(dq), frame
                    )
            self._new_finished_episodes = False

        # direct_info scalars — log with /frame, /iter, /time suffixes.
        # All use frame as global_step (matching rl_games convention) so WandB
        # x-axes align when comparing runs from both training backends.
        for k, v in self._direct_info.items():
            self.writer.add_scalar(f"{k}/frame", v, frame)
            self.writer.add_scalar(f"{k}/iter",  v, frame)
            self.writer.add_scalar(f"{k}/time",  v, frame)

        # stdout pretty-print (mirrors rl_games after_print_stats format)
        if self._direct_info:
            for k, v in self._direct_info.items():
                v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"  {k:<50}: {v_str}")
            print()
    # ════════════════════════════════════════════════════════════════════════════

    def update_epoch(self) -> int:
        self.epoch_num += 1
        return self.epoch_num

    def train(self) -> Tuple[float, int]:
        self.init_tensors()
        self.last_mean_rewards = -100500
        total_time = 0
        self.obs_dict = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.cfg.multi_gpu:
            torch.cuda.set_device(self.local_rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            if self.has_asymmetric_critic:
                model_params.append(self.asymmetric_critic_net.state_dict())
            dist.broadcast_object_list(model_params, src=0)
            self.model.load_state_dict(model_params[0])
            if self.has_asymmetric_critic:
                self.asymmetric_critic_net.load_state_dict(model_params[1])

        while True:
            epoch_num = self.update_epoch()
            (
                step_time,
                play_time,
                update_time,
                sum_time,
                a_losses,
                c_losses,
                b_losses,
                entropies,
                kls,
                current_lr,
                lr_mul,
                csigmas,
            ) = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            if self.global_rank == 0:
                # do we need scaled_time?
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = (
                    self.curr_frames * self.world_size
                    if self.cfg.multi_gpu
                    else self.curr_frames
                )
                self.frame += curr_frames

                print_statistics(
                    print_stats=self.cfg.print_stats,
                    curr_frames=curr_frames,
                    step_time=step_time,
                    step_inference_time=scaled_play_time,
                    total_time=scaled_time,
                    epoch_num=epoch_num,
                    max_epochs=self.cfg.max_epochs,
                    frame=frame,
                    max_frames=self.cfg.max_frames,
                )

                self.write_stats(
                    total_time=total_time,
                    epoch_num=epoch_num,
                    step_time=step_time,
                    play_time=play_time,
                    update_time=update_time,
                    a_losses=a_losses,
                    c_losses=c_losses,
                    entropies=entropies,
                    kls=kls,
                    current_lr=current_lr,
                    lr_mul=lr_mul,
                    frame=frame,
                    scaled_time=scaled_time,
                    scaled_play_time=scaled_play_time,
                    curr_frames=curr_frames,
                    csigmas=csigmas,
                )

                if len(b_losses) > 0:
                    self.writer.add_scalar(
                        tag="losses/bounds_loss",
                        scalar_value=torch.mean(torch.stack(b_losses)).item(),
                        global_step=frame,
                    )

                if self.cfg.multi_gpu:
                    # Gather state from all gpus
                    state = self.get_full_state_weights()
                    state_list = [None] * self.world_size
                    dist.gather_object(state, state_list)
                    all_state_dict = {i: state_list[i] for i in range(self.world_size)}
                else:
                    all_state_dict = None

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]

                    for i in range(self.value_size):
                        rewards_name = "rewards" if i == 0 else "rewards{0}".format(i)
                        self.writer.add_scalar(
                            tag=rewards_name + "/step".format(),
                            scalar_value=mean_rewards[i],
                            global_step=frame,
                        )
                        self.writer.add_scalar(
                            tag=rewards_name + "/iter".format(),
                            scalar_value=mean_rewards[i],
                            global_step=frame,
                        )
                        self.writer.add_scalar(
                            tag=rewards_name + "/time".format(),
                            scalar_value=mean_rewards[i],
                            global_step=frame,
                        )
                        self.writer.add_scalar(
                            tag="shaped_" + rewards_name + "/step".format(),
                            scalar_value=mean_shaped_rewards[i],
                            global_step=frame,
                        )
                        self.writer.add_scalar(
                            tag="shaped_" + rewards_name + "/iter".format(),
                            scalar_value=mean_shaped_rewards[i],
                            global_step=frame,
                        )
                        self.writer.add_scalar(
                            tag="shaped_" + rewards_name + "/time".format(),
                            scalar_value=mean_shaped_rewards[i],
                            global_step=frame,
                        )

                    self.writer.add_scalar(
                        tag="episode_lengths/step",
                        scalar_value=mean_lengths,
                        global_step=frame,
                    )
                    self.writer.add_scalar(
                        tag="episode_lengths/iter",
                        scalar_value=mean_lengths,
                        global_step=frame,
                    )
                    self.writer.add_scalar(
                        tag="episode_lengths/time",
                        scalar_value=mean_lengths,
                        global_step=frame,
                    )

                    # ════════════════════════════════════════════════════════════
                    # ENV-SPECIFIC LOGGING — SimToolReal task metrics.
                    # Keys must match the env `infos` dict (env.py extras).
                    # Add / remove entries here when the task changes.
                    # ════════════════════════════════════════════════════════════
                    _ENV_INFO_TAGS = [
                        ("successes",               self.game_successes),
                        ("closest_keypoint_max_dist", self.game_closest_kp),
                        ("true_objective",          self.game_true_obj),
                    ]
                    for _tag, _meter in _ENV_INFO_TAGS:
                        if _meter.current_size > 0:
                            _mval = _meter.get_mean()[0].item()
                            self.writer.add_scalar(f"{_tag}/step",  _mval, frame)
                            self.writer.add_scalar(f"{_tag}/iter",  _mval, frame)
                            self.writer.add_scalar(f"{_tag}/time",  _mval, frame)
                    # ════════════════════════════════════════════════════════════

                    # General env-info logging (episode_cumulative, scalar extras, median/max)
                    self._log_env_info(frame, epoch_num, total_time)

                    if self.cfg.save_frequency > 0:
                        if epoch_num % self.cfg.save_frequency == 0:
                            self.save(
                                filename=(
                                    self.nn_dir
                                    / f"ep_{epoch_num}_rew_{mean_rewards[0]}.pth"
                                ),
                                override_state=all_state_dict,
                            )

                    if (
                        mean_rewards[0] > self.last_mean_rewards
                        and epoch_num >= self.cfg.save_best_after
                    ):
                        print("saving next best rewards: ", mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(
                            filename=(self.nn_dir / "best.pth"),
                            override_state=all_state_dict,
                        )

                if epoch_num >= self.cfg.max_epochs and self.cfg.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print(
                            "WARNING: Max epochs reached before any env terminated at least once"
                        )
                        mean_rewards = -np.inf

                    self.save(
                        filename=(
                            self.nn_dir
                            / f"last_ep_{epoch_num}_rew_{str(mean_rewards).replace('[', '_').replace(']', '_')}.pth"
                        ),
                        override_state=all_state_dict,
                    )
                    print("MAX EPOCHS NUM!")
                    should_exit = True

                if self.frame >= self.cfg.max_frames and self.cfg.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print(
                            "WARNING: Max frames reached before any env terminated at least once"
                        )
                        mean_rewards = -np.inf

                    self.save(
                        filename=(
                            self.nn_dir
                            / f"last_frame_{self.frame}_rew_{str(mean_rewards).replace('[', '_').replace(']', '_')}.pth"
                        ),
                        override_state=all_state_dict,
                    )
                    print("MAX FRAMES NUM!")
                    should_exit = True

                update_time = 0
            else:
                if self.cfg.multi_gpu:
                    state = self.get_full_state_weights()
                    dist.gather_object(state)

            if self.cfg.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, src=0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

    def prepare_dataset(self, batch_dict: Dict[str, Any]) -> None:
        obses = batch_dict["obses"]
        returns = batch_dict["returns"]
        dones = batch_dict["dones"]
        values = batch_dict["values"]
        actions = batch_dict["actions"]
        neglogpacs = batch_dict["neglogpacs"]
        mus = batch_dict["mus"]
        sigmas = batch_dict["sigmas"]
        rnn_states = batch_dict.get("rnn_states", None)

        advantages = returns - values

        if self.cfg.normalize_value:
            if self.cfg.freeze_critic:
                self.value_mean_std.eval()
            else:
                self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()

        advantages = torch.sum(advantages, dim=1)

        if self.cfg.normalize_advantage:
            if self.cfg.multi_gpu:
                mean, var, _ = torch_utils.distributed_mean_var_count(
                    mean=advantages.mean(), var=advantages.var(), count=len(advantages)
                )
                std = torch.sqrt(var)
            else:
                mean = advantages.mean()
                std = advantages.std()
            advantages = (advantages - mean) / (std + 1e-8)

        dataset_dict = {}
        dataset_dict["old_values"] = values
        dataset_dict["old_logp_actions"] = neglogpacs
        dataset_dict["advantages"] = advantages
        dataset_dict["returns"] = returns
        dataset_dict["actions"] = actions
        dataset_dict["obs"] = obses
        dataset_dict["dones"] = dones
        dataset_dict["rnn_states"] = rnn_states
        dataset_dict["mu"] = mus
        dataset_dict["sigma"] = sigmas

        self.dataset.update_values_dict(dataset_dict)

        if self.has_asymmetric_critic:
            dataset_dict = {}
            dataset_dict["old_values"] = values
            dataset_dict["advantages"] = advantages
            dataset_dict["returns"] = returns
            dataset_dict["actions"] = actions
            dataset_dict["states"] = batch_dict["states"]
            dataset_dict["dones"] = dones
            self.asymmetric_critic_net.update_dataset(dataset_dict)

    def train_epoch(
        self,
    ) -> Tuple[
        float,
        float,
        float,
        float,
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        float,
        float,
        List[torch.Tensor],  # csigmas (EPO sigma tracking)
    ]:
        ## A2CBase ##
        self.env.set_train_info(self.frame, self)

        ## ContinuousA2CBase ##
        self.set_eval()
        play_time_start = time.perf_counter()
        with torch.no_grad():
            batch_dict, play_steps_extras = self.play_steps()

            if self.cfg.sapg is not None and self.cfg.sapg.use_others_experience:
                batch_dict = self.augment_batch_using_others_experience(
                    batch_dict=batch_dict, play_steps_extras=play_steps_extras
                )
            # Always shuffle for SAPG to balance block representation across minibatches.
            # rl_games does this unconditionally for mixed_expl; without it minibatches
            # are contiguous per block (e.g. mb0 = all block-0 samples, mb3 = all block-5)
            # which destabilises per-block gradient updates.
            if self.cfg.sapg is not None:
                batch_dict = shuffle_batch(
                    batch_dict=batch_dict, seq_length=self.cfg.seq_length
                )

        play_time_end = time.perf_counter()
        update_time_start = time.perf_counter()

        self.set_train()
        self.curr_frames = batch_dict.pop("played_frames")
        self.prepare_dataset(batch_dict)
        self._maybe_run_evolutionary_update()
        if self.has_asymmetric_critic:
            self.train_asymmetric_critic()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []
        csigmas = []  # EPO: track mean action sigma per update

        for mini_ep in range(0, self.cfg.mini_epochs):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, current_lr, lr_mul, cmu, csigma, b_loss = (
                    self.train_actor_critic(self.dataset[i])
                )
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                csigmas.append(csigma.detach().mean())
                if self.cfg.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)
                if self.cfg.schedule_type == "legacy":
                    av_kls = kl
                    if self.cfg.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.world_size
                    self.current_lr, self.current_entropy_coef = self.scheduler.update(
                        current_lr=self.current_lr,
                        entropy_coef=self.current_entropy_coef,
                        epoch=self.epoch_num,
                        frames=0,
                        kl_dist=av_kls.item(),
                    )
                    self.update_lr(self.current_lr)

            av_kls = torch.mean(torch.stack(ep_kls))
            if self.cfg.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.world_size
            if self.cfg.schedule_type == "standard":
                self.current_lr, self.current_entropy_coef = self.scheduler.update(
                    current_lr=self.current_lr,
                    entropy_coef=self.current_entropy_coef,
                    epoch=self.epoch_num,
                    frames=0,
                    kl_dist=av_kls.item(),
                )
                self.update_lr(self.current_lr)

            kls.append(av_kls)
            if self.cfg.normalize_input:
                self.model.running_mean_std.eval()  # don't need to update statstics more than one miniepoch

        update_time_end = time.perf_counter()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return (
            batch_dict["step_time"],
            play_time,
            update_time,
            total_time,
            a_losses,
            c_losses,
            b_losses,
            entropies,
            kls,
            current_lr,
            lr_mul,
            csigmas,
        )

    def train_actor_critic(
        self, input_dict: Dict[str, Any]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
        float,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute gradients needed to step the networks of the algorithm.

        Core algo logic is defined here

        Args:
            input_dict (:obj:`dict`): Algo inputs as a dict.

        """
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        returns_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]

        lr_mul = 1.0
        curr_e_clip = self.cfg.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }

        if self.is_rnn:
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.cfg.seq_length

            if self.cfg.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]

        with torch.cuda.amp.autocast(enabled=self.cfg.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropies = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]

            N = mu.shape[0]

            ratio = torch.exp(old_action_log_probs_batch - action_log_probs)
            surr1 = advantage * ratio
            surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
            a_losses = torch.max(-surr1, -surr2)

            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -curr_e_clip, curr_e_clip
            )
            value_losses = (values - returns_batch) ** 2
            value_losses_clipped = (value_pred_clipped - returns_batch) ** 2
            c_losses = torch.max(value_losses, value_losses_clipped)
            c_losses = c_losses.squeeze(dim=1)

            if self.cfg.bounds_loss_coef is not None:
                if self.cfg.bound_loss_type == "regularisation":
                    b_losses = (mu * mu).sum(dim=-1)
                elif self.cfg.bound_loss_type == "bound":
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                    b_losses = (mu_loss_low + mu_loss_high).sum(dim=-1)
                else:
                    raise ValueError(
                        f"Unknown bound loss type {self.cfg.bound_loss_type}"
                    )
            else:
                b_losses = torch.zeros(N, device=self.device)

            assert a_losses.shape == (N,), (
                f"a_losses shape mismatch: {a_losses.shape} != {(N,)}"
            )
            assert c_losses.shape == (N,), (
                f"c_losses shape mismatch: {c_losses.shape} != {(N,)}"
            )
            assert entropies.shape == (N,), (
                f"entropies shape mismatch: {entropies.shape} != {(N,)}"
            )
            assert b_losses.shape == (N,), (
                f"b_losses shape mismatch: {b_losses.shape} != {(N,)}"
            )

            a_loss, c_loss, b_loss = (
                a_losses.mean(),
                c_losses.mean(),
                b_losses.mean(),
            )

            if self.uses_intrinsic_reward:
                # Per-sample entropy coefficient: look up from conditioning index in obs
                conditioning_idx = obs_batch[:, -CONDITIONING_IDX_DIM].long()  # (N,)
                entropy_coef_per_sample = self.intr_reward_coef_per_block[
                    conditioning_idx
                ]  # (N,)
                entropy = entropies.mean()
                entropy_loss = (entropy_coef_per_sample * entropies).mean()
            else:
                entropy = entropies.mean()
                entropy_loss = entropy * self.current_entropy_coef

            loss = (
                a_loss
                + 0.5 * c_loss * self.cfg.critic_coef
                - entropy_loss
                + (b_loss * self.cfg.bounds_loss_coef if self.cfg.bounds_loss_coef is not None else 0.0)
            )
            if self.cfg.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        self.truncate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = True
            kl_dist = torch_utils.policy_kl(
                mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl
            )

        return (
            a_loss,
            c_loss,
            entropy,
            kl_dist,
            self.current_lr,
            lr_mul,
            mu.detach(),
            sigma.detach(),
            b_loss,
        )

    def get_asymmetric_critic_value(self, obs_dict: Dict[str, Any]) -> torch.Tensor:
        return self.asymmetric_critic_net.get_value(obs_dict)

    def train_asymmetric_critic(self) -> float:
        return self.asymmetric_critic_net.train_net()

    def get_full_state_weights(self) -> Dict[str, Any]:
        state = self.get_weights()
        state["epoch"] = self.epoch_num
        state["frame"] = self.frame
        state["optimizer"] = self.optimizer.state_dict()

        if self.has_asymmetric_critic:
            state["assymetric_vf_nets"] = self.asymmetric_critic_net.state_dict()

        # This is actually the best reward ever achieved. last_mean_rewards is perhaps not the best variable name
        # We save it to the checkpoint to prevent overriding the "best ever" checkpoint upon experiment restart
        state["last_mean_rewards"] = self.last_mean_rewards

        if self.env is not None:
            env_state = self.env.get_env_state()
            state["env_state"] = env_state

        return state

    def set_full_state_weights(self, weights, set_epoch=True) -> None:
        self.set_weights(weights)
        if set_epoch:
            self.epoch_num = weights["epoch"]
            self.frame = weights["frame"]

        if self.has_asymmetric_critic:
            self.asymmetric_critic_net.load_state_dict(weights["assymetric_vf_nets"])

        self.optimizer.load_state_dict(weights["optimizer"])

        self.last_mean_rewards = weights.get("last_mean_rewards", -1000000000)

        if self.env is not None:
            env_state = weights.get("env_state", None)
            self.env.set_env_state(env_state)

    def get_weights(self) -> Dict[str, Any]:
        state = self.get_stats_weights()
        state["model"] = self.model.state_dict()
        return state

    def get_stats_weights(self, model_stats=False) -> Dict[str, Any]:
        state = {}
        if self.cfg.mixed_precision:
            state["scaler"] = self.scaler.state_dict()
        if self.has_asymmetric_critic:
            state["central_val_stats"] = self.asymmetric_critic_net.get_stats_weights(
                model_stats
            )
        if model_stats:
            if self.cfg.normalize_input:
                state["running_mean_std"] = self.model.running_mean_std.state_dict()
            if self.cfg.normalize_value:
                state["reward_mean_std"] = self.model.value_mean_std.state_dict()

        return state

    def set_stats_weights(self, weights) -> None:
        if self.cfg.normalize_input and "running_mean_std" in weights:
            self.model.running_mean_std.load_state_dict(weights["running_mean_std"])
        if self.cfg.normalize_value and "normalize_value" in weights:
            self.model.value_mean_std.load_state_dict(weights["reward_mean_std"])
        if self.cfg.mixed_precision and "scaler" in weights:
            self.scaler.load_state_dict(weights["scaler"])

    def set_weights(self, weights) -> None:
        self.model.load_state_dict(weights["model"])
        self.set_stats_weights(weights)

    def play_steps(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        update_list = self.update_list
        step_time = 0.0

        if self.is_rnn:
            mb_rnn_states = self.mb_rnn_states
            rnn_state_buffer = [
                torch.zeros(
                    (self.cfg.horizon_length, *s.shape), dtype=s.dtype, device=s.device
                )
                for s in self.rnn_states
            ]

        for n in range(self.cfg.horizon_length):
            if self.is_rnn:
                if n % self.cfg.seq_length == 0:
                    for s, mb_s in zip(self.rnn_states, mb_rnn_states):
                        mb_s[n // self.cfg.seq_length, :, :, :] = s

                for i, s in enumerate(self.rnn_states):
                    rnn_state_buffer[i][n, :, :, :] = s

                if self.has_asymmetric_critic:
                    self.asymmetric_critic_net.pre_step_rnn(n)

            res_dict = self.get_action_values(obs_dict=self.obs_dict)

            if self.is_rnn:
                self.rnn_states = res_dict["rnn_states"]

            self.experience_buffer.update_data("obses", n, self.obs_dict["obs"])
            self.experience_buffer.update_data("dones", n, self.dones.byte())

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            if self.has_asymmetric_critic:
                self.experience_buffer.update_data("states", n, self.obs_dict["states"])

            if self.uses_entropy_in_returns:
                mu = res_dict["mus"]
                sigma = res_dict["sigmas"]
                entropy = (
                    torch.distributions.Normal(mu, sigma)
                    .entropy()
                    .sum(dim=-1, keepdim=True)
                )  # (N, 1)
                self.experience_buffer.update_data("intr_rewards", n, entropy)

            step_time_start = time.perf_counter()
            self.obs_dict, rewards, self.dones, infos = self.env_step(
                res_dict["actions"]
            )
            step_time_end = time.perf_counter()

            step_time += step_time_end - step_time_start

            shaped_rewards = self.rewards_shaper(rewards)

            if self.cfg.value_bootstrap and "time_outs" in infos:
                shaped_rewards += (
                    self.cfg.gamma
                    * res_dict["values"]
                    * self.cast_obs(infos["time_outs"]).unsqueeze(1).float()
                )

            self.experience_buffer.update_data("rewards", n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False).flatten()
            env_done_indices = all_done_indices[:: self.num_agents]

            if self.is_rnn and len(all_done_indices) > 0:
                if self.cfg.zero_rnn_on_done:
                    for s in self.rnn_states:
                        s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0
                if self.has_asymmetric_critic:
                    self.asymmetric_critic_net.post_step_rnn(all_done_indices)

            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(
                self.current_shaped_rewards[env_done_indices]
            )
            self.game_lengths.update(self.current_lengths[env_done_indices].unsqueeze(-1))

            # ════════════════════════════════════════════════════════════════════
            # ENV-SPECIFIC UPDATE — sample task metrics at episode boundaries.
            # Keys must match what the env puts in the `infos` dict (env.py extras).
            # ════════════════════════════════════════════════════════════════════
            if len(env_done_indices) > 0:
                for _meter, _key in [
                    (self.game_successes,  "successes"),
                    (self.game_closest_kp, "closest_keypoint_max_dist"),
                    (self.game_true_obj,   "true_objective"),
                ]:
                    _val = infos.get(_key, None)
                    if _val is not None:
                        _meter.update(_val[env_done_indices].unsqueeze(-1))
            # ════════════════════════════════════════════════════════════════════

            # General env-info processing (scalars, episode_cumulative, median/max stats)
            self._process_infos(infos, env_done_indices)

            if self.block_game_rewards is not None and len(env_done_indices) > 0:
                M = self.cfg.sapg.num_conditionings
                block_size = self.cfg.num_actors // M
                for k in range(M):
                    block_done = env_done_indices[
                        (env_done_indices >= k * block_size)
                        & (env_done_indices < (k + 1) * block_size)
                    ]
                    if len(block_done) > 0:
                        self.block_game_rewards[k].update(
                            self.current_rewards[block_done]
                        )

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = (
                self.current_shaped_rewards * not_dones.unsqueeze(1)
            )
            self.current_lengths = self.current_lengths * not_dones

        last_values = self.get_values(
            obs_dict=self.obs_dict, rnn_states=self.rnn_states
        )

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()

        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_intr_rewards = self.experience_buffer.tensor_dict.get("intr_rewards", None)
        if mb_intr_rewards is not None:
            # intr_reward_coef: (num_actors,) → broadcast to (H, num_actors, 1)
            mb_total_rewards = (
                mb_rewards
                + self.intr_reward_coef.unsqueeze(0).unsqueeze(-1) * mb_intr_rewards
            )
        else:
            mb_total_rewards = mb_rewards
        mb_advs = self.discount_values(
            fdones=fdones,
            last_extrinsic_values=last_values,
            mb_fdones=mb_fdones,
            mb_extrinsic_values=mb_values,
            mb_rewards=mb_total_rewards,
        )
        mb_returns = mb_advs + mb_values
        batch_dict = self.experience_buffer.get_transformed_list(
            transform_op=swap_and_flatten01, tensor_list=self.tensor_list
        )

        batch_dict["returns"] = swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size
        batch_dict["step_time"] = step_time

        if self.is_rnn:
            states = []
            for mb_s in mb_rnn_states:
                t_size = mb_s.size()[0] * mb_s.size()[2]
                h_size = mb_s.size()[3]
                states.append(mb_s.permute(1, 2, 0, 3).reshape(-1, t_size, h_size))
            batch_dict["rnn_states"] = states

        extras = {
            "rewards": mb_rewards,
            "obs": self.experience_buffer.tensor_dict["obses"],
            "last_obs_dict": self.obs_dict,
            "states": self.experience_buffer.tensor_dict.get("states", None),
            "dones": mb_fdones,
            "last_dones": fdones,
            "rnn_states": rnn_state_buffer if self.is_rnn else None,
            "last_rnn_states": self.rnn_states,
            "mb_extr_rewards": mb_rewards,
            "mb_intr_rewards": mb_intr_rewards,  # SAPG: per-step entropy, or None
        }

        return batch_dict, extras

    def augment_batch_using_others_experience(
        self,
        batch_dict: Dict[str, Any],
        play_steps_extras: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Augment a SAPG rollout with re-labeled experience from other conditioning blocks.

        This mirrors the local rl_games mixed-expl leader-follower update:
        - sample one or more follower blocks in addition to the leader block
        - re-label obs/state conditioning as if those samples belonged to another block
        - recompute value targets under that re-labeled conditioning
        - keep the full leader rollout but only the matching follower slice from each repeated
          follower rollout

        Layout assumptions:
        - flattened rollout tensors concatenate repeated rollouts along axis 0
        - recurrent hidden states concatenate repeated rollouts along axis 1
        """
        new_batch_dict = {}
        num_blocks = self.cfg.sapg.num_conditionings
        block_size = self.env.num_envs // num_blocks

        num_repeat = min(num_blocks, int(self.cfg.sapg.off_policy_ratio) + 1)
        sampled_block_idxs = [0] + [
            int(x)
            for x in np.random.choice(
                range(1, num_blocks), size=num_repeat - 1, replace=False
            )
        ]
        if self.cfg.multi_gpu:
            dist.broadcast_object_list(sampled_block_idxs, src=0)

        for key, val in batch_dict.items():
            if key in ["played_frames", "step_time"]:
                new_batch_dict[key] = val
            elif key == "obses":
                intr_coef_embd = torch.cat(
                    [
                        torch.roll(
                            self.conditioning_idxs, shifts=block_size * i, dims=0
                        )
                        for i in sampled_block_idxs
                    ],
                    dim=0,
                )
                obses = torch.cat([val] * len(sampled_block_idxs), dim=0)
                obses[:, -CONDITIONING_IDX_DIM:] = intr_coef_embd.repeat_interleave(
                    self.cfg.horizon_length, dim=0
                )
                mask = torch.zeros(len(obses), dtype=torch.bool, device=obses.device)
                mask[len(val) :] = True
                obses = filter_leader(
                    val=obses,
                    orig_len=len(val),
                    sampled_block_idxs=sampled_block_idxs,
                    num_blocks=num_blocks,
                )
                mask = filter_leader(
                    val=mask,
                    orig_len=len(val),
                    sampled_block_idxs=sampled_block_idxs,
                    num_blocks=num_blocks,
                )
                new_batch_dict[key] = obses
                new_batch_dict["off_policy_mask"] = mask
            elif key == "states":
                if val is None:
                    new_batch_dict[key] = None
                elif self.conditioning_idxs is not None:
                    # Roll the conditioning idx in states for each sampled block
                    intr_coef_embd = torch.cat(
                        [
                            torch.roll(
                                self.conditioning_idxs,
                                shifts=block_size * i,
                                dims=0,
                            )
                            for i in sampled_block_idxs
                        ],
                        dim=0,
                    )
                    states = torch.cat([val] * len(sampled_block_idxs), dim=0)
                    states[:, -CONDITIONING_IDX_DIM:] = (
                        intr_coef_embd.repeat_interleave(self.cfg.horizon_length, dim=0)
                    )
                    new_batch_dict[key] = filter_leader(
                        val=states,
                        orig_len=len(val),
                        sampled_block_idxs=sampled_block_idxs,
                        num_blocks=num_blocks,
                    )
                else:
                    new_batch_dict[key] = filter_leader(
                        val=torch.cat([val] * len(sampled_block_idxs), dim=0),
                        orig_len=len(val),
                        sampled_block_idxs=sampled_block_idxs,
                        num_blocks=num_blocks,
                    )
            elif key in ["values", "returns"]:
                pass  # handled below
            elif key == "rnn_states":
                if val is not None:
                    repeated = [
                        torch.cat([val[i]] * len(sampled_block_idxs), dim=1)
                        for i in range(len(val))
                    ]
                    new_batch_dict[key] = [
                        filter_leader(
                            val=repeated[i],
                            orig_len=val[i].shape[1],
                            sampled_block_idxs=sampled_block_idxs,
                            num_blocks=num_blocks,
                            axis=1,
                        )
                        for i in range(len(val))
                    ]
                else:
                    new_batch_dict[key] = None
            else:
                new_batch_dict[key] = filter_leader(
                    val=torch.cat([val] * len(sampled_block_idxs), dim=0),
                    orig_len=len(val),
                    sampled_block_idxs=sampled_block_idxs,
                    num_blocks=num_blocks,
                )

        new_returns_list = [batch_dict["returns"]]
        new_values_list = [batch_dict["values"]]

        for block_idx in sampled_block_idxs[1:]:
            mb_rewards = play_steps_extras["rewards"]
            mb_obs = play_steps_extras["obs"]
            last_obs_dict = play_steps_extras["last_obs_dict"]
            last_rnn_states = play_steps_extras["last_rnn_states"]
            mb_states = play_steps_extras["states"]
            mb_rnn_states = play_steps_extras["rnn_states"]

            rolled_conditioning = torch.roll(
                self.conditioning_idxs, shifts=block_size * block_idx, dims=0
            )
            mb_obs[:, :, -CONDITIONING_IDX_DIM:] = rolled_conditioning
            last_obs_dict["obs"][:, -CONDITIONING_IDX_DIM:] = rolled_conditioning

            # Also update states conditioning if using asymmetric critic
            if mb_states is not None and self.conditioning_idxs is not None:
                mb_states[:, :, -CONDITIONING_IDX_DIM:] = rolled_conditioning
            if "states" in last_obs_dict and last_obs_dict["states"] is not None and self.conditioning_idxs is not None:
                last_obs_dict["states"][:, -CONDITIONING_IDX_DIM:] = rolled_conditioning

            flattened_rnn_states = (
                [
                    rnn_s.transpose(0, 1).reshape(
                        rnn_s.transpose(0, 1).shape[0], -1, *rnn_s.shape[3:]
                    )
                    for rnn_s in mb_rnn_states
                ]
                if mb_rnn_states is not None
                else None
            )

            flattened_mb_obs = mb_obs.reshape(-1, *mb_obs.shape[2:])
            flattened_mb_states = (
                mb_states.reshape(-1, *mb_states.shape[2:])
                if mb_states is not None
                else None
            )

            mb_values = []
            CHUNK_SIZE = 8192  # Can't do this in one pass because OOM
            num_chunks = math.ceil(flattened_mb_obs.shape[0] / CHUNK_SIZE)
            for i in range(num_chunks):
                start_idx = i * CHUNK_SIZE
                end_idx = (i + 1) * CHUNK_SIZE
                if end_idx >= flattened_mb_obs.shape[0]:
                    end_idx = flattened_mb_obs.shape[0]

                mb_values.append(
                    self.get_values(
                        obs_dict={
                            "obs": flattened_mb_obs[start_idx:end_idx],
                            "states": flattened_mb_states[start_idx:end_idx]
                            if mb_states is not None
                            else None,
                        },
                        rnn_states=[
                            s[:, start_idx:end_idx] for s in flattened_rnn_states
                        ]
                        if flattened_rnn_states is not None
                        else None,
                    )
                )

            mb_values = torch.cat(mb_values, dim=0)
            last_values = self.get_values(
                obs_dict=last_obs_dict, rnn_states=last_rnn_states
            )

            mb_values = mb_values.reshape(*mb_obs.shape[:2], *mb_values.shape[1:])
            mb_values = torch.cat([mb_values, last_values.unsqueeze(0)], dim=0)

            mb_fdones = play_steps_extras["dones"]
            fdones = play_steps_extras["last_dones"]

            mb_fdones = torch.cat([mb_fdones, fdones.unsqueeze(0)], dim=0)
            mb_intr_rewards = play_steps_extras["mb_intr_rewards"]
            if mb_intr_rewards is not None:
                # Roll entropy coefficients to match this off-policy block's assignment
                rolled_intr_coef = torch.roll(
                    self.intr_reward_coef, shifts=block_size * block_idx, dims=0
                )  # (num_actors,)
                mb_returns = (
                    mb_rewards
                    + rolled_intr_coef.unsqueeze(0).unsqueeze(-1) * mb_intr_rewards
                    + self.cfg.gamma * mb_values[1:] * (1 - mb_fdones[1:]).unsqueeze(-1)
                )
            else:
                mb_returns = mb_rewards + self.cfg.gamma * mb_values[1:] * (
                    1 - mb_fdones[1:]
                ).unsqueeze(-1)

            new_returns_list.append(swap_and_flatten01(mb_returns))
            new_values_list.append(swap_and_flatten01(mb_values[:-1]))

        new_batch_dict["returns"] = filter_leader(
            val=torch.cat(new_returns_list, dim=0),
            orig_len=len(batch_dict["returns"]),
            sampled_block_idxs=sampled_block_idxs,
            num_blocks=num_blocks,
        )
        new_batch_dict["values"] = filter_leader(
            val=torch.cat(new_values_list, dim=0),
            orig_len=len(batch_dict["values"]),
            sampled_block_idxs=sampled_block_idxs,
            num_blocks=num_blocks,
        )

        # Reset obs / states conditioning back to original in play_steps_extras
        # (they were mutated in the per-block loop above)
        play_steps_extras["obs"][:, :, -CONDITIONING_IDX_DIM:] = self.conditioning_idxs
        play_steps_extras["last_obs_dict"]["obs"][:, -CONDITIONING_IDX_DIM:] = (
            self.conditioning_idxs
        )
        if play_steps_extras["states"] is not None and self.conditioning_idxs is not None:
            play_steps_extras["states"][:, :, -CONDITIONING_IDX_DIM:] = self.conditioning_idxs
        last_obs_dict = play_steps_extras["last_obs_dict"]
        if "states" in last_obs_dict and last_obs_dict["states"] is not None and self.conditioning_idxs is not None:
            last_obs_dict["states"][:, -CONDITIONING_IDX_DIM:] = self.conditioning_idxs

        return new_batch_dict

    def _run_evolutionary_update(self) -> None:
        """EPO: evolutionary update step.

        Match the original EPO repo:
        - rank blocks by best true objective within the current interval
        - average the top-k parent embeddings
        - replace the configured number of worst policies
        - reset interval trackers afterwards
        """
        assert self.cfg.sapg is not None, "_run_evolutionary_update requires sapg config"
        assert self.cfg.epo is not None, "_run_evolutionary_update requires epo config"
        if not self._epo_target_objective_known:
            return
        assert (
            self._epo_best_objective_curr_interval_per_block is not None
        ), "EPO interval stats must be initialized before update"

        num_blocks = self.cfg.sapg.num_conditionings
        num_best = min(self.cfg.epo.num_best_policies, num_blocks)
        num_replace = min(self.cfg.epo.num_replace_policies, num_blocks - 1)
        if num_best < 1 or num_replace < 1:
            return

        objectives = self._epo_best_objective_curr_interval_per_block
        sorted_idx = sorted(range(num_blocks), key=lambda i: objectives[i], reverse=True)
        best_policies = sorted_idx[:num_best]
        worst_policies = sorted_idx[-num_replace:]

        # EPO repo parity note:
        # - The original repo does NOT mutate the existing embedding tensor in place.
        # - Instead, `merge_extra_params()` clones the tensor and rebinds:
        #       extra_params_copy = self.extra_params.detach().clone()
        #       extra_params_copy[replaced_policy] = merged_params
        #       self.extra_params = nn.Parameter(extra_params_copy, requires_grad=True)
        #
        # Why we do NOT copy that here:
        # - The optimizer in the original repo is created earlier from `self.model.parameters()`.
        # - Rebinding to a fresh `nn.Parameter` without rebuilding the optimizer can leave the
        #   optimizer param groups pointing at the old object instead of the new one.
        # - That means the post-replacement embedding may stop receiving optimizer updates.
        #
        # So even though rebinding is the literal upstream code, it looks much more like a bug
        # than an intended algorithmic choice. For local parity we keep the algorithmic part
        # (top-2 averaging into the worst policy) but apply it safely by mutating the existing
        # parameter storage in place.
        #
        # If we ever need exact bug-for-bug emulation for debugging, the upstream-style code
        # would be:
        #     old = self.model.a2c_network.conditioning
        #     cloned = old.detach().clone()
        #     cloned[replaced_policy] = merged_params
        #     self.model.a2c_network.conditioning = nn.Parameter(cloned, requires_grad=True)
        # But do not do that in normal training without also rebuilding the optimizer.
        cond_matrix = self.model.a2c_network.conditioning.data  # (M, C)
        merged_params = cond_matrix[best_policies].mean(dim=0)
        for replaced_policy in worst_policies:
            cond_matrix[replaced_policy] = merged_params

        self._epo_best_objective_curr_interval_per_block = [float("-inf")] * num_blocks
        self._epo_target_objective_known = False

        # BUG_FROM_EPO_REPO compatibility switch:
        # See module-level constant for the full explanation. This is intentionally hard-coded
        # instead of exposed through task config because it is a repo-specific bug-emulation knob,
        # not a meaningful training hyperparameter.
        if not BUG_FROM_EPO_REPO_KEEP_FINISHED_AGENTS_AFTER_REPLACEMENT:
            self._epo_finished_agents.clear()

        print(
            f"EPO evolution: best_policies={best_policies}, "
            f"replaced={worst_policies}"
        )

    def save(
        self, filename: Path, override_state: Optional[Dict[str, Any]] = None
    ) -> None:
        state = {self.global_rank: self.get_full_state_weights()}
        if override_state is not None:
            state = override_state
        torch_utils.save_checkpoint(filename=filename, state=state)

    def restore(self, filename: Path, set_epoch: bool = True) -> None:
        checkpoint = torch_utils.load_checkpoint(filename)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def override_sigma(self, sigma) -> None:
        net = self.model.network
        if hasattr(net, "sigma") and hasattr(net, "fixed_sigma"):
            if net.fixed_sigma:
                with torch.no_grad():
                    net.sigma.fill_(float(sigma))
            else:
                print("Print cannot set new sigma because fixed_sigma is False")
        else:
            print("Print cannot set new sigma because sigma is not present")
