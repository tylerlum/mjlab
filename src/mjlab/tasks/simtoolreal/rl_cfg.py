"""RL runner configuration for the SimToolReal MJLab port."""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def simtoolreal_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the initial PPO runner path for SimToolReal.

  The original SimToolReal policy was trained with rl_games/SAPG. MJLab's native
  runner path is RSL-RL PPO today; this config is intentionally a runnable baseline
  until the rl_games/simple_rl runner is wired in.
  """
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.8,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.002,
      num_learning_epochs=5,
      num_mini_batches=8,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.995,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="simtoolreal",
    save_interval=100,
    num_steps_per_env=32,
    max_iterations=30_000,
    clip_actions=1.0,
    wandb_tags=("simtoolreal", "mjlab", "ppo-baseline"),
  )
