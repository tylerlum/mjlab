"""SAPG helper utilities shared by simple_rl training code.

These functions intentionally mirror the current local rl_games fork used in this repo.
They are not a generic SAPG library: tensor layout assumptions here are specific to the
rollout/minibatch shapes produced by the PPO + LSTM training path.
"""

from typing import Any, Dict, List

import numpy as np
import torch


def create_sinusoidal_encoding(
    genvec: torch.Tensor, embd_size: int, n: int = 100
) -> torch.Tensor:
    """
    Creates sinusoidal positional encoding for a vector of values.
    Used to encode entropy coefficient values into fixed embeddings.

    Args:
        genvec: 1D tensor of values to encode, shape (N,)
        embd_size: Embedding dimension
        n: Base for positional encoding (default 100)

    Returns:
        Tensor of shape (N, embd_size)
    """
    assert genvec.ndim == 1, f"genvec must be 1D, got {genvec.ndim}"
    N = len(genvec)
    encoding = torch.zeros(N, embd_size, device=genvec.device)
    for i in range(embd_size // 2):
        denom = n ** (2 * i / embd_size)
        encoding[:, 2 * i] = torch.sin(genvec / denom)
        encoding[:, 2 * i + 1] = torch.cos(genvec / denom)
    return encoding


def remove_envs_from_info(infos: Dict[str, Any], num_envs: int) -> Dict[str, Any]:
    for key in list(infos.keys()):
        if isinstance(infos[key], dict):
            infos[key] = remove_envs_from_info(infos[key], num_envs)
        elif isinstance(infos[key], list) or isinstance(
            infos[key], (np.ndarray, torch.Tensor)
        ):
            if key in ["successes", "closest_keypoint_max_dist"]:
                block_size = len(infos[key]) - num_envs
                if len(infos[key]) % block_size == 0:
                    for i in range(len(infos[key]) // block_size):
                        infos[f"{key}_per_block/block_{i}"] = infos[key][
                            i * block_size : (i + 1) * block_size
                        ]
            infos[key] = infos[key][num_envs:]
    return infos


def shuffle_batch(batch_dict: Dict[str, Any], seq_length: int) -> Dict[str, Any]:
    """Shuffle a flattened rollout while preserving recurrent sequence chunks.

    Args:
        batch_dict: Flattened PPO batch. Non-RNN tensors are shaped `(N, ...)`. Recurrent
            states are stored as a list of tensors shaped `(num_layers, num_seqs, hidden)`.
        seq_length: Chunk size to preserve during shuffling. This is the RNN `seq_length`,
            not necessarily the full rollout horizon.

    Returns:
        The same dict with all sample tensors permuted consistently in sequence-sized chunks.

    Assumptions:
        - `len(batch_dict["returns"])` is divisible by `seq_length`.
        - For each recurrent-state tensor, axis 1 indexes sequences in the same order as the
          flattened batch's `(batch_size // seq_length)` chunks.
    """
    N = len(batch_dict["returns"])
    assert N % seq_length == 0, (
        f"N={N} must be divisible by seq_length={seq_length}"
    )
    batch_size = N // seq_length
    device = batch_dict["returns"].device

    indices = torch.randperm(batch_size).to(device).reshape(
        batch_size, 1
    ) * seq_length + torch.arange(seq_length).to(device).reshape(
        1, seq_length
    )
    assert indices.shape == (batch_size, seq_length), (
        f"indices.shape={indices.shape} must be (batch_size, seq_length)=({batch_size}, {seq_length})"
    )

    flattened_indices = indices.reshape(-1)
    agent_perm = indices[:, 0] // seq_length  # randperm(batch_size), shape (batch_size,)
    for key, val in batch_dict.items():
        if key == "rnn_states":
            if val is None:
                continue
            shuffled = []
            for s in val:
                # s shape: (num_layers, num_seqs_per_chunk * batch_size, hidden_size)
                # where num_seqs_per_chunk can exceed 1 when the hidden-state buffer stores
                # multiple sequences per shuffled sample group.
                num_seqs_per_agent = s.shape[1] // batch_size
                if num_seqs_per_agent <= 1:
                    seq_idx = agent_perm
                else:
                    # Expand the permutation so each shuffled chunk carries all of its
                    # corresponding recurrent-state slices.
                    seq_idx = (
                        agent_perm.unsqueeze(1) * num_seqs_per_agent
                        + torch.arange(num_seqs_per_agent, device=s.device).unsqueeze(0)
                    ).reshape(-1)
                shuffled.append(s[:, seq_idx])
            batch_dict[key] = shuffled
        elif key in ["played_frames", "step_time"]:
            continue
        else:
            batch_dict[key] = val[flattened_indices]
    return batch_dict


def filter_leader(
    val: torch.Tensor,
    orig_len: int,
    sampled_block_idxs: List[int],
    num_blocks: int,
    axis: int = 0,
) -> torch.Tensor:
    """Keep the leader batch plus the sampled follower blocks used by leader-follower SAPG.

    `augment_batch_using_others_experience()` constructs repeated copies of the rollout, each
    re-labeled with another block's conditioning. Leader-follower mode keeps:
    - the full original leader rollout from repeat index 0
    - only the specific follower block slice from each additional repeated rollout

    Args:
        val: Tensor whose repeated rollouts are concatenated along `axis`.
        orig_len: Length of a single repeated rollout along `axis`, before filtering.
        sampled_block_idxs: Block ids used for each repeated rollout copy. `0` is the leader.
        num_blocks: Number of SAPG blocks.
        axis: Concatenation axis. Most rollout tensors use axis 0. Recurrent hidden states use
            axis 1 because they are shaped `(num_layers, num_seqs, hidden)`.
    """
    assert axis in (0, 1), f"Unsupported axis={axis}; expected 0 or 1"
    block_size = orig_len // num_blocks
    filtered_val_list = []
    for i, block_idx in enumerate(sampled_block_idxs):
        if block_idx == 0:  # Leader
            start_idx = i * orig_len
            end_idx = (i + 1) * orig_len
        else:  # Followers
            start_idx = i * orig_len + (block_idx - 1) * block_size
            end_idx = i * orig_len + block_idx * block_size

        if axis == 0:
            filtered_val = val[start_idx:end_idx]
        else:
            filtered_val = val[:, start_idx:end_idx]
        filtered_val_list.append(filtered_val)

    new_val = torch.cat(filtered_val_list, dim=axis)
    return new_val
