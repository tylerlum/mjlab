import os
from typing import Tuple, Union

import torch
import torch.nn as nn

from simple_rl.utils import torch_utils

"""
Updates statistics from full data.
"""


class RunningMeanStd(nn.Module):
    def __init__(
        self,
        insize: Union[int, Tuple[int, ...]],
        epsilon: float = 1e-05,
        per_channel: bool = False,
        norm_only: bool = False,
    ) -> None:
        super(RunningMeanStd, self).__init__()
        print("RunningMeanStd: ", insize)
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0, 2, 3]
            if len(self.insize) == 2:
                self.axis = [0, 2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0]
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(in_size, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    def _update_mean_var_count_from_moments(
        self,
        mean: torch.Tensor,
        var: torch.Tensor,
        count: torch.Tensor,
        batch_mean: torch.Tensor,
        batch_var: torch.Tensor,
        batch_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Multi-GPU
        if os.getenv("LOCAL_RANK") and os.getenv("WORLD_SIZE"):
            batch_mean, batch_var, batch_count = torch_utils.distributed_mean_var_count(
                mean=batch_mean, var=batch_var, count=batch_count
            )

        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count

    def forward(
        self,
        input: torch.Tensor,
        denorm: bool = False,
    ) -> torch.Tensor:
        if self.training:
            mean = input.mean(self.axis)  # along channel axis
            var = input.var(self.axis)
            self.running_mean, self.running_var, self.count = (
                self._update_mean_var_count_from_moments(
                    mean=self.running_mean,
                    var=self.running_var,
                    count=self.count,
                    batch_mean=mean,
                    batch_var=var,
                    batch_count=input.size()[0],
                )
            )

        # Change shape
        if self.per_channel:
            if len(self.insize) == 3:
                current_mean = self.running_mean.view(
                    [1, self.insize[0], 1, 1]
                ).expand_as(input)
                current_var = self.running_var.view(
                    [1, self.insize[0], 1, 1]
                ).expand_as(input)
            if len(self.insize) == 2:
                current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(
                    input
                )
                current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(
                    input
                )
            if len(self.insize) == 1:
                current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(
                    input
                )
                current_var = self.running_var.view([1, self.insize[0]]).expand_as(
                    input
                )
        else:
            current_mean = self.running_mean
            current_var = self.running_var
        # Get output

        if denorm:
            y = torch.clamp(input, min=-5.0, max=5.0)
            y = (
                torch.sqrt(current_var.float() + self.epsilon) * y
                + current_mean.float()
            )
        else:
            if self.norm_only:
                y = input / torch.sqrt(current_var.float() + self.epsilon)
            else:
                y = (input - current_mean.float()) / torch.sqrt(
                    current_var.float() + self.epsilon
                )
                y = torch.clamp(y, min=-5.0, max=5.0)
        return y
