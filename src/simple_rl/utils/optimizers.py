"""Optimizer helpers for simple_rl."""

from __future__ import annotations

from typing import Iterable, Literal

import torch
from torch import Tensor
from torch.optim import Optimizer

OptimizerName = Literal["adam", "adamw", "muon"]

_NS_COEFS = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]


def _zeropower_via_newtonschulz5(grad: Tensor, eps: float = 1e-7) -> Tensor:
    """Project a matrix gradient toward an orthogonal update, matching PufferLib's Muon."""
    grad = grad.clone()
    x = grad if grad.size(-2) <= grad.size(-1) else grad.mT
    x = x / torch.clamp(grad.norm(dim=(-2, -1), keepdim=True), min=eps)

    for a, b, c in _NS_COEFS:
        s = x @ x.mT
        y = c * s
        y.diagonal(dim1=-2, dim2=-1).add_(b)
        y = y @ s
        y.diagonal(dim1=-2, dim2=-1).add_(a)
        x = y @ x

    return x if grad.size(-2) <= grad.size(-1) else x.mT


def _lr_to_scalar(lr: float | Tensor) -> float:
    if isinstance(lr, Tensor):
        if lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        return float(lr.item())
    return float(lr)


class Muon(Optimizer):
    """Simple Muon optimizer adapted from PufferLib's MIT-licensed implementation.

    This variant applies Muon to all parameters, which matches PufferLib's RL usage.
    For tensors with fewer than 2 dims, the step reduces to momentum SGD.
    """

    def __init__(
        self,
        params,
        lr: float = 0.0025,
        momentum: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Momentum should be >= 0 but is: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Weight decay should be >= 0 but is: {weight_decay}")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = _lr_to_scalar(group["lr"])
            momentum = group["momentum"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        grad, memory_format=torch.preserve_format
                    )

                buf = state["momentum_buffer"]
                buf.mul_(momentum)
                buf.add_(grad)
                grad = grad.add(buf, alpha=momentum)

                if grad.ndim >= 2:
                    grad_2d = grad.reshape(grad.shape[0], -1)
                    grad_2d = _zeropower_via_newtonschulz5(grad_2d, eps=eps)
                    grad = grad_2d * max(1.0, grad_2d.size(-2) / grad_2d.size(-1)) ** 0.5
                    grad = grad.reshape_as(param)

                if weight_decay != 0.0:
                    param.mul_(1 - lr * weight_decay)
                param.sub_(grad, alpha=lr)

        return loss


def build_optimizer(
    params: Iterable[Tensor],
    optimizer_name: OptimizerName,
    learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> Optimizer:
    if optimizer_name == "adam":
        return torch.optim.Adam(
            params,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
    if optimizer_name == "muon":
        return Muon(
            params,
            lr=learning_rate,
            momentum=beta1,
            eps=eps,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")
