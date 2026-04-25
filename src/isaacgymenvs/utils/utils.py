"""Compatibility utilities used by vendored SimToolReal training code."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def flatten_dict(
  value: Mapping[str, Any],
  prefix: str = "",
  separator: str = "/",
) -> dict[str, Any]:
  """Flatten nested dictionaries with the key format expected by simple_rl."""
  items: dict[str, Any] = {}
  for key, child in value.items():
    flat_key = f"{prefix}{separator}{key}" if prefix else str(key)
    if isinstance(child, Mapping):
      items.update(flatten_dict(child, prefix=flat_key, separator=separator))
    else:
      items[flat_key] = child
  return items
