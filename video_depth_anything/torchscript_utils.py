"""
TorchScript-compatible utility functions to replace einops operations.

This module provides native PyTorch implementations of common einops patterns
used in Video Depth Anything, making the model compatible with torch.jit.script().
"""

import torch
from typing import Tuple


def rearrange_bcfhw_to_bfchw(x: torch.Tensor) -> torch.Tensor:
    """
    Rearrange tensor from (b, c, f, h, w) to (b*f, c, h, w).

    Args:
        x: Input tensor of shape (b, c, f, h, w)

    Returns:
        Tensor of shape (b*f, c, h, w)
    """
    b, c, f, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)


def rearrange_bfchw_to_bcfhw(x: torch.Tensor, f: int) -> torch.Tensor:
    """
    Rearrange tensor from (b*f, c, h, w) to (b, c, f, h, w).

    Args:
        x: Input tensor of shape (b*f, c, h, w)
        f: Number of frames

    Returns:
        Tensor of shape (b, c, f, h, w)
    """
    bf, c, h, w = x.shape
    b = bf // f
    return x.reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4)


def rearrange_bfdc_to_bdfc(x: torch.Tensor, f: int) -> torch.Tensor:
    """
    Rearrange tensor from (b*f, d, c) to (b*d, f, c).

    Args:
        x: Input tensor of shape (b*f, d, c)
        f: Number of frames

    Returns:
        Tensor of shape (b*d, f, c)
    """
    bf, d, c = x.shape
    b = bf // f
    # Reshape to (b, f, d, c) then permute to (b, d, f, c) then flatten to (b*d, f, c)
    x = x.reshape(b, f, d, c)
    x = x.permute(0, 2, 1, 3)  # (b, d, f, c)
    x = x.reshape(b * d, f, c)
    return x


def rearrange_bdfc_to_bfdc(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Rearrange tensor from (b*d, f, c) to (b*f, d, c).

    Args:
        x: Input tensor of shape (b*d, f, c)
        d: Spatial dimension (height * width)

    Returns:
        Tensor of shape (b*f, d, c)
    """
    bd, f, c = x.shape
    b = bd // d
    # Reshape to (b, d, f, c) then permute to (b, f, d, c) then flatten to (b*f, d, c)
    x = x.reshape(b, d, f, c)
    x = x.permute(0, 2, 1, 3)  # (b, f, d, c)
    x = x.reshape(b * f, d, c)
    return x


def repeat_bdc_to_bdnc(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Repeat tensor from (b, n, c) to (b*d, n, c).

    Args:
        x: Input tensor of shape (b, n, c)
        d: Number of times to repeat

    Returns:
        Tensor of shape (b*d, n, c)
    """
    b, n, c = x.shape
    # Expand: (b, n, c) -> (b, d, n, c) then reshape to (b*d, n, c)
    x = x.unsqueeze(1).expand(b, d, n, c)
    x = x.reshape(b * d, n, c)
    return x
