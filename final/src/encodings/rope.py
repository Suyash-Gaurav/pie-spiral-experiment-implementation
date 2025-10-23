"""
Rotary Positional Embedding (RoPE) Implementation

Standard RoPE implementation for baseline comparison and hybrid encoding.
Based on the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding"

Key features:
- Rotary position encoding for relative position information
- Supports standard RoPE and RoPE-NTK (Neural Tangent Kernel) scaling
- Efficient implementation using complex number rotations
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE)
    
    Encodes position information by rotating query and key vectors in the attention mechanism.
    Treats pairs of features as complex numbers and applies position-dependent rotations.
    
    Args:
        dim: Dimension of the embedding (should be even)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation (default: 10000)
        scaling_factor: Scaling factor for RoPE-NTK extension (default: 1.0)
        device: Device to place tensors on
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 100000,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be even for RoPE, got {dim}")
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        
        # Compute frequency bands
        # inv_freq = 1 / (base^(2i/dim)) for i in [0, dim/2)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=self.device) / dim)
        )
        self.register_buffer('inv_freq', inv_freq)
        
        # Pre-compute rotation matrices for efficiency
        self._compute_rotation_cache(max_seq_len)
    
    def _compute_rotation_cache(self, seq_len: int):
        """
        Pre-compute rotation matrices for positions 0 to seq_len-1
        
        Args:
            seq_len: Sequence length to cache
        """
        # Position indices
        t = torch.arange(seq_len, dtype=torch.float32, device=self.device)
        
        # Apply scaling for RoPE-NTK
        t = t / self.scaling_factor
        
        # Compute frequencies: outer product of positions and inverse frequencies
        # Shape: (seq_len, dim/2)
        freqs = torch.outer(t, self.inv_freq)
        
        # Create rotation matrix using cos and sin
        # Shape: (seq_len, dim)
        cos_cached = freqs.cos().to(self.dtype)
        sin_cached = freqs.sin().to(self.dtype)
        
        self.register_buffer('cos_cached', cos_cached)
        self.register_buffer('sin_cached', sin_cached)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half the hidden dimensions
        
        For input [x1, x2, x3, x4, ...], returns [-x2, x1, -x4, x3, ...]
        This implements the rotation in complex plane
        
        Args:
            x: Input tensor of shape (..., dim)
        
        Returns:
            Rotated tensor of same shape
        """
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).flatten(-2)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary positional embedding to query and key tensors
        
        Args:
            q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
               or (batch_size, num_heads, seq_len, head_dim)
            k: Key tensor of same shape as q
            positions: Optional position indices of shape (batch_size, seq_len)
                      If None, uses sequential positions 0, 1, 2, ...
        
        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs
        """
        # Determine input format and reshape if needed
        if q.dim() == 4:
            if q.shape[1] == k.shape[1] and q.shape[2] != k.shape[2]:
                # Format: (batch, num_heads, seq_len, head_dim)
                batch_size, num_heads, seq_len, head_dim = q.shape
                transpose_format = True
            else:
                # Format: (batch, seq_len, num_heads, head_dim)
                batch_size, seq_len, num_heads, head_dim = q.shape
                transpose_format = False
        else:
            raise ValueError(f"Expected 4D tensor, got shape {q.shape}")
        
        # Get rotation matrices
        if positions is None:
            # Use sequential positions
            if seq_len > self.max_seq_len:
                # Recompute cache for longer sequences
                self._compute_rotation_cache(seq_len)
            
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
        else:
            # Use provided positions
            positions = positions.flatten()
            cos = self.cos_cached[positions]
            sin = self.sin_cached[positions]
            cos = cos.view(batch_size, seq_len, -1)
            sin = sin.view(batch_size, seq_len, -1)
        
        # Reshape for broadcasting
        if transpose_format:
            # (batch, num_heads, seq_len, head_dim)
            cos = cos.unsqueeze(1)  # (batch, 1, seq_len, dim/2) or (seq_len, dim/2)
            sin = sin.unsqueeze(1)
        else:
            # (batch, seq_len, num_heads, head_dim)
            cos = cos.unsqueeze(2)  # (batch, seq_len, 1, dim/2) or (seq_len, dim/2)
            sin = sin.unsqueeze(2)
        
        # Expand cos and sin to match head_dim
        if cos.shape[-1] != head_dim:
            # Repeat for each dimension pair
            cos = cos.repeat_interleave(2, dim=-1)
            sin = sin.repeat_interleave(2, dim=-1)
        
        # Apply rotation: x_rotated = x * cos + rotate_half(x) * sin
        q_rotated = q * cos + self._rotate_half(q) * sin
        k_rotated = k * cos + self._rotate_half(k) * sin
        
        return q_rotated, k_rotated
    
    def forward_inference(
        self,
        x: torch.Tensor,
        position: int
    ) -> torch.Tensor:
        """
        Apply RoPE for a single position (useful for incremental decoding)
        
        Args:
            x: Input tensor of shape (batch_size, num_heads, head_dim)
            position: Position index
        
        Returns:
            Rotated tensor of same shape
        """
        if position >= self.max_seq_len:
            self._compute_rotation_cache(position + 1)
        
        cos = self.cos_cached[position]
        sin = self.sin_cached[position]
        
        # Expand to match input shape
        cos = cos.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0)
        sin = sin.repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(0)
        
        return x * cos + self._rotate_half(x) * sin


class RoPENTK(RotaryPositionalEmbedding):
    """
    RoPE with Neural Tangent Kernel (NTK) scaling for better length extrapolation
    
    Applies a scaling factor to extend the effective context length of RoPE.
    
    Args:
        dim: Dimension of the embedding
        max_seq_len: Maximum sequence length
        base: Base for frequency computation
        original_max_seq_len: Original max sequence length before scaling
        device: Device to place tensors on
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 100000,
        base: float = 10000.0,
        original_max_seq_len: int = 4096,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        # Compute NTK scaling factor
        scaling_factor = max_seq_len / original_max_seq_len
        
        # Adjust base frequency for NTK scaling
        adjusted_base = base * (scaling_factor ** (dim / (dim - 2)))
        
        super().__init__(
            dim=dim,
            max_seq_len=max_seq_len,
            base=adjusted_base,
            scaling_factor=1.0,  # Already incorporated in base
            device=device,
            dtype=dtype,
        )
