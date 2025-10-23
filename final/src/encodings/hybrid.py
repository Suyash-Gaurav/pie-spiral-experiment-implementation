"""
Hybrid Positional Encoding Implementation

Combines RoPE for short-range positions with π-Spiral for long-range positions
using a smooth sigmoid transition.

Key features:
- RoPE for indices ≤ K (preserves short-range performance)
- π-Spiral for indices > K (enables long-range context)
- Smooth sigmoid blending at the transition boundary
- Configurable transition threshold K
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from .rope import RotaryPositionalEmbedding
from .pi_spiral import PiSpiralPositional


class HybridPositionalEncoding(nn.Module):
    """
    Hybrid Positional Encoding: RoPE + π-Spiral with smooth transition
    
    Uses RoPE for positions ≤ K and blends to π-Spiral for positions > K
    with a smooth sigmoid transition to avoid discontinuities.
    
    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        hybrid_K: Transition threshold (default: 16000)
        transition_width: Width of sigmoid transition region (default: 1000)
        rope_base: Base for RoPE frequency computation
        irrational: Irrational constant for π-Spiral ('pi', 'e', 'sqrt2', 'phi', 'prng')
        device: Device to place tensors on
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 100000,
        hybrid_K: int = 16000,
        transition_width: int = 1000,
        rope_base: float = 10000.0,
        irrational: str = 'pi',
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.hybrid_K = hybrid_K
        self.transition_width = transition_width
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        
        # Initialize RoPE for short-range
        self.rope = RotaryPositionalEmbedding(
            dim=d_model,
            max_seq_len=max_seq_len,
            base=rope_base,
            device=device,
            dtype=dtype,
        )
        
        # Initialize π-Spiral for long-range
        self.pi_spiral = PiSpiralPositional(
            d_model=d_model,
            max_len=max_seq_len,
            irrational=irrational,
            device=device,
            dtype=dtype,
        )
        
        # Pre-compute blending weights
        self._compute_blend_weights(max_seq_len)
    
    def _compute_blend_weights(self, seq_len: int):
        """
        Compute smooth blending weights using sigmoid function
        
        Weight function: w(n) = sigmoid((n - K) / transition_width)
        - w(n) ≈ 0 for n << K (use RoPE)
        - w(n) ≈ 1 for n >> K (use π-Spiral)
        - w(n) ≈ 0.5 at n = K (equal blend)
        
        Args:
            seq_len: Sequence length to compute weights for
        """
        positions = torch.arange(seq_len, dtype=self.dtype, device=self.device)
        
        # Sigmoid blending: 0 for RoPE, 1 for π-Spiral
        blend_weights = torch.sigmoid(
            (positions - self.hybrid_K) / self.transition_width
        )
        
        self.register_buffer('blend_weights', blend_weights)
        self.register_buffer('rope_weights', 1.0 - blend_weights)
    
    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Apply hybrid positional encoding to input tensor
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            positions: Optional position indices of shape (batch_size, seq_len)
            return_components: If True, return (output, rope_component, spiral_component)
        
        Returns:
            Tensor with hybrid positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        if d_model != self.d_model:
            raise ValueError(f"Input d_model {d_model} doesn't match encoding d_model {self.d_model}")
        
        # Recompute blend weights if sequence is longer than cached
        if seq_len > self.blend_weights.shape[0]:
            self._compute_blend_weights(seq_len)
        
        # Get blending weights for this sequence
        if positions is None:
            rope_w = self.rope_weights[:seq_len].view(1, seq_len, 1)
            spiral_w = self.blend_weights[:seq_len].view(1, seq_len, 1)
        else:
            # Use provided positions
            pos_flat = positions.flatten()
            rope_w = self.rope_weights[pos_flat].view(batch_size, seq_len, 1)
            spiral_w = self.blend_weights[pos_flat].view(batch_size, seq_len, 1)
        
        # Get RoPE encoding (applied to input directly)
        # Note: RoPE is typically applied to Q and K in attention, not to embeddings
        # For embedding-style application, we use the positional encoding directly
        rope_encoding = self.pi_spiral.get_encoding(
            positions if positions is not None else torch.arange(seq_len, device=x.device)
        )
        if rope_encoding.dim() == 2:
            rope_encoding = rope_encoding.unsqueeze(0)
        
        # Get π-Spiral encoding
        spiral_encoding = self.pi_spiral.get_encoding(
            positions if positions is not None else torch.arange(seq_len, device=x.device)
        )
        if spiral_encoding.dim() == 2:
            spiral_encoding = spiral_encoding.unsqueeze(0)
        
        # Blend encodings
        blended_encoding = rope_w * rope_encoding + spiral_w * spiral_encoding
        
        # Add to input
        output = x + blended_encoding
        
        if return_components:
            return output, rope_encoding, spiral_encoding
        
        return output
    
    def forward_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply hybrid encoding to query and key tensors for attention
        
        This method applies RoPE-style rotation to Q and K with hybrid blending.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len, num_heads, head_dim)
            k: Key tensor of same shape as q
            positions: Optional position indices
        
        Returns:
            Tuple of (encoded_q, encoded_k)
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Recompute blend weights if needed
        if seq_len > self.blend_weights.shape[0]:
            self._compute_blend_weights(seq_len)
        
        # Get blending weights
        if positions is None:
            rope_w = self.rope_weights[:seq_len].view(1, seq_len, 1, 1)
            spiral_w = self.blend_weights[:seq_len].view(1, seq_len, 1, 1)
            pos_indices = torch.arange(seq_len, device=q.device)
        else:
            pos_flat = positions.flatten()
            rope_w = self.rope_weights[pos_flat].view(batch_size, seq_len, 1, 1)
            spiral_w = self.blend_weights[pos_flat].view(batch_size, seq_len, 1, 1)
            pos_indices = positions
        
        # Apply RoPE
        q_rope, k_rope = self.rope(q, k, pos_indices)
        
        # Apply π-Spiral (as additive encoding)
        spiral_enc = self.pi_spiral.get_encoding(pos_indices)
        if spiral_enc.dim() == 2:
            spiral_enc = spiral_enc.unsqueeze(0)
        spiral_enc = spiral_enc.unsqueeze(2)  # Add head dimension
        
        q_spiral = q + spiral_enc
        k_spiral = k + spiral_enc
        
        # Blend
        q_hybrid = rope_w * q_rope + spiral_w * q_spiral
        k_hybrid = rope_w * k_rope + spiral_w * k_spiral
        
        return q_hybrid, k_hybrid
    
    def get_blend_weight(self, position: int) -> float:
        """
        Get the blending weight for a specific position
        
        Args:
            position: Position index
        
        Returns:
            Blend weight (0 = full RoPE, 1 = full π-Spiral)
        """
        if position >= self.blend_weights.shape[0]:
            self._compute_blend_weights(position + 1)
        
        return self.blend_weights[position].item()
    
    def visualize_blend_weights(self, max_pos: Optional[int] = None) -> torch.Tensor:
        """
        Get blend weights for visualization
        
        Args:
            max_pos: Maximum position to visualize (default: max_seq_len)
        
        Returns:
            Tensor of shape (max_pos,) with blend weights
        """
        max_pos = max_pos or self.max_seq_len
        
        if max_pos > self.blend_weights.shape[0]:
            self._compute_blend_weights(max_pos)
        
        return self.blend_weights[:max_pos]


class AdaptiveHybridEncoding(HybridPositionalEncoding):
    """
    Adaptive Hybrid Encoding with learnable transition parameters
    
    Extends HybridPositionalEncoding with learnable K and transition_width
    to automatically optimize the blending strategy during training.
    
    Args:
        d_model: Model dimension
        max_seq_len: Maximum sequence length
        initial_K: Initial transition threshold
        initial_width: Initial transition width
        rope_base: Base for RoPE frequency computation
        irrational: Irrational constant for π-Spiral
        device: Device to place tensors on
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 100000,
        initial_K: int = 16000,
        initial_width: int = 1000,
        rope_base: float = 10000.0,
        irrational: str = 'pi',
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        # Initialize with fixed parameters first
        super().__init__(
            d_model=d_model,
            max_seq_len=max_seq_len,
            hybrid_K=initial_K,
            transition_width=initial_width,
            rope_base=rope_base,
            irrational=irrational,
            device=device,
            dtype=dtype,
        )
        
        # Make K and transition_width learnable
        self.K_param = nn.Parameter(
            torch.tensor(float(initial_K), dtype=dtype, device=device)
        )
        self.width_param = nn.Parameter(
            torch.tensor(float(initial_width), dtype=dtype, device=device)
        )
    
    def _compute_blend_weights(self, seq_len: int):
        """Compute blend weights using learnable parameters"""
        positions = torch.arange(seq_len, dtype=self.dtype, device=self.device)
        
        # Use learnable parameters (apply softplus to ensure positivity)
        K = torch.nn.functional.softplus(self.K_param)
        width = torch.nn.functional.softplus(self.width_param)
        
        blend_weights = torch.sigmoid((positions - K) / width)
        
        self.register_buffer('blend_weights', blend_weights, persistent=False)
        self.register_buffer('rope_weights', 1.0 - blend_weights, persistent=False)
