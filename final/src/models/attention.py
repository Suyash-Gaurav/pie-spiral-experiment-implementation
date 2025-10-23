"""
Attention Mechanisms with π-Spiral Encoding and Attractor State

Implements attention layers that integrate:
- π-Spiral positional encoding
- Cross-attention to O(1) attractor state
- Hybrid positional encoding support
- Flash attention compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal

from ..encodings import PiSpiralPositional, AttractorState, HybridPositionalEncoding


class PiSpiralAttention(nn.Module):
    """
    Multi-head attention with π-Spiral positional encoding
    
    Integrates π-Spiral encoding into the attention mechanism with optional
    cross-attention to attractor state for O(1) global context.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        pos_encoding: Type of positional encoding ('pi_spiral', 'hybrid', 'rope', 'none')
        use_attractor: Whether to use attractor state cross-attention
        attractor_inject: How to inject attractor ('cross_attn', 'residual', 'none')
        irrational: Irrational constant for π-Spiral
        hybrid_K: Transition threshold for hybrid encoding
        max_seq_len: Maximum sequence length
        device: Device to place tensors on
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        pos_encoding: Literal['pi_spiral', 'hybrid', 'rope', 'none'] = 'pi_spiral',
        use_attractor: bool = True,
        attractor_inject: Literal['cross_attn', 'residual', 'none'] = 'cross_attn',
        irrational: str = 'pi',
        hybrid_K: int = 16000,
        max_seq_len: int = 100000,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        self.pos_encoding = pos_encoding
        self.use_attractor = use_attractor
        self.attractor_inject = attractor_inject
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        
        # Scaling factor for attention scores
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        # Initialize positional encoding
        if pos_encoding == 'pi_spiral':
            self.pos_encoder = PiSpiralPositional(
                d_model=d_model,
                max_len=max_seq_len,
                irrational=irrational,
                device=device,
                dtype=dtype,
            )
        elif pos_encoding == 'hybrid':
            self.pos_encoder = HybridPositionalEncoding(
                d_model=d_model,
                max_seq_len=max_seq_len,
                hybrid_K=hybrid_K,
                irrational=irrational,
                device=device,
                dtype=dtype,
            )
        else:
            self.pos_encoder = None
        
        # Initialize attractor state if needed
        if use_attractor and attractor_inject != 'none':
            self.attractor = AttractorState(
                d_model=d_model,
                device=device,
                dtype=dtype,
            )
            
            if attractor_inject == 'cross_attn':
                # Cross-attention to attractor state
                self.attractor_cross_attn = AttractorCrossAttention(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                    device=device,
                    dtype=dtype,
                )
            elif attractor_inject == 'residual':
                # Residual adapter for attractor
                self.attractor_adapter = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        else:
            self.attractor = None
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        use_flash: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of π-Spiral attention
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional mask of shape (batch_size, seq_len, seq_len)
            positions: Optional position indices of shape (batch_size, seq_len)
            use_flash: Whether to use flash attention (if available)
        
        Returns:
            Tuple of (output, attention_weights)
            - output: shape (batch_size, seq_len, d_model)
            - attention_weights: shape (batch_size, num_heads, seq_len, seq_len) or None
        """
        batch_size, seq_len, d_model = x.shape
        
        # Apply positional encoding
        if self.pos_encoder is not None:
            pos_encoding = self.pos_encoder.get_encoding(
                positions if positions is not None else torch.arange(seq_len, device=x.device)
            )
            if pos_encoding.dim() == 2:
                pos_encoding = pos_encoding.unsqueeze(0)
            x_with_pos = x + pos_encoding
        else:
            x_with_pos = x
        
        # Project to Q, K, V
        q = self.q_proj(x_with_pos)
        k = self.k_proj(x_with_pos)
        v = self.v_proj(x_with_pos)
        
        # Reshape for multi-head attention
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        # (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention
        if use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's flash attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
            attn_weights = None
        else:
            # Standard attention computation
            attn_output, attn_weights = self._compute_attention(q, k, v, attention_mask)
        
        # Reshape back
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.out_dropout(output)
        
        # Integrate attractor state if enabled
        if self.attractor is not None and self.attractor_inject != 'none':
            # Update attractor state
            if self.pos_encoder is not None:
                pos_enc_for_attractor = pos_encoding
            else:
                pos_enc_for_attractor = torch.zeros_like(x)
            
            attractor_state = self.attractor(x, pos_enc_for_attractor)
            
            if self.attractor_inject == 'cross_attn':
                # Cross-attend to attractor state
                attractor_output = self.attractor_cross_attn(output, attractor_state)
                output = output + attractor_output
            elif self.attractor_inject == 'residual':
                # Add attractor as residual
                # Pool attractor state to sequence dimension
                attractor_pooled = attractor_state.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
                attractor_residual = self.attractor_adapter(attractor_pooled)
                output = output + attractor_residual
        
        return output, attn_weights
    
    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute standard scaled dot-product attention
        
        Args:
            q: Query tensor (batch, num_heads, seq_len, head_dim)
            k: Key tensor (batch, num_heads, seq_len, head_dim)
            v: Value tensor (batch, num_heads, seq_len, head_dim)
            attention_mask: Optional mask (batch, seq_len, seq_len) or (batch, 1, seq_len, seq_len)
        
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        # Compute attention scores
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
        # -> (batch, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # Add head dimension
            attn_scores = attn_scores + attention_mask
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output, attn_weights


class AttractorCrossAttention(nn.Module):
    """
    Cross-attention to attractor state for O(1) global context
    
    Allows the model to attend to the compressed global context stored
    in the attractor state without increasing memory complexity.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        device: Device to place tensors on
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Projections for cross-attention
        self.q_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attractor_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-attend to attractor state
        
        Args:
            x: Query tensor of shape (batch_size, seq_len, d_model)
            attractor_state: Attractor state of shape (batch_size, d_state, d_model)
        
        Returns:
            Cross-attention output of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        _, d_state, _ = attractor_state.shape
        
        # Project queries from x
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        
        # Project keys and values from attractor state
        # Reshape attractor state for projection
        attractor_flat = attractor_state.view(batch_size * d_state, d_model)
        k = self.k_proj(attractor_flat).view(batch_size, d_state, d_model)
        v = self.v_proj(attractor_flat).view(batch_size, d_state, d_model)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, d_state, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, d_state, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute cross-attention scores
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, d_state)
        # -> (batch, num_heads, seq_len, d_state)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (batch, num_heads, seq_len, d_state) @ (batch, num_heads, d_state, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        output = self.out_proj(attn_output)
        
        return output
