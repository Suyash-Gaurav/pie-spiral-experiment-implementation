"""
Transformer Model with π-Spiral Positional Encoding

Complete transformer implementation that integrates π-Spiral encoding
and attractor state mechanism for long-context processing.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

from .attention import PiSpiralAttention


class PiSpiralTransformerBlock(nn.Module):
    """
    Single transformer block with π-Spiral attention
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension (default: 4 * d_model)
        dropout: Dropout probability
        activation: Activation function ('relu', 'gelu', 'swish')
        pos_encoding: Type of positional encoding
        use_attractor: Whether to use attractor state
        attractor_inject: How to inject attractor
        irrational: Irrational constant for π-Spiral
        hybrid_K: Transition threshold for hybrid encoding
        layer_norm_eps: Epsilon for layer normalization
        device: Device to place tensors on
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: Literal['relu', 'gelu', 'swish'] = 'gelu',
        pos_encoding: str = 'pi_spiral',
        use_attractor: bool = True,
        attractor_inject: str = 'cross_attn',
        irrational: str = 'pi',
        hybrid_K: int = 16000,
        layer_norm_eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff or 4 * d_model
        
        # Self-attention with π-Spiral encoding
        self.self_attn = PiSpiralAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            pos_encoding=pos_encoding,
            use_attractor=use_attractor,
            attractor_inject=attractor_inject,
            irrational=irrational,
            hybrid_K=hybrid_K,
            device=device,
            dtype=dtype,
        )
        
        # Feed-forward network
        self.ff = FeedForward(
            d_model=d_model,
            d_ff=self.d_ff,
            dropout=dropout,
            activation=activation,
            device=device,
            dtype=dtype,
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, device=device, dtype=dtype)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        use_flash: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through transformer block
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            attention_mask: Optional attention mask
            positions: Optional position indices
            use_flash: Whether to use flash attention
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and layer norm (pre-norm)
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attn(x, attention_mask, positions, use_flash)
        x = residual + self.dropout(attn_output)
        
        # Feed-forward with residual connection and layer norm (pre-norm)
        residual = x
        x = self.norm2(x)
        ff_output = self.ff(x)
        x = residual + self.dropout(ff_output)
        
        return x


class FeedForward(nn.Module):
    """
    Feed-forward network with configurable activation
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        activation: Activation function
        device: Device to place tensors on
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: Literal['relu', 'gelu', 'swish'] = 'gelu',
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff, device=device, dtype=dtype)
        self.linear2 = nn.Linear(d_ff, d_model, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # SiLU is equivalent to Swish
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor of same shape
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class PiSpiralTransformer(nn.Module):
    """
    Complete transformer model with π-Spiral positional encoding
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        activation: Activation function
        pos_encoding: Type of positional encoding
        use_attractor: Whether to use attractor state
        attractor_inject: How to inject attractor
        inject_layers: Which layers to inject attractor ('all', 'last_N', list of indices)
        N_inject: Number of last layers to inject (if inject_layers='last_N')
        irrational: Irrational constant for π-Spiral
        hybrid_K: Transition threshold for hybrid encoding
        tie_weights: Whether to tie input and output embeddings
        device: Device to place tensors on
        dtype: Data type for tensors
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        max_seq_len: int = 100000,
        dropout: float = 0.1,
        activation: str = 'gelu',
        pos_encoding: str = 'pi_spiral',
        use_attractor: bool = True,
        attractor_inject: str = 'cross_attn',
        inject_layers: str = 'last_N',
        N_inject: int = 4,
        irrational: str = 'pi',
        hybrid_K: int = 16000,
        tie_weights: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        
        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        
        # Determine which layers should use attractor
        if inject_layers == 'all':
            attractor_layers = set(range(num_layers))
        elif inject_layers == 'last_N':
            attractor_layers = set(range(num_layers - N_inject, num_layers))
        elif isinstance(inject_layers, (list, tuple)):
            attractor_layers = set(inject_layers)
        else:
            attractor_layers = set()
        
        # Transformer layers
        self.layers = nn.ModuleList([
            PiSpiralTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                pos_encoding=pos_encoding,
                use_attractor=(i in attractor_layers) and use_attractor,
                attractor_inject=attractor_inject if (i in attractor_layers) else 'none',
                irrational=irrational,
                hybrid_K=hybrid_K,
                device=device,
                dtype=dtype,
            )
            for i in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        
        # Output projection
        self.output_projection = nn.Linear(
            d_model, vocab_size, bias=False, device=device, dtype=dtype
        )
        
        # Tie weights if specified
        if tie_weights:
            self.output_projection.weight = self.token_embedding.weight
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        use_flash: bool = False,
        return_hidden_states: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            positions: Optional position indices
            use_flash: Whether to use flash attention
            return_hidden_states: Whether to return all hidden states
        
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
            or tuple of (logits, hidden_states) if return_hidden_states=True
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        x = self.dropout(x)
        
        # Prepare attention mask if provided
        if attention_mask is not None:
            # Convert to additive mask (0 for valid, -inf for masked)
            attention_mask = (1.0 - attention_mask) * -1e9
        
        # Store hidden states if requested
        all_hidden_states = [] if return_hidden_states else None
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask, positions, use_flash)
            if return_hidden_states:
                all_hidden_states.append(x)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        if return_hidden_states:
            return logits, all_hidden_states
        
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        use_flash: bool = False,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            use_flash: Whether to use flash attention
        
        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits = self.forward(input_ids, use_flash=use_flash)
                
                # Get logits for last token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
