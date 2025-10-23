"""
Model Adapters for Injecting π-Spiral Encoding into Pre-trained Models

Provides utilities to inject π-Spiral positional encoding and attractor state
into existing transformer models (Qwen2.5, Llama-3, etc.) without full retraining.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable, List
import logging

from ..encodings import PiSpiralPositional, AttractorState, HybridPositionalEncoding
from .attention import AttractorCrossAttention

logger = logging.getLogger(__name__)


class ModelAdapter:
    """
    Adapter for injecting π-Spiral encoding into pre-trained models
    
    Supports models from Hugging Face transformers library:
    - Qwen2.5 (1.5B, 7B, etc.)
    - Llama-3 (8B, 34B, etc.)
    - Other transformer architectures
    
    Args:
        model: Pre-trained model instance
        pos_encoding: Type of positional encoding ('pi_spiral', 'hybrid', 'rope')
        inject_layers: Which layers to inject attractor ('all', 'last_N', or list)
        N_inject: Number of last layers (if inject_layers='last_N')
        use_attractor: Whether to use attractor state
        attractor_inject: How to inject attractor ('cross_attn', 'residual', 'none')
        irrational: Irrational constant for π-Spiral
        hybrid_K: Transition threshold for hybrid encoding
        window_size: Sliding window size for attention (None for full attention)
    """
    
    def __init__(
        self,
        model: nn.Module,
        pos_encoding: str = 'pi_spiral',
        inject_layers: str = 'last_N',
        N_inject: int = 4,
        use_attractor: bool = True,
        attractor_inject: str = 'cross_attn',
        irrational: str = 'pi',
        hybrid_K: int = 16000,
        window_size: Optional[int] = None,
    ):
        self.model = model
        self.pos_encoding = pos_encoding
        self.inject_layers = inject_layers
        self.N_inject = N_inject
        self.use_attractor = use_attractor
        self.attractor_inject = attractor_inject
        self.irrational = irrational
        self.hybrid_K = hybrid_K
        self.window_size = window_size
        
        # Detect model architecture
        self.model_type = self._detect_model_type()
        logger.info(f"Detected model type: {self.model_type}")
        
        # Get model configuration
        self.config = self._get_model_config()
        
        # Initialize positional encoding
        self.pos_encoder = self._create_positional_encoder()
        
        # Initialize attractor state if needed
        if use_attractor and attractor_inject != 'none':
            self.attractor = AttractorState(
                d_model=self.config['d_model'],
                device=self.model.device,
                dtype=next(self.model.parameters()).dtype,
            )
        else:
            self.attractor = None
        
        # Inject into model layers
        self._inject_into_layers()
    
    def _detect_model_type(self) -> str:
        """Detect the model architecture type"""
        model_class = self.model.__class__.__name__.lower()
        
        if 'qwen' in model_class:
            return 'qwen'
        elif 'llama' in model_class:
            return 'llama'
        elif 'gpt' in model_class:
            return 'gpt'
        elif 'bert' in model_class:
            return 'bert'
        else:
            return 'generic'
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Extract model configuration"""
        config = {}
        
        if hasattr(self.model, 'config'):
            model_config = self.model.config
            config['d_model'] = getattr(model_config, 'hidden_size', 512)
            config['num_layers'] = getattr(model_config, 'num_hidden_layers', 6)
            config['num_heads'] = getattr(model_config, 'num_attention_heads', 8)
            config['max_seq_len'] = getattr(model_config, 'max_position_embeddings', 100000)
        else:
            # Default values
            config['d_model'] = 512
            config['num_layers'] = 6
            config['num_heads'] = 8
            config['max_seq_len'] = 100000
        
        return config
    
    def _create_positional_encoder(self):
        """Create positional encoder based on configuration"""
        if self.pos_encoding == 'pi_spiral':
            return PiSpiralPositional(
                d_model=self.config['d_model'],
                max_len=self.config['max_seq_len'],
                irrational=self.irrational,
                device=self.model.device,
                dtype=next(self.model.parameters()).dtype,
            )
        elif self.pos_encoding == 'hybrid':
            return HybridPositionalEncoding(
                d_model=self.config['d_model'],
                max_seq_len=self.config['max_seq_len'],
                hybrid_K=self.hybrid_K,
                irrational=self.irrational,
                device=self.model.device,
                dtype=next(self.model.parameters()).dtype,
            )
        else:
            return None
    
    def _inject_into_layers(self):
        """Inject π-Spiral encoding and attractor into model layers"""
        # Determine which layers to inject
        num_layers = self.config['num_layers']
        
        if self.inject_layers == 'all':
            target_layers = list(range(num_layers))
        elif self.inject_layers == 'last_N':
            target_layers = list(range(num_layers - self.N_inject, num_layers))
        elif isinstance(self.inject_layers, (list, tuple)):
            target_layers = list(self.inject_layers)
        else:
            target_layers = []
        
        logger.info(f"Injecting π-Spiral into layers: {target_layers}")
        
        # Get layer modules
        layer_modules = self._get_layer_modules()
        
        # Inject into each target layer
        for layer_idx in target_layers:
            if layer_idx < len(layer_modules):
                self._inject_into_layer(layer_modules[layer_idx], layer_idx)
    
    def _get_layer_modules(self) -> List[nn.Module]:
        """Get list of transformer layer modules"""
        if self.model_type == 'qwen':
            if hasattr(self.model, 'transformer'):
                return list(self.model.transformer.h)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                return list(self.model.model.layers)
        elif self.model_type == 'llama':
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                return list(self.model.model.layers)
        elif self.model_type == 'gpt':
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
                return list(self.model.transformer.h)
        
        # Generic fallback
        if hasattr(self.model, 'layers'):
            return list(self.model.layers)
        
        return []
    
    def _inject_into_layer(self, layer: nn.Module, layer_idx: int):
        """Inject π-Spiral encoding into a single layer"""
        # Store original forward method
        original_forward = layer.forward
        
        # Create wrapper that adds π-Spiral encoding
        def wrapped_forward(*args, **kwargs):
            # Get hidden states (first argument or 'hidden_states' kwarg)
            if len(args) > 0:
                hidden_states = args[0]
            else:
                hidden_states = kwargs.get('hidden_states')
            
            # Apply positional encoding if available
            if self.pos_encoder is not None and hidden_states is not None:
                batch_size, seq_len, _ = hidden_states.shape
                positions = torch.arange(seq_len, device=hidden_states.device)
                pos_encoding = self.pos_encoder.get_encoding(positions)
                if pos_encoding.dim() == 2:
                    pos_encoding = pos_encoding.unsqueeze(0)
                
                # Add positional encoding
                hidden_states = hidden_states + pos_encoding
                
                # Update args or kwargs
                if len(args) > 0:
                    args = (hidden_states,) + args[1:]
                else:
                    kwargs['hidden_states'] = hidden_states
            
            # Call original forward
            output = original_forward(*args, **kwargs)
            
            # Inject attractor if enabled
            if self.attractor is not None and self.attractor_inject != 'none':
                # Extract hidden states from output
                if isinstance(output, tuple):
                    layer_output = output[0]
                else:
                    layer_output = output
                
                # Update attractor state
                if self.pos_encoder is not None:
                    pos_enc = pos_encoding
                else:
                    pos_enc = torch.zeros_like(layer_output)
                
                attractor_state = self.attractor(layer_output, pos_enc)
                
                # Inject attractor output
                if self.attractor_inject == 'cross_attn':
                    # Create cross-attention module if not exists
                    if not hasattr(layer, '_pi_spiral_cross_attn'):
                        layer._pi_spiral_cross_attn = AttractorCrossAttention(
                            d_model=self.config['d_model'],
                            num_heads=self.config['num_heads'],
                            device=layer_output.device,
                            dtype=layer_output.dtype,
                        )
                    
                    attractor_output = layer._pi_spiral_cross_attn(layer_output, attractor_state)
                    layer_output = layer_output + attractor_output
                
                elif self.attractor_inject == 'residual':
                    # Simple residual connection
                    attractor_pooled = attractor_state.mean(dim=1, keepdim=True)
                    attractor_pooled = attractor_pooled.expand(-1, layer_output.shape[1], -1)
                    layer_output = layer_output + attractor_pooled
                
                # Update output
                if isinstance(output, tuple):
                    output = (layer_output,) + output[1:]
                else:
                    output = layer_output
            
            return output
        
        # Replace forward method
        layer.forward = wrapped_forward
        logger.info(f"Injected π-Spiral into layer {layer_idx}")


def inject_pi_spiral_encoding(
    model: nn.Module,
    pos_encoding: str = 'pi_spiral',
    inject_layers: str = 'last_N',
    N_inject: int = 4,
    use_attractor: bool = True,
    attractor_inject: str = 'cross_attn',
    irrational: str = 'pi',
    hybrid_K: int = 16000,
    window_size: Optional[int] = None,
) -> ModelAdapter:
    """
    Convenience function to inject π-Spiral encoding into a model
    
    Args:
        model: Pre-trained model instance
        pos_encoding: Type of positional encoding
        inject_layers: Which layers to inject
        N_inject: Number of last layers
        use_attractor: Whether to use attractor state
        attractor_inject: How to inject attractor
        irrational: Irrational constant
        hybrid_K: Transition threshold
        window_size: Sliding window size
    
    Returns:
        ModelAdapter instance
    
    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
        >>> adapter = inject_pi_spiral_encoding(model, pos_encoding='hybrid')
    """
    adapter = ModelAdapter(
        model=model,
        pos_encoding=pos_encoding,
        inject_layers=inject_layers,
        N_inject=N_inject,
        use_attractor=use_attractor,
        attractor_inject=attractor_inject,
        irrational=irrational,
        hybrid_K=hybrid_K,
        window_size=window_size,
    )
    
    return adapter


class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation (LoRA) for efficient fine-tuning with π-Spiral
    
    Adds low-rank matrices to attention layers for parameter-efficient adaptation.
    
    Args:
        original_layer: Original linear layer
        rank: Rank of low-rank matrices
        alpha: Scaling factor
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation"""
        # Original output
        original_output = self.original_layer(x)
        
        # LoRA output
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B
        lora_output = lora_output * self.scaling
        
        return original_output + lora_output
