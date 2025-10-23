"""
Model Architecture Modules for π-Spiral Experiment

This package contains model implementations that integrate π-Spiral positional encoding:
- Transformer models with π-Spiral encoding
- Attention mechanisms with attractor state
- Model adapters for existing architectures (Qwen, Llama)
"""

from .transformer import PiSpiralTransformer, PiSpiralTransformerBlock
from .attention import PiSpiralAttention, AttractorCrossAttention
from .adapters import ModelAdapter, inject_pi_spiral_encoding

__all__ = [
    'PiSpiralTransformer',
    'PiSpiralTransformerBlock',
    'PiSpiralAttention',
    'AttractorCrossAttention',
    'ModelAdapter',
    'inject_pi_spiral_encoding',
]
