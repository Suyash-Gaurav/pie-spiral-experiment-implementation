"""
Diagnostic Tools for π-Spiral Experiment

This package contains diagnostic and analysis tools for:
- Positional collision analysis
- Spectral analysis of positional encodings
- Attention visualization
- Perplexity drift analysis
- Memory profiling
"""

from .encoding_analysis import EncodingAnalyzer, compare_multiple_encodings
from .attention_viz import AttentionVisualizer, analyze_attention_at_depth
from .visualization import DiagnosticVisualizer, create_diagnostic_plots
from .profiling import (
    ModelProfiler,
    ConvergenceDiagnostics,
    PerplexityTracker,
    profile_model_comprehensive,
)

__all__ = [
    'EncodingAnalyzer',
    'compare_multiple_encodings',
    'AttentionVisualizer',
    'analyze_attention_at_depth',
    'DiagnosticVisualizer',
    'create_diagnostic_plots',
    'ModelProfiler',
    'ConvergenceDiagnostics',
    'PerplexityTracker',
    'profile_model_comprehensive',
]
