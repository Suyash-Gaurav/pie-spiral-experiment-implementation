"""
Test Suite for π-Spiral Positional Encoding Experiment

Contains:
- Unit tests for encoding modules
- Integration tests for model components
- Benchmark evaluation tests (NIAH, RULER, InfiniteBench)
- Performance tests
"""

from .test_encodings import (
    TestPiSpiralPositional,
    TestAttractorState,
    TestRoPEPositional,
    TestHybridPositionalEncoding,
    TestEncodingIntegration,
)

from .test_niah import NIAHEvaluator, run_niah_benchmark, compare_niah_results
from .test_ruler import RULEREvaluator, run_ruler_benchmark
from .test_infinitebench import InfiniteBenchEvaluator, run_infinitebench, run_infinitebench_subset

__all__ = [
    # Unit tests
    'TestPiSpiralPositional',
    'TestAttractorState',
    'TestRoPEPositional',
    'TestHybridPositionalEncoding',
    'TestEncodingIntegration',
    # Benchmark evaluators
    'NIAHEvaluator',
    'run_niah_benchmark',
    'compare_niah_results',
    'RULEREvaluator',
    'run_ruler_benchmark',
    'InfiniteBenchEvaluator',
    'run_infinitebench',
    'run_infinitebench_subset',
]
