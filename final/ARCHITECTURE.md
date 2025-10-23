# Architecture Documentation

This document describes the architecture and design decisions of the π-Spiral Positional Encoding implementation.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Design Decisions](#design-decisions)
- [Data Flow](#data-flow)
- [Extension Points](#extension-points)
- [Performance Considerations](#performance-considerations)

---

## Overview

The π-Spiral Positional Encoding system is designed as a modular, extensible framework for experimenting with non-periodic positional encodings in transformer models. The architecture follows a layered approach with clear separation of concerns.

### Design Principles

1. **Modularity:** Components are independent and can be used separately
2. **Extensibility:** Easy to add new encodings, models, and benchmarks
3. **Compatibility:** Works with existing Hugging Face models
4. **Efficiency:** Optimized for long-context processing
5. **Reproducibility:** Deterministic behavior with seed control

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Training   │  │  Evaluation  │  │ Diagnostics  │      │
│  │   Scripts    │  │   Scripts    │  │   Scripts    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Configuration Layer                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         ExperimentConfig (YAML/JSON)                  │  │
│  │  - Model Config  - Training Config  - Data Config    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                      Core Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Encodings   │  │    Models    │  │   Training   │      │
│  │  - π-Spiral  │  │  - Transform │  │   - Trainer  │      │
│  │  - RoPE      │  │  - Attention │  │   - Logger   │      │
│  │  - Hybrid    │  │  - Adapters  │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Support Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Data Utils  │  │ Diagnostics  │  │    Tests     │      │
│  │  - NIAH      │  │  - Analysis  │  │  - Unit      │      │
│  │  - RULER     │  │  - Profiling │  │  - Benchmark │      │
│  │  - Infinite  │  │  - Viz       │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
src/
├── encodings/              # Positional encoding implementations
│   ├── pi_spiral.py       # π-Spiral encoding + attractor
│   ├── rope.py            # RoPE baseline
│   ├── hybrid.py          # Hybrid encoding
│   └── __init__.py        # Public API
│
├── models/                 # Model architectures
│   ├── transformer.py     # Complete transformer
│   ├── attention.py       # Attention mechanisms
│   ├── adapters.py        # Pre-trained model adapters
│   └── __init__.py        # Public API
│
├── training/               # Training infrastructure
│   ├── trainer.py         # Training loop
│   ├── logger.py          # Logging and tracking
│   └── __init__.py        # Public API
│
├── diagnostics/            # Analysis and visualization
│   ├── encoding_analysis.py   # Positional analysis
│   ├── attention_viz.py       # Attention visualization
│   ├── profiling.py           # Performance profiling
│   ├── visualization.py       # General visualization
│   └── __init__.py            # Public API
│
├── config.py              # Configuration management
├── data_utils.py          # Data loading and preprocessing
└── __init__.py            # Package initialization
```

---

## Core Components

### 1. Positional Encodings

#### PiSpiralPositional

**Purpose:** Non-periodic positional encoding using irrational constants.

**Key Features:**
- Generates 2D unit vectors: `e_n = [cos(2π * frac(n*c)), sin(2π * frac(n*c))]`
- Supports multiple irrational constants (π, e, √2, φ)
- On-the-fly computation (no pre-computed tables)
- Deterministic with seed control

**Design Decisions:**
- **Why 2D vectors?** Provides sufficient expressiveness while maintaining efficiency
- **Why fractional part?** Creates non-periodic patterns from irrational numbers
- **Why on-the-fly?** Avoids memory overhead of pre-computed tables

**Implementation:**
```python
class PiSpiralPositional(nn.Module):
    def __init__(self, d_model, max_len, irrational='pi'):
        # Initialize irrational constant
        self.constant = self._get_constant(irrational)
    
    def get_encoding(self, positions):
        # Compute: e_n = [cos(2π * frac(n*c)), sin(2π * frac(n*c))]
        phases = (positions * self.constant) % 1.0
        angles = 2 * math.pi * phases
        return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
```

#### AttractorState

**Purpose:** O(1) memory complexity for global context compression.

**Key Features:**
- Incremental update: `C_t = α * C_{t-1} + e_t ⊗ h_t`
- Multiple alpha policies (fixed, exp_c_over_N, learned)
- Configurable injection methods (cross-attention, residual)
- Layer-wise injection control

**Design Decisions:**
- **Why exponential decay?** Balances recent vs distant context
- **Why outer product?** Captures position-content interactions
- **Why O(1)?** Constant memory regardless of sequence length

**Implementation:**
```python
class AttractorState(nn.Module):
    def __init__(self, d_model, alpha_policy='exp_c_over_N'):
        self.state = None  # Shape: [batch, d_model, d_model]
        self.alpha_policy = alpha_policy
    
    def update(self, hidden_states, positional_encodings):
        # C_t = α * C_{t-1} + e_t ⊗ h_t
        alpha = self.get_alpha(N=hidden_states.size(1))
        outer_product = torch.einsum('bld,blp->bdp', 
                                     hidden_states, 
                                     positional_encodings)
        self.state = alpha * self.state + outer_product
        return self.state
```

#### HybridPositionalEncoding

**Purpose:** Smooth blending of RoPE (short-range) and π-Spiral (long-range).

**Key Features:**
- RoPE for positions ≤ K
- π-Spiral for positions > K
- Smooth sigmoid transition
- Configurable transition width

**Design Decisions:**
- **Why hybrid?** Preserves short-range performance while enabling long-range
- **Why sigmoid?** Provides smooth, differentiable transition
- **Why configurable K?** Different models need different transition points

**Implementation:**
```python
class HybridPositionalEncoding(nn.Module):
    def __init__(self, d_model, hybrid_K=16000, transition_width=1000):
        self.rope = RoPEPositional(d_model)
        self.pi_spiral = PiSpiralPositional(d_model)
        self.K = hybrid_K
        self.width = transition_width
    
    def get_blend_weight(self, position):
        # Sigmoid transition centered at K
        return torch.sigmoid((position - self.K) / self.width)
    
    def forward(self, x, positions):
        rope_enc = self.rope(x, positions)
        spiral_enc = self.pi_spiral(x, positions)
        weight = self.get_blend_weight(positions)
        return (1 - weight) * rope_enc + weight * spiral_enc
```

---

### 2. Model Architecture

#### PiSpiralTransformer

**Purpose:** Complete transformer model with π-Spiral encoding.

**Architecture:**
```
Input Embeddings
      ↓
[Positional Encoding: RoPE/π-Spiral/Hybrid]
      ↓
┌─────────────────┐
│  Transformer    │
│  Layer 1        │  ← Standard layers
│  ...            │
│  Layer N-4      │
├─────────────────┤
│  Layer N-3      │  ← Attractor injection
│  Layer N-2      │     (last N layers)
│  Layer N-1      │
│  Layer N        │
└─────────────────┘
      ↓
[Attractor Cross-Attention] (optional)
      ↓
Output Logits
```

**Design Decisions:**
- **Why inject in last N layers?** Final layers handle high-level reasoning
- **Why cross-attention?** Allows model to query global context
- **Why optional attractor?** Enables ablation studies

#### PiSpiralAdapter

**Purpose:** Adapt pre-trained models with π-Spiral encoding.

**Strategy:**
1. Load pre-trained model (frozen or fine-tunable)
2. Replace positional encoding module
3. Inject attractor state in specified layers
4. Preserve original weights

**Design Decisions:**
- **Why adapter pattern?** Minimal changes to pre-trained models
- **Why layer injection?** Gradual integration of new mechanism
- **Why preserve weights?** Leverage pre-training

---

### 3. Training Infrastructure

#### Trainer

**Purpose:** Complete training loop with all necessary features.

**Features:**
- Gradient accumulation
- Mixed precision (fp16, bf16)
- Gradient checkpointing
- Learning rate scheduling
- Checkpointing and resumption
- Multi-GPU support

**Design Decisions:**
- **Why gradient accumulation?** Enables large effective batch sizes
- **Why mixed precision?** Reduces memory and increases speed
- **Why checkpointing?** Enables training on long sequences

#### Logger

**Purpose:** Comprehensive experiment tracking.

**Features:**
- WandB integration
- TensorBoard integration
- Memory tracking
- Throughput measurement
- Metric aggregation

**Design Decisions:**
- **Why multiple backends?** Flexibility in experiment tracking
- **Why memory tracking?** Critical for long-context experiments
- **Why throughput?** Performance optimization

---

### 4. Diagnostic Tools

#### Encoding Analysis

**Purpose:** Analyze positional encoding properties.

**Analyses:**
- Positional collision (cosine similarity)
- Spectral analysis (FFT)
- Non-periodicity verification
- Distance preservation

**Design Decisions:**
- **Why cosine similarity?** Measures positional discrimination
- **Why FFT?** Reveals periodic patterns
- **Why multiple metrics?** Comprehensive understanding

#### Profiling

**Purpose:** Performance and resource analysis.

**Metrics:**
- VRAM usage vs sequence length
- Tokens per second vs sequence length
- State size verification
- Cost curves

**Design Decisions:**
- **Why vs length?** Verify O(1) memory claim
- **Why tokens/sec?** Practical performance metric
- **Why cost curves?** Operational viability

---

## Design Decisions

### 1. Configuration System

**Decision:** Use dataclasses + YAML/JSON

**Rationale:**
- Type safety with dataclasses
- Human-readable YAML files
- Easy serialization/deserialization
- Validation support

**Alternatives Considered:**
- Pure Python dicts: Less type safety
- Hydra: Too heavyweight
- argparse: Not hierarchical enough

### 2. Positional Encoding Interface

**Decision:** Separate encoding from model

**Rationale:**
- Modularity: Easy to swap encodings
- Testability: Can test encodings independently
- Reusability: Use in different models

**Interface:**
```python
class PositionalEncoding(nn.Module):
    def get_encoding(self, positions: Tensor) -> Tensor:
        """Get positional encodings for given positions."""
        pass
    
    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """Add positional encoding to input."""
        return x + self.get_encoding(positions)
```

### 3. Attractor State Management

**Decision:** Stateful module with explicit reset

**Rationale:**
- Efficiency: Avoid recomputation
- Clarity: Explicit state management
- Flexibility: Can reset between sequences

**Alternatives Considered:**
- Stateless: Would require passing state explicitly
- Automatic reset: Less control over state lifecycle

### 4. Adapter Pattern for Pre-trained Models

**Decision:** Non-invasive injection via module replacement

**Rationale:**
- Compatibility: Works with any transformer
- Preservation: Keeps pre-trained weights
- Flexibility: Can inject at different layers

**Implementation Strategy:**
```python
def inject_pi_spiral_encoding(model, **config):
    # 1. Replace positional encoding
    replace_module(model, 'pos_encoding', new_encoding)
    
    # 2. Inject attractor in specified layers
    for layer_idx in injection_layers:
        add_attractor_cross_attention(model.layers[layer_idx])
    
    # 3. Return adapter handle
    return PiSpiralAdapter(model, config)
```

### 5. Benchmark Integration

**Decision:** Unified interface for all benchmarks

**Rationale:**
- Consistency: Same evaluation protocol
- Extensibility: Easy to add new benchmarks
- Comparability: Fair comparisons

**Interface:**
```python
class Benchmark:
    def create_dataset(self, **config) -> Dataset:
        """Create benchmark dataset."""
        pass
    
    def evaluate(self, model, dataset) -> Dict[str, float]:
        """Evaluate model on benchmark."""
        pass
    
    def visualize(self, results) -> Figure:
        """Visualize results."""
        pass
```

---

## Data Flow

### Training Flow

```
Configuration (YAML)
        ↓
ExperimentConfig.from_yaml()
        ↓
Initialize Model + Encoding
        ↓
Load/Create Dataset
        ↓
┌─────────────────────┐
│   Training Loop     │
│  ┌───────────────┐  │
│  │ Forward Pass  │  │
│  │   ↓           │  │
│  │ Compute Loss  │  │
│  │   ↓           │  │
│  │ Backward Pass │  │
│  │   ↓           │  │
│  │ Update Params │  │
│  └───────────────┘  │
│         ↓           │
│  Log Metrics        │
│         ↓           │
│  Save Checkpoint    │
└─────────────────────┘
        ↓
Final Model + Results
```

### Evaluation Flow

```
Configuration + Model
        ↓
Load Benchmark Dataset
        ↓
┌─────────────────────┐
│  Evaluation Loop    │
│  ┌───────────────┐  │
│  │ Get Sample    │  │
│  │   ↓           │  │
│  │ Model Forward │  │
│  │   ↓           │  │
│  │ Compute Metric│  │
│  │   ↓           │  │
│  │ Aggregate     │  │
│  └───────────────┘  │
└─────────────────────┘
        ↓
Results + Visualizations
```

### Diagnostic Flow

```
Model + Configuration
        ↓
┌─────────────────────────┐
│  Diagnostic Analyses    │
│  ┌───────────────────┐  │
│  │ Encoding Analysis │  │
│  │ Attention Viz     │  │
│  │ Memory Profiling  │  │
│  │ Spectral Analysis │  │
│  └───────────────────┘  │
└─────────────────────────┘
        ↓
Reports + Plots
```

---

## Extension Points

### Adding New Positional Encodings

1. **Create encoding class:**
```python
class MyEncoding(nn.Module):
    def get_encoding(self, positions):
        # Your implementation
        pass
```

2. **Register in factory:**
```python
ENCODING_REGISTRY = {
    'rope': RoPEPositional,
    'pi_spiral': PiSpiralPositional,
    'my_encoding': MyEncoding,  # Add here
}
```

3. **Add configuration:**
```yaml
model:
  pos_encoding:
    type: my_encoding
    # Your parameters
```

### Adding New Benchmarks

1. **Create benchmark class:**
```python
class MyBenchmark:
    def create_dataset(self, **config):
        # Dataset creation
        pass
    
    def evaluate(self, model, dataset):
        # Evaluation logic
        pass
```

2. **Register in data_utils:**
```python
BENCHMARK_REGISTRY = {
    'niah': NIAHBenchmark,
    'ruler': RULERBenchmark,
    'my_benchmark': MyBenchmark,  # Add here
}
```

3. **Add to configuration:**
```yaml
evaluation:
  benchmarks:
    - my_benchmark
```

### Adding New Diagnostic Tools

1. **Create diagnostic function:**
```python
def my_diagnostic(model, config):
    # Analysis logic
    return results
```

2. **Register in diagnostics:**
```python
DIAGNOSTIC_REGISTRY = {
    'encoding': analyze_encoding,
    'attention': visualize_attention,
    'my_diagnostic': my_diagnostic,  # Add here
}
```

3. **Use in scripts:**
```bash
python scripts/diagnose.py --mode my_diagnostic
```

---

## Performance Considerations

### Memory Optimization

1. **Gradient Checkpointing:**
   - Trades computation for memory
   - Essential for long sequences
   - Configurable per layer

2. **Flash Attention:**
   - O(N) memory instead of O(N²)
   - Significant speedup
   - Requires CUDA

3. **Quantization:**
   - 4-bit/8-bit weights
   - Reduces memory by 4-8x
   - Minimal accuracy loss

4. **Attractor State:**
   - O(1) memory for global context
   - Constant size regardless of length
   - Efficient outer product computation

### Computational Optimization

1. **On-the-fly Encoding:**
   - No pre-computed tables
   - Minimal memory overhead
   - Fast computation

2. **Vectorized Operations:**
   - Batch processing
   - GPU-optimized kernels
   - Minimal Python overhead

3. **Mixed Precision:**
   - bf16 for most operations
   - fp32 for critical computations
   - 2x speedup typical

### Scalability

1. **Multi-GPU:**
   - Data parallelism
   - Model parallelism (for large models)
   - Gradient accumulation across GPUs

2. **Sequence Length:**
   - Sliding window for very long sequences
   - Efficient attention mechanisms
   - Memory-efficient implementations

3. **Batch Size:**
   - Gradient accumulation
   - Dynamic batching
   - Memory-aware scheduling

---

## Testing Strategy

### Unit Tests
- Individual component testing
- Encoding correctness
- Shape verification
- Determinism checks

### Integration Tests
- End-to-end workflows
- Configuration loading
- Model initialization
- Training loops

### Benchmark Tests
- NIAH evaluation
- RULER tasks
- InfiniteBench
- Baseline comparisons

### Performance Tests
- Memory profiling
- Throughput measurement
- Scaling verification
- Regression detection

---

## Future Architecture Considerations

### Planned Enhancements

1. **Distributed Training:**
   - DeepSpeed integration
   - FSDP support
   - Pipeline parallelism

2. **Efficient Inference:**
   - vLLM integration
   - KV cache optimization
   - Speculative decoding

3. **Model Zoo:**
   - Pre-trained checkpoints
   - Fine-tuned variants
   - Domain-specific models

4. **AutoML:**
   - Hyperparameter tuning
   - Architecture search
   - Configuration optimization

---

## References

- **RoPE:** Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- **ALiBi:** Press et al., "Train Short, Test Long: Attention with Linear Biases"
- **Flash Attention:** Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention"
- **Transformers:** Vaswani et al., "Attention Is All You Need"

---

**Document Version:** 1.0  
**Last Updated:** 2024-10-23  
**Maintainer:** π-Spiral Team
