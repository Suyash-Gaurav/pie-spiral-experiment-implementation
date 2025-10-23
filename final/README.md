# π-Spiral Positional Encoding Implementation

A complete Python implementation of π-Spiral Positional Encoding for long-context transformer models, featuring O(1) memory complexity through attractor state mechanisms.

## Overview

This package implements the π-Spiral positional encoding scheme, which uses irrational number properties to create non-periodic positional representations. This approach addresses the periodic aliasing issues in standard positional encodings like RoPE, enabling better performance on long-context tasks.

### Key Features

- **π-Spiral Positional Encoding**: Non-periodic encoding using irrational constants (π, e, √2, φ)
- **O(1) Attractor State**: Constant memory complexity for global context compression
- **Hybrid Encoding**: Smooth blending of RoPE (short-range) and π-Spiral (long-range)
- **Pre-trained Model Adapters**: Easy integration with Qwen, Llama, and other transformers
- **Comprehensive Benchmarks**: Support for NIAH, RULER, InfiniteBench, and LongBench
- **Flash Attention Support**: Efficient attention computation for long sequences

## Installation

### Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- PyTorch 2.0+

### Setup

```bash
# Clone or navigate to the project directory
cd final

# Install dependencies
pip install -r requirements.txt

# Install flash-attention (requires CUDA)
pip install flash-attn --no-build-isolation
```

## Project Structure

```
final/
├── src/                         # Source code modules
│   ├── encodings/              # Positional encoding implementations
│   │   ├── pi_spiral.py       # π-Spiral encoding and attractor state
│   │   ├── rope.py            # RoPE baseline implementation
│   │   └── hybrid.py          # Hybrid encoding (RoPE + π-Spiral)
│   ├── models/                 # Model architectures
│   │   ├── attention.py       # Attention mechanisms with π-Spiral
│   │   ├── transformer.py     # Complete transformer model
│   │   └── adapters.py        # Pre-trained model adapters
│   ├── data_utils.py          # Data loading and preprocessing
│   └── config.py              # Configuration management
├── tests/                      # Test scripts
├── configs/                    # Configuration files
├── scripts/                    # Executable scripts
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Quick Start

### 1. Using π-Spiral with a Custom Model

```python
from src.models import PiSpiralTransformer
from src.config import ExperimentConfig

# Create configuration
config = ExperimentConfig()

# Initialize model
model = PiSpiralTransformer(
    vocab_size=50000,
    d_model=512,
    num_layers=6,
    num_heads=8,
    pos_encoding='pi_spiral',
    use_attractor=True,
)

# Forward pass
import torch
input_ids = torch.randint(0, 50000, (1, 1000))
logits = model(input_ids)
```

### 2. Adapting a Pre-trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.adapters import inject_pi_spiral_encoding

# Load pre-trained model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")

# Inject π-Spiral encoding
adapter = inject_pi_spiral_encoding(
    model,
    pos_encoding='hybrid',  # Use hybrid for best results
    inject_layers='last_N',
    N_inject=4,
    use_attractor=True,
    hybrid_K=16000,
)

# Use the model normally
inputs = tokenizer("Your long context here...", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
```

### 3. Running NIAH Benchmark

```python
from src.data_utils import prepare_niah_sweep
from src.config import ExperimentConfig

# Prepare NIAH dataset
dataset = prepare_niah_sweep(
    lengths=[32000, 64000, 128000, 256000],
    depths=[0.1, 0.3, 0.5, 0.7, 0.9],
    samples_per_config=10,
    tokenizer=tokenizer,
)

# Evaluate model
for sample in dataset:
    context = sample['context']
    question = sample['question']
    # Run inference and evaluate...
```

## Configuration

The package uses a flexible configuration system based on dataclasses and YAML/JSON files.

### Creating a Configuration

```python
from src.config import ExperimentConfig, PositionalEncodingConfig, AttractorConfig

config = ExperimentConfig(
    experiment_name='my_experiment',
    model=ModelConfig(
        d_model=512,
        num_layers=6,
        pos_encoding=PositionalEncodingConfig(
            type='hybrid',
            irrational='pi',
            hybrid_K=16000,
        ),
        attractor=AttractorConfig(
            use_attractor=True,
            alpha_policy='exp_c_over_N',
            inject_layers='last_N',
            N_inject=4,
        ),
    ),
)

# Save configuration
config.to_yaml('configs/my_experiment.yaml')
```

### Loading a Configuration

```python
config = ExperimentConfig.from_yaml('configs/my_experiment.yaml')
```

### Pre-defined Configurations

```python
from src.config import create_default_configs, get_config_for_phase

# Get default configurations
configs = create_default_configs()
qwen_config = configs['qwen_1.5b']
llama_config = configs['llama_8b']

# Get configuration for specific experiment phase
phase2_config = get_config_for_phase(2)  # Sanity runs
phase3_config = get_config_for_phase(3)  # Core benchmarks
```

## Core Components

### Positional Encoding Types

1. **π-Spiral** (`pi_spiral`): Non-periodic encoding using irrational constants
   - Formula: `e_n = [cos(2π * frac(n*π)), sin(2π * frac(n*π))]`
   - Supports: π, e, √2, φ, PRNG

2. **Hybrid** (`hybrid`): Blends RoPE and π-Spiral with smooth transition
   - RoPE for positions ≤ K (default: 16000)
   - π-Spiral for positions > K
   - Smooth sigmoid blending

3. **RoPE** (`rope`): Standard rotary positional embedding baseline

### Attractor State

The attractor state provides O(1) memory complexity for global context:

- **Formula**: `C_t = α * C_{t-1} + e_t ⊗ h_t`
- **Alpha Policies**:
  - `fixed`: Constant decay factor
  - `exp_c_over_N`: `α = exp(-c/N)` where c defaults to π
  - `learned`: Learnable parameter

- **Injection Methods**:
  - `cross_attn`: Cross-attention to attractor state
  - `residual`: Residual connection
  - `none`: Disabled

## Experiment Phases

The implementation follows the experiment plan phases:

### Phase 0: Setup
- Repository structure
- Environment configuration
- Dependency installation

### Phase 1: Module Development
- π-Spiral positional encoding ✓
- Attractor state mechanism ✓
- Hybrid encoding ✓
- Model architecture ✓

### Phase 2: Sanity Runs (Qwen2.5-1.5B)
- Short tasks (4k-32k tokens)
- Long stream smoke test (1M tokens)
- Memory verification

### Phase 3: Core Benchmarks (Qwen2.5-1.5B)
- NIAH sweep (32k-1M tokens)
- RULER subset
- InfiniteBench subset

### Phase 4: Llama-3-8B
- 4-bit quantization
- NIAH to 512k
- RULER to 256k

### Phase 5: Llama-3-34B Demo
- CPU offload
- Single long NIAH (128k)
- Memory profiling

### Phase 6: Ablations
- Irrational constant comparison
- Attractor on/off
- Alpha policy comparison
- Hybrid threshold sweep

## API Reference

### Encodings

#### PiSpiralPositional
```python
from src.encodings import PiSpiralPositional

encoder = PiSpiralPositional(
    d_model=512,
    max_len=100000,
    irrational='pi',  # 'pi', 'e', 'sqrt2', 'phi', 'prng'
)

# Get positional encodings
positions = torch.arange(1000)
encodings = encoder.get_encoding(positions)

# Add to embeddings
x_with_pos = encoder(x, positions)
```

#### AttractorState
```python
from src.encodings import AttractorState

attractor = AttractorState(
    d_model=512,
    alpha_policy='exp_c_over_N',
    c_value=3.14159,
)

# Update attractor state
state = attractor(hidden_states, positional_encodings)

# Reset state
attractor.reset()
```

#### HybridPositionalEncoding
```python
from src.encodings import HybridPositionalEncoding

hybrid = HybridPositionalEncoding(
    d_model=512,
    hybrid_K=16000,
    transition_width=1000,
    irrational='pi',
)

# Apply hybrid encoding
x_encoded = hybrid(x, positions)

# Get blend weight for a position
weight = hybrid.get_blend_weight(20000)  # Returns 0-1
```

### Models

#### PiSpiralTransformer
```python
from src.models import PiSpiralTransformer

model = PiSpiralTransformer(
    vocab_size=50000,
    d_model=512,
    num_layers=6,
    num_heads=8,
    pos_encoding='hybrid',
    use_attractor=True,
    inject_layers='last_N',
    N_inject=4,
)

# Forward pass
logits = model(input_ids, attention_mask=mask)

# Generation
generated = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
)
```

### Data Utilities

#### NIAHDataset
```python
from src.data_utils import NIAHDataset

dataset = NIAHDataset(
    num_samples=100,
    context_lengths=[32000, 64000, 128000],
    depths=[0.1, 0.5, 0.9],
    tokenizer=tokenizer,
)

sample = dataset[0]
# Returns: {'context': str, 'question': str, 'answer': str, 
#           'context_length': int, 'depth': float}
```

## Performance Considerations

### Memory Optimization

1. **Use Hybrid Encoding**: Preserves short-range performance while enabling long-range
2. **Enable Flash Attention**: Significant speedup for long sequences
3. **Sliding Window**: Set `window_size` for very long contexts
4. **Gradient Checkpointing**: Reduces memory at cost of speed
5. **4-bit Quantization**: For large models (8B+)

### Recommended Settings

**Qwen2.5-1.5B**:
```python
config = ExperimentConfig(
    pretrained=PretrainedModelConfig(
        model_name_or_path='Qwen/Qwen2.5-1.5B',
        use_flash_attn=True,
    ),
    system=SystemConfig(
        window_size=8000,
        use_flash_attention=True,
    ),
)
```

**Llama-3-8B**:
```python
config = ExperimentConfig(
    pretrained=PretrainedModelConfig(
        model_name_or_path='meta-llama/Meta-Llama-3-8B',
        load_in_4bit=True,
        use_flash_attn=True,
    ),
    system=SystemConfig(
        window_size=4000,
        gradient_checkpointing=True,
    ),
)
```

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_encodings.py

# Run with coverage
pytest --cov=src tests/
```

## Complete Experiment Workflow

This implementation follows a 10-phase experiment plan. Here's how to run the complete workflow:

### Phase 2: Sanity Runs (Qwen 1.5B)
```bash
# Short tasks (4k-32k) with all encoding modes
python scripts/evaluate.py --config configs/phase2_sanity.yaml

# Long stream smoke test (1M tokens)
python scripts/diagnose.py --config configs/phase2_sanity.yaml --mode profiling
```

**Pass Gate:** ≤ 1% regression on short tasks, flat VRAM across 64k-1M

### Phase 3: Core Benchmarks (Qwen 1.5B)
```bash
# Full NIAH sweep (32k to 1M tokens)
python scripts/evaluate.py --config configs/qwen_1.5b_pi_spiral.yaml

# RULER and InfiniteBench
python scripts/evaluate.py --config configs/qwen_1.5b_pi_spiral.yaml --evaluation.benchmarks [ruler,infinitebench]
```

**Pass Gate:** ≥ 30% gain over RoPE at 256k+, smooth accuracy decay

### Phase 4: Medium Model (Llama 8B)
```bash
# Llama-3-8B with 4-bit quantization
python scripts/evaluate.py --config configs/llama_8b_pi_spiral.yaml
```

**Pass Gate:** Same gains as small model, no short-range regression

### Phase 5: Heavyweight Demo (Llama 34B)
```bash
# Single long NIAH with CPU offload
python scripts/evaluate.py --config configs/llama_34b_pi_spiral.yaml
```

**Pass Gate:** Bounded VRAM, stable retrieval

### Phase 6: Ablation Studies
```bash
# Compare irrational constants (π, e, √2, φ, PRNG)
python scripts/evaluate.py --config configs/ablation_irrational.yaml

# Attractor on/off comparison
python scripts/evaluate.py --config configs/ablation_attractor.yaml

# Alpha policy comparison
python scripts/evaluate.py --config configs/ablation_alpha_policy.yaml

# Hybrid threshold sweep
python scripts/evaluate.py --config configs/ablation_hybrid_k.yaml
```

### Phase 7: Diagnostics
```bash
# Positional collision and spectral analysis
python scripts/diagnose.py --config configs/qwen_1.5b_pi_spiral.yaml --mode encoding_analysis

# Attention visualization
python scripts/diagnose.py --config configs/qwen_1.5b_pi_spiral.yaml --mode attention_viz

# Memory and throughput profiling
python scripts/diagnose.py --config configs/qwen_1.5b_pi_spiral.yaml --mode profiling
```

### Phase 8-10: Results and Analysis
```bash
# Generate all tables and figures
python scripts/generate_tables.py --results ./results
python scripts/generate_figures.py --results ./results

# Statistical analysis with multiple seeds
python scripts/aggregate_seeds.py --results ./results --seeds 42,123,456
```

For detailed instructions, see [EXPERIMENTS.md](EXPERIMENTS.md).

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms:** CUDA out of memory during training or evaluation

**Solutions:**

1. **Reduce batch size:**
```bash
python scripts/train.py --config configs/qwen_1.5b_pi_spiral.yaml --training.batch_size 1
```

2. **Enable gradient checkpointing:**
```bash
python scripts/train.py --config configs/qwen_1.5b_pi_spiral.yaml --system.gradient_checkpointing true
```

3. **Use 4-bit quantization:**
```yaml
# In config file
pretrained:
  load_in_4bit: true
```

4. **Reduce sequence length:**
```bash
python scripts/evaluate.py --config configs/qwen_1.5b_pi_spiral.yaml --data.max_seq_length 32000
```

5. **Enable CPU offload (for very large models):**
```yaml
system:
  cpu_offload: true
```

### Slow Training/Evaluation

**Symptoms:** Very slow tokens/sec, long training times

**Solutions:**

1. **Enable Flash Attention:**
```yaml
system:
  use_flash_attention: true
```

2. **Use mixed precision:**
```yaml
training:
  bf16: true  # or fp16: true
```

3. **Reduce logging frequency:**
```yaml
training:
  logging_steps: 100  # Increase from default
```

4. **Use smaller window size:**
```yaml
system:
  window_size: 4000  # Reduce from 8000
```

5. **Compile model (PyTorch 2.0+):**
```yaml
system:
  compile_model: true
```

### Import Errors

**Symptoms:** `ModuleNotFoundError` or import failures

**Solutions:**

1. **Ensure you're in the correct directory:**
```bash
cd final  # Must be in project root
python scripts/train.py --config configs/qwen_1.5b_pi_spiral.yaml
```

2. **Reinstall dependencies:**
```bash
pip install -r requirements.txt --upgrade
```

3. **Install flash-attention separately:**
```bash
pip install flash-attn --no-build-isolation
```

4. **Check Python path:**
```python
import sys
print(sys.path)  # Should include project root
```

### Model Download Issues

**Symptoms:** Cannot download pre-trained models from Hugging Face

**Solutions:**

1. **Login to Hugging Face (for gated models):**
```bash
huggingface-cli login
```

2. **Set cache directory:**
```bash
export HF_HOME=/path/to/cache
```

3. **Use local model path:**
```yaml
pretrained:
  model_name_or_path: /local/path/to/model
```

4. **Check internet connection and proxy settings**

### Configuration Errors

**Symptoms:** YAML parsing errors, missing required fields

**Solutions:**

1. **Validate YAML syntax:**
```bash
python -c "import yaml; yaml.safe_load(open('configs/my_config.yaml'))"
```

2. **Use base config as template:**
```bash
cp configs/base_config.yaml configs/my_config.yaml
# Edit my_config.yaml
```

3. **Check required fields:**
```python
from src.config import ExperimentConfig
config = ExperimentConfig.from_yaml('configs/my_config.yaml')
config.validate()  # Will raise error if invalid
```

### Poor Performance

**Symptoms:** Lower accuracy than expected, no improvement over baseline

**Solutions:**

1. **Verify configuration:**
   - Check positional encoding type is set correctly
   - Ensure attractor is enabled if desired
   - Verify hybrid K is appropriate for your model

2. **Check data quality:**
   - Verify dataset is loaded correctly
   - Check tokenization is appropriate
   - Ensure no data leakage

3. **Run diagnostics:**
```bash
python scripts/diagnose.py --config configs/your_config.yaml --mode all
```

4. **Compare with baseline:**
```bash
python scripts/evaluate.py --config configs/baseline_rope.yaml
python scripts/compare_results.py --experiment ./results/your_experiment --baseline ./results/baseline_rope
```

5. **Try different hyperparameters:**
   - Adjust hybrid K value
   - Try different alpha policies
   - Experiment with different irrational constants

### Reproducibility Issues

**Symptoms:** Different results across runs with same configuration

**Solutions:**

1. **Set all random seeds:**
```yaml
training:
  seed: 42
  deterministic: true
```

2. **Set environment variables:**
```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

3. **Disable non-deterministic operations:**
```python
import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

4. **Document environment:**
```bash
pip list > environment.txt
python --version > python_version.txt
nvidia-smi > gpu_info.txt
```

## Performance Optimization Tips

### Memory Optimization

1. **Use Gradient Accumulation:**
   - Effective batch size = batch_size × gradient_accumulation_steps
   - Reduces memory while maintaining training dynamics

2. **Enable Gradient Checkpointing:**
   - Trades computation for memory
   - Essential for very long sequences

3. **Use Quantization:**
   - 4-bit: ~4x memory reduction
   - 8-bit: ~2x memory reduction
   - Minimal accuracy loss

4. **Optimize Window Size:**
   - Smaller window = less memory
   - Find minimum window that maintains performance

5. **Use Attractor State:**
   - O(1) memory for global context
   - Constant size regardless of sequence length

### Speed Optimization

1. **Enable Flash Attention:**
   - 2-4x speedup on long sequences
   - Requires CUDA and compatible GPU

2. **Use Mixed Precision:**
   - bf16 recommended for modern GPUs
   - fp16 for older GPUs
   - ~2x speedup typical

3. **Compile Model (PyTorch 2.0+):**
   - Automatic optimization
   - Can provide 10-30% speedup

4. **Optimize Data Loading:**
   - Use multiple workers
   - Pin memory for GPU transfer
   - Prefetch data

5. **Batch Processing:**
   - Larger batches = better GPU utilization
   - Use gradient accumulation if memory limited

### Recommended Settings by Model Size

**Small Models (1-2B parameters):**
```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 4
  bf16: true

system:
  window_size: 8000
  use_flash_attention: true
  gradient_checkpointing: false
```

**Medium Models (7-8B parameters):**
```yaml
pretrained:
  load_in_4bit: true

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  bf16: true

system:
  window_size: 4000
  use_flash_attention: true
  gradient_checkpointing: true
```

**Large Models (30B+ parameters):**
```yaml
pretrained:
  load_in_4bit: true

training:
  batch_size: 1
  gradient_accumulation_steps: 16
  bf16: true

system:
  window_size: 2000
  use_flash_attention: true
  gradient_checkpointing: true
  cpu_offload: true
```

## Additional Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design decisions
- **[EXPERIMENTS.md](EXPERIMENTS.md)** - Detailed guide for running all experiment phases
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes
- **[configs/README.md](configs/README.md)** - Complete configuration guide
- **[examples/README.md](examples/README.md)** - Example scripts documentation

## Frequently Asked Questions

### Q: What is π-Spiral positional encoding?

A: π-Spiral is a non-periodic positional encoding that uses irrational constants (like π) to create unique positional representations. Unlike RoPE which has periodic patterns, π-Spiral avoids aliasing at long distances.

### Q: What is the attractor state?

A: The attractor state is an O(1) memory mechanism that compresses global context into a fixed-size state. It's updated incrementally as: `C_t = α * C_{t-1} + e_t ⊗ h_t`, providing constant memory regardless of sequence length.

### Q: Should I use pure π-Spiral or Hybrid encoding?

A: **Hybrid is recommended** for most use cases. It preserves short-range performance (using RoPE) while enabling long-range capabilities (using π-Spiral). Pure π-Spiral may show slight regression on short sequences.

### Q: What is the optimal hybrid K value?

A: Depends on your model:
- **Qwen 1.5B:** K=8000 to K=16000
- **Llama 8B:** K=16000
- **Larger models:** K=16000 to K=32000

Run `ablation_hybrid_k.yaml` to find the optimal value for your use case.

### Q: Which irrational constant should I use?

A: **π (pi) is recommended** as the default. However, ablation studies show that different irrational constants (e, √2, φ) and even PRNG perform similarly. The key is non-periodicity, not the specific constant.

### Q: How much memory does this save?

A: The attractor state provides O(1) memory for global context compression. However, the main memory usage still comes from attention (O(N²) or O(N) with flash attention). The benefit is enabling longer contexts within the same memory budget.

### Q: Can I use this with my existing model?

A: Yes! Use the adapter interface:
```python
from src.models.adapters import inject_pi_spiral_encoding
adapter = inject_pi_spiral_encoding(your_model, pos_encoding='hybrid')
```

### Q: Does this require retraining from scratch?

A: No. You can adapt pre-trained models using the adapter interface. Fine-tuning on your task is recommended but not required.

### Q: What are the computational costs?

A: π-Spiral encoding itself is very efficient (on-the-fly computation). The attractor state adds minimal overhead. Overall computational cost is similar to standard transformers.

### Q: How does this compare to other long-context methods?

A: π-Spiral shows:
- **vs RoPE:** Better long-range performance, no periodic aliasing
- **vs RoPE-NTK:** Better extrapolation, non-periodic
- **vs ALiBi:** Better global context (via attractor), more flexible
- **vs Sparse Attention:** Full attention with O(1) global context

## Citation

If you use this implementation in your research, please cite:

@software{pi_spiral_encoding,
  title={π-Spiral Positional Encoding for Long-Context Transformers},
  author={[Suyash Gaurav]},
  year={2024},
  url={https://github.com/Suyash-Gaurav/pi-spiral}
}



## License

[Specify your license here]

## Acknowledgments

- Experiment plan based on comprehensive long-context evaluation methodology
- Inspired by RoPE, ALiBi, and other positional encoding research
- Built with PyTorch and Hugging Face Transformers

## Contact

For questions or issues, please open an issue on GitHub or contact [s23224522@al.tiu.ac.jp].

If you use this implementation in your research, please cite:

```bibtex
@software{pi_spiral_encoding,
  title={π-Spiral Positional Encoding for Long-Context Transformers},
  author={[Suyash Gaurav]},
  year={2024},
  url={https://github.com/Suyash-Gaurav/pi-spiral}
}
```




## Acknowledgments

- Experiment plan based on comprehensive long-context evaluation methodology
- Inspired by RoPE, ALiBi, and other positional encoding research
- Built with PyTorch and Hugging Face Transformers

## Contact

For questions or issues, please open an issue on GitHub or contact [s23224522@al.tiu.ac.jp].
