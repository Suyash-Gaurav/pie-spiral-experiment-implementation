# Configuration Files Guide

This directory contains YAML configuration files for all experiment phases and scenarios in the π-Spiral Positional Encoding project. Each configuration file is designed for specific experiment phases as outlined in the experiment plan.

## Table of Contents

- [Base Configuration](#base-configuration)
- [Model-Specific Configurations](#model-specific-configurations)
- [Baseline Configurations](#baseline-configurations)
- [Ablation Study Configurations](#ablation-study-configurations)
- [Phase-Specific Configurations](#phase-specific-configurations)
- [Configuration Structure](#configuration-structure)
- [Usage Examples](#usage-examples)

---

## Base Configuration

### `base_config.yaml`
**Purpose:** Template configuration with all available parameters and default values.

**Key Features:**
- Complete parameter documentation
- Default values for all settings
- Suitable for custom model training from scratch
- Includes all positional encoding options

**Use Case:** Starting point for creating custom configurations or understanding available parameters.

**Command:**
```bash
python scripts/train.py --config configs/base_config.yaml
```

---

## Model-Specific Configurations

These configurations are optimized for specific pre-trained models with appropriate memory and compute settings.

### `qwen_1.5b_pi_spiral.yaml`
**Purpose:** Phase 2 & 3 - Sanity runs and core benchmarks on Qwen2.5-1.5B

**Model:** Qwen/Qwen2.5-1.5B  
**Quantization:** None (bf16)  
**Memory:** ~6-8 GB VRAM  
**Experiment Phases:** 2, 3

**Key Settings:**
- Hybrid positional encoding (K=16000)
- Attractor state enabled
- Window size: 8000
- NIAH lengths: 32k to 1M tokens
- Full benchmark suite: NIAH, RULER, InfiniteBench

**Pass Gates:**
- ≤ 1% short-range regression (4k-32k)
- ≥ 30% gain over RoPE at 256k+ on deep NIAH
- Flat VRAM across 64k to 1M

**Command:**
```bash
python scripts/train.py --config configs/qwen_1.5b_pi_spiral.yaml
python scripts/evaluate.py --config configs/qwen_1.5b_pi_spiral.yaml
```

---

### `llama_8b_pi_spiral.yaml`
**Purpose:** Phase 4 - Medium model evaluation with 4-bit quantization

**Model:** meta-llama/Meta-Llama-3-8B  
**Quantization:** 4-bit  
**Memory:** ~12-14 GB VRAM  
**Experiment Phase:** 4

**Key Settings:**
- 4-bit quantization for memory efficiency
- Hybrid encoding (K=16000)
- Window size: 4000 (smaller for stability)
- NIAH up to 512k tokens
- RULER up to 256k tokens
- Gradient checkpointing enabled

**Pass Gates:**
- Same qualitative gains as small model
- No short-range regressions with Hybrid
- Stable performance with quantization

**Command:**
```bash
python scripts/train.py --config configs/llama_8b_pi_spiral.yaml
python scripts/evaluate.py --config configs/llama_8b_pi_spiral.yaml
```

---

### `llama_34b_pi_spiral.yaml`
**Purpose:** Phase 5 - Heavyweight demo showing O(1) memory scaling

**Model:** meta-llama/Meta-Llama-3-34B  
**Quantization:** 4-bit with CPU offload  
**Memory:** ~16 GB VRAM + CPU RAM  
**Experiment Phase:** 5

**Key Settings:**
- 4-bit quantization + CPU offload
- Window size: 2000 (minimal for memory)
- Batch size: 1
- Single long NIAH at 128k
- One InfiniteBench task at 100k
- Focus on memory profiling

**Pass Gates:**
- Bounded VRAM (no growth with length)
- Stable retrieval at depth
- Throughput can be slow (acceptable)

**Command:**
```bash
python scripts/evaluate.py --config configs/llama_34b_pi_spiral.yaml
```

---

## Baseline Configurations

Comparison baselines for evaluating π-Spiral performance gains.

### `baseline_rope.yaml`
**Purpose:** Standard RoPE baseline for comparison

**Model:** Qwen/Qwen2.5-1.5B  
**Encoding:** RoPE (Rotary Position Embedding)  
**Use Case:** Primary baseline for all comparisons

**Key Settings:**
- Standard RoPE with base=10000
- No attractor state
- Same evaluation protocol as π-Spiral
- NIAH up to 256k tokens

**Expected Behavior:**
- Good short-range performance
- Degraded long-range performance due to periodic aliasing
- Memory scales with sequence length

**Command:**
```bash
python scripts/evaluate.py --config configs/baseline_rope.yaml
```

---

### `baseline_rope_ntk.yaml`
**Purpose:** RoPE with NTK-aware scaling baseline

**Model:** Qwen/Qwen2.5-1.5B  
**Encoding:** RoPE with NTK-aware interpolation  
**Use Case:** Strong baseline with context extension

**Key Settings:**
- NTK-aware scaling for context extension
- Original max position: 32768
- Extended max position: 1M
- No attractor state

**Expected Behavior:**
- Better long-range than standard RoPE
- Still suffers from periodic aliasing
- Improved extrapolation capabilities

**Command:**
```bash
python scripts/evaluate.py --config configs/baseline_rope_ntk.yaml
```

---

### `baseline_alibi.yaml`
**Purpose:** ALiBi (Attention with Linear Biases) baseline

**Model:** Qwen/Qwen2.5-1.5B  
**Encoding:** ALiBi  
**Use Case:** Alternative positional encoding baseline

**Key Settings:**
- Linear biases added to attention scores
- No positional embeddings
- Auto-calculated slopes per head
- Good extrapolation properties

**Expected Behavior:**
- Good short and medium range performance
- Better extrapolation than standard RoPE
- No global context compression

**Command:**
```bash
python scripts/evaluate.py --config configs/baseline_alibi.yaml
```

---

## Ablation Study Configurations

Phase 6 ablations to isolate individual component contributions.

### `ablation_irrational.yaml`
**Purpose:** Compare different irrational constants (π, e, √2, φ, PRNG)

**Ablation Mode:** `irrational`  
**Values:** `[pi, e, sqrt2, phi, prng]`  
**Experiment Phase:** 6

**Key Settings:**
- Tests 5 different irrational constants
- NIAH at 512k tokens
- One RULER task at 256k
- Same model and settings for fair comparison

**Research Question:**
- Is π special, or does any non-periodic sequence work?
- If PRNG ≈ π, claim becomes "non-periodic encoding"

**Pass Gate:**
- Clear ranking of irrational constants
- Statistical significance in differences

**Command:**
```bash
python scripts/evaluate.py --config configs/ablation_irrational.yaml
```

---

### `ablation_attractor.yaml`
**Purpose:** Isolate attractor state contribution (on vs off)

**Ablation Mode:** `attractor`  
**Values:** `[true, false]`  
**Experiment Phase:** 6

**Key Settings:**
- π-Spiral positions + attractor vs positions alone
- NIAH at 512k tokens
- One RULER task at 256k

**Research Question:**
- How much gain comes from non-periodic positions vs O(1) attractor?
- Is the attractor necessary for long-context performance?

**Pass Gate:**
- Clear separation of contributions
- Quantify attractor's impact

**Command:**
```bash
python scripts/evaluate.py --config configs/ablation_attractor.yaml
```

---

### `ablation_alpha_policy.yaml`
**Purpose:** Compare alpha decay schedules for attractor state

**Ablation Mode:** `alpha_policy`  
**Values:** `[exp_c_over_N, fixed, learned]`  
**Experiment Phase:** 6

**Key Settings:**
- Three alpha policies:
  - `exp_c_over_N`: α = exp(-π/N) (theoretically motivated)
  - `fixed`: α = 0.99 (constant)
  - `learned`: α as learnable parameter
- Also tests different c values for exp_c_over_N

**Research Question:**
- What's the optimal decay schedule?
- Does theoretical motivation (π/N) matter?

**Pass Gate:**
- Clear ranking of policies
- Identify optimal schedule

**Command:**
```bash
python scripts/evaluate.py --config configs/ablation_alpha_policy.yaml
```

---

### `ablation_hybrid_k.yaml`
**Purpose:** Sweep hybrid threshold K (transition point from RoPE to π-Spiral)

**Ablation Mode:** `hybrid_k`  
**Values:** `[8000, 16000, 32000, 64000]`  
**Experiment Phase:** 6

**Key Settings:**
- Tests 4 different K values
- Includes short-range guard (4k-32k)
- Tests long-range performance (128k-256k)
- Verifies smooth transition

**Research Question:**
- What's the optimal transition point?
- Can we preserve short-range while gaining long-range?

**Pass Gate:**
- At least one K with ≤ 1% short-range drop
- Significant long-range gains

**Recommended K:**
- K=8k for Qwen 1.5B
- K=16k for Llama 8B

**Command:**
```bash
python scripts/evaluate.py --config configs/ablation_hybrid_k.yaml
```

---

## Phase-Specific Configurations

### `phase2_sanity.yaml`
**Purpose:** Phase 2 sanity runs - prove nothing breaks

**Experiment Phase:** 2  
**Model:** Qwen/Qwen2.5-1.5B

**Key Settings:**
- Short tasks: 4k to 32k tokens
- Long stream smoke test: 1M tokens
- Tests all three modes: RoPE, π-Spiral, Hybrid
- Extensive memory and throughput logging

**Pass Gates:**
- ≤ 1% delta on 4k-32k with Hybrid
- Flat VRAM across 64k to 1M
- Stable tokens/sec

**Command:**
```bash
python scripts/train.py --config configs/phase2_sanity.yaml
python scripts/diagnose.py --config configs/phase2_sanity.yaml --mode profiling
```

---

## Configuration Structure

All configuration files follow this structure:

```yaml
# Experiment Identification
experiment_name: string
run_name: string (optional)
output_dir: path
logging_dir: path

# Pretrained Model (if using pre-trained)
pretrained:
  model_name_or_path: string
  trust_remote_code: bool
  torch_dtype: string
  load_in_4bit: bool
  load_in_8bit: bool
  use_flash_attn: bool
  device_map: string

# Model Configuration
model:
  # Positional Encoding
  pos_encoding:
    type: string  # rope, rope_ntk, alibi, pi_spiral, hybrid
    irrational: string  # pi, e, sqrt2, phi, prng
    hybrid_K: int
    transition_width: int
    max_seq_len: int
    rope_base: float
  
  # Attractor State
  attractor:
    use_attractor: bool
    alpha_policy: string  # fixed, exp_c_over_N, learned
    alpha_value: float
    c_value: float
    inject_layers: string  # last_N, all, custom
    N_inject: int
    attractor_inject: string  # cross_attn, residual, none
    d_state: int

# Training Configuration
training:
  batch_size: int
  gradient_accumulation_steps: int
  num_epochs: int
  learning_rate: float
  weight_decay: float
  max_grad_norm: float
  lr_scheduler_type: string
  warmup_steps: int
  bf16: bool
  logging_steps: int
  eval_steps: int
  save_steps: int
  seed: int
  deterministic: bool

# Data Configuration
data:
  dataset_name: string
  data_dir: path
  max_seq_length: int
  preprocessing_num_workers: int
  dataloader_num_workers: int

# Evaluation Configuration
evaluation:
  benchmarks: list
  niah_lengths: list
  niah_depths: list
  ruler_tasks: list
  ruler_max_length: int
  max_new_tokens: int
  temperature: float
  top_k: int
  top_p: float
  do_sample: bool
  metrics: list

# System Configuration
system:
  device: string
  num_gpus: int
  window_size: int
  gradient_checkpointing: bool
  cpu_offload: bool
  use_flash_attention: bool
  compile_model: bool
  log_memory: bool
  log_throughput: bool

# Ablation Settings (optional)
ablation_mode: string
ablation_values: list
```

---

## Usage Examples

### Running a Single Configuration

```bash
# Training
python scripts/train.py --config configs/qwen_1.5b_pi_spiral.yaml

# Evaluation
python scripts/evaluate.py --config configs/qwen_1.5b_pi_spiral.yaml

# Diagnostics
python scripts/diagnose.py --config configs/qwen_1.5b_pi_spiral.yaml
```

### Overriding Configuration Parameters

```bash
# Override specific parameters
python scripts/train.py \
  --config configs/qwen_1.5b_pi_spiral.yaml \
  --output_dir ./custom_results \
  --training.batch_size 2 \
  --model.pos_encoding.hybrid_K 8000
```

### Running Ablation Studies

```bash
# Run all ablation values automatically
python scripts/evaluate.py --config configs/ablation_irrational.yaml

# Run specific ablation value
python scripts/evaluate.py \
  --config configs/ablation_irrational.yaml \
  --ablation_value pi
```

### Loading Configuration in Python

```python
from src.config import ExperimentConfig

# Load from YAML
config = ExperimentConfig.from_yaml('configs/qwen_1.5b_pi_spiral.yaml')

# Modify programmatically
config.training.batch_size = 2
config.model.pos_encoding.hybrid_K = 8000

# Save modified config
config.to_yaml('configs/my_custom_config.yaml')
```

---

## Experiment Phase Mapping

| Phase | Configuration Files | Purpose |
|-------|-------------------|---------|
| Phase 0 | `base_config.yaml` | Setup and template |
| Phase 1 | Code modules | Module development |
| Phase 2 | `phase2_sanity.yaml`, `qwen_1.5b_pi_spiral.yaml` | Sanity runs |
| Phase 3 | `qwen_1.5b_pi_spiral.yaml` | Core benchmarks (small model) |
| Phase 4 | `llama_8b_pi_spiral.yaml` | Medium model evaluation |
| Phase 5 | `llama_34b_pi_spiral.yaml` | Heavyweight demo |
| Phase 6 | `ablation_*.yaml` | Ablation studies |
| Phase 7 | All configs + diagnostics | Diagnostics and analysis |
| Phase 8 | All configs | Cost and reliability |
| Phase 9 | All configs | Results packaging |
| Phase 10 | All configs | Decision tree |

---

## Creating Custom Configurations

To create a custom configuration:

1. **Start with base_config.yaml:**
   ```bash
   cp configs/base_config.yaml configs/my_experiment.yaml
   ```

2. **Modify key parameters:**
   - Set `experiment_name`
   - Choose model (`pretrained.model_name_or_path`)
   - Configure positional encoding (`model.pos_encoding`)
   - Set evaluation benchmarks and lengths
   - Adjust system settings for your hardware

3. **Validate configuration:**
   ```python
   from src.config import ExperimentConfig
   config = ExperimentConfig.from_yaml('configs/my_experiment.yaml')
   config.validate()  # Checks for errors
   ```

4. **Test with small dataset:**
   ```bash
   python scripts/train.py \
     --config configs/my_experiment.yaml \
     --data.niah_lengths [4000] \
     --training.max_steps 10
   ```

---

## Configuration Best Practices

1. **Memory Management:**
   - Use 4-bit quantization for models ≥ 8B
   - Enable gradient checkpointing for long sequences
   - Set appropriate window_size for your VRAM

2. **Reproducibility:**
   - Always set `training.seed`
   - Enable `training.deterministic`
   - Document any manual changes

3. **Experiment Tracking:**
   - Use descriptive `experiment_name`
   - Set unique `run_name` for variants
   - Keep `output_dir` organized by phase

4. **Baseline Comparisons:**
   - Run baselines with same evaluation protocol
   - Use same random seeds
   - Match batch sizes and other hyperparameters

5. **Ablation Studies:**
   - Change only one variable at a time
   - Use same base configuration
   - Run multiple seeds for statistical significance

---

## Troubleshooting

### Out of Memory (OOM)

- Reduce `training.batch_size`
- Increase `training.gradient_accumulation_steps`
- Enable `system.gradient_checkpointing`
- Reduce `system.window_size`
- Enable `pretrained.load_in_4bit`

### Slow Training

- Enable `system.use_flash_attention`
- Reduce `data.preprocessing_num_workers`
- Set `system.compile_model = true` (PyTorch 2.0+)
- Use smaller `evaluation.niah_lengths` for testing

### Configuration Errors

- Validate YAML syntax
- Check required fields are present
- Ensure paths exist
- Verify model names on Hugging Face

---

## Additional Resources

- **Experiment Plan:** `../exp plan.md`
- **Main README:** `../README.md`
- **API Documentation:** `../docs/`
- **Example Scripts:** `../examples/`

---

## Contact

For questions about configurations, please refer to the main README or open an issue.
