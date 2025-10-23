# Experiments Guide

This document provides detailed instructions for running all experiment phases of the π-Spiral Positional Encoding project, following the experiment plan methodology.

## Table of Contents

- [Overview](#overview)
- [Experiment Phases](#experiment-phases)
- [Running Experiments](#running-experiments)
- [Pass Gates and Validation](#pass-gates-and-validation)
- [Results Analysis](#results-analysis)
- [Troubleshooting](#troubleshooting)

---

## Overview

The experiment plan consists of 10 phases, from initial setup to final decision-making. Each phase has specific goals, deliverables, and pass gates that must be satisfied before proceeding.

### Hardware Requirements

- **Phase 0-3 (Qwen 1.5B):** Single A100 16GB or equivalent
- **Phase 4 (Llama 8B):** Single A100 16GB with 4-bit quantization
- **Phase 5 (Llama 34B):** Single A100 16GB with 4-bit + CPU offload
- **Phases 6-10:** Same as Phase 3

### Time Budget

- **Phase 0-1:** 1-2 hours (setup and implementation)
- **Phase 2:** 2-4 hours (sanity runs)
- **Phase 3:** 8-12 hours (core benchmarks)
- **Phase 4:** 6-10 hours (medium model)
- **Phase 5:** 2-4 hours (single heavyweight demo)
- **Phase 6:** 6-8 hours (ablations)
- **Phase 7:** 2-4 hours (diagnostics)
- **Phase 8:** 4-6 hours (cost and reliability)
- **Phase 9:** 2-4 hours (packaging)
- **Phase 10:** 1-2 hours (decision tree)

**Total:** ~35-55 hours

---

## Experiment Phases

### Phase 0: Repository, Environments, Data

**Goal:** Clean, reproducible skeleton before touching experiments.

**Checklist:**
- [x] Repository structure created
- [x] Dependencies installed (`requirements.txt`)
- [x] Deterministic seeds set globally
- [x] Flash attention and quantization verified
- [x] CUDA toolkit available

**Commands:**
```bash
# Install dependencies
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import flash_attn; print('Flash Attention: OK')"
python -c "import bitsandbytes; print('Bitsandbytes: OK')"

# Set deterministic behavior
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

**Deliverables:**
- Working environment
- All dependencies installed
- Deterministic configuration verified

---

### Phase 1: π-Spiral Module and Hybrid Gate

**Goal:** Drop-in positional module with O(1) global attractor and hybrid switch.

**Implementation:**
- [x] `PiSpiralPositional`: 2D unit vectors with irrational constants
- [x] `AttractorState`: Incremental update with configurable alpha
- [x] `HybridPositionalEncoding`: Smooth RoPE-to-π-Spiral transition
- [x] Configuration flags for all options

**Unit Tests:**
```bash
# Run encoding tests
pytest tests/test_encodings.py -v

# Check specific tests
pytest tests/test_encodings.py::test_pi_spiral_shape -v
pytest tests/test_encodings.py::test_attractor_state -v
pytest tests/test_encodings.py::test_hybrid_encoding -v
```

**Microbenchmarks:**
```bash
# Verify O(1) state size
python -c "
from src.encodings import AttractorState
import torch

attractor = AttractorState(d_model=512)
for length in [1000, 10000, 100000]:
    h = torch.randn(1, length, 512)
    e = torch.randn(1, length, 2)
    state = attractor(h, e)
    print(f'Length {length}: State size {state.shape}')
"
```

**Pass Gate:**
- [x] Unit tests pass
- [x] Shape checks correct
- [x] Deterministic behavior verified
- [x] O(1) state size confirmed

---

### Phase 2: Sanity Runs on Qwen2.5-1.5B

**Goal:** Prove nothing breaks and memory stays bounded.

**Configuration:** `configs/phase2_sanity.yaml`

**Tasks:**

#### 2.1 Short Tasks (4k to 32k)
```bash
# Run sanity checks with all three modes
python scripts/evaluate.py \
  --config configs/phase2_sanity.yaml \
  --evaluation.niah_lengths [4000,8000,16000,32000] \
  --model.pos_encoding.type rope

python scripts/evaluate.py \
  --config configs/phase2_sanity.yaml \
  --evaluation.niah_lengths [4000,8000,16000,32000] \
  --model.pos_encoding.type pi_spiral

python scripts/evaluate.py \
  --config configs/phase2_sanity.yaml \
  --evaluation.niah_lengths [4000,8000,16000,32000] \
  --model.pos_encoding.type hybrid
```

**Expected Results:**
- RoPE: Baseline performance
- π-Spiral: ≤ 1% regression acceptable
- Hybrid: ≤ 1% regression (strict requirement)

#### 2.2 Long Stream Smoke Test (1M tokens)
```bash
# Run memory profiling on long sequence
python scripts/diagnose.py \
  --config configs/phase2_sanity.yaml \
  --mode profiling \
  --profiling.max_length 1000000 \
  --profiling.window_size 8000
```

**Expected Results:**
- Flat VRAM across 64k to 1M
- Stable tokens/sec
- No memory leaks

**Pass Gate:**
- ≤ 1% delta on 4k-32k with Hybrid vs RoPE
- Flat VRAM across 64k to 1M length
- Stable throughput

**Validation:**
```bash
# Check pass gates
python scripts/validate_phase.py --phase 2 --results ./results/phase2_sanity
```

---

### Phase 3: Core Benchmarks on Qwen2.5-1.5B

**Goal:** Establish long-context credibility on small model.

**Configuration:** `configs/qwen_1.5b_pi_spiral.yaml`

**Tasks:**

#### 3.1 NIAH Sweep
```bash
# Full NIAH sweep
python scripts/evaluate.py \
  --config configs/qwen_1.5b_pi_spiral.yaml \
  --evaluation.benchmarks [niah] \
  --evaluation.niah_lengths [32000,64000,128000,256000,512000,1000000]
```

**Outputs:**
- Accuracy heatmaps (length × depth)
- AUC vs depth curves
- Per-configuration accuracy

#### 3.2 RULER Subset
```bash
# RULER tasks up to 256k
python scripts/evaluate.py \
  --config configs/qwen_1.5b_pi_spiral.yaml \
  --evaluation.benchmarks [ruler] \
  --evaluation.ruler_max_length 256000
```

**Outputs:**
- Per-task accuracy curves
- Radar plot across tasks
- Aggregated scores

#### 3.3 InfiniteBench Subset
```bash
# InfiniteBench at long contexts
python scripts/evaluate.py \
  --config configs/qwen_1.5b_pi_spiral.yaml \
  --evaluation.benchmarks [infinitebench]
```

**Outputs:**
- Task scores vs length
- Comparison with baselines

#### 3.4 System Metrics
```bash
# Profile memory and throughput
python scripts/diagnose.py \
  --config configs/qwen_1.5b_pi_spiral.yaml \
  --mode profiling
```

**Outputs:**
- VRAM vs length curves
- Tokens/sec vs length curves
- State size verification

**Pass Gate:**
- ≥ 30% absolute gain over RoPE on deep NIAH at ≥ 256k
- Smooth accuracy decay with length
- No periodic oscillations
- Flat memory scaling

**Validation:**
```bash
# Compare with baseline
python scripts/compare_results.py \
  --experiment ./results/qwen_1.5b_pi_spiral \
  --baseline ./results/baseline_rope \
  --metrics accuracy,memory,throughput
```

---

### Phase 4: Lift to Llama-3-8B

**Goal:** Repeat winning subset at 7-8B within 16GB.

**Configuration:** `configs/llama_8b_pi_spiral.yaml`

**Tasks:**

#### 4.1 Main Evaluation
```bash
# NIAH to 512k
python scripts/evaluate.py \
  --config configs/llama_8b_pi_spiral.yaml \
  --evaluation.benchmarks [niah] \
  --evaluation.niah_lengths [32000,64000,128000,256000,512000]

# RULER to 256k
python scripts/evaluate.py \
  --config configs/llama_8b_pi_spiral.yaml \
  --evaluation.benchmarks [ruler] \
  --evaluation.ruler_max_length 256000
```

#### 4.2 Short-Range Guard
```bash
# Verify no regression at short ranges
python scripts/evaluate.py \
  --config configs/llama_8b_pi_spiral.yaml \
  --evaluation.niah_lengths [4000,8000,16000,32000]
```

**Pass Gate:**
- Same qualitative gains as small model
- ≤ 1% drop at 4k-32k with Hybrid
- Stable with 4-bit quantization

---

### Phase 5: One Heavyweight Demo on Llama-3-34B

**Goal:** Show O(1) memory story scales.

**Configuration:** `configs/llama_34b_pi_spiral.yaml`

**Tasks:**

#### 5.1 Single Long NIAH
```bash
# NIAH at 128k
python scripts/evaluate.py \
  --config configs/llama_34b_pi_spiral.yaml \
  --evaluation.benchmarks [niah] \
  --evaluation.niah_lengths [128000]
```

#### 5.2 InfiniteBench Task
```bash
# One task at 100k
python scripts/evaluate.py \
  --config configs/llama_34b_pi_spiral.yaml \
  --evaluation.benchmarks [infinitebench]
```

#### 5.3 Memory Profiling
```bash
# Detailed memory traces
python scripts/diagnose.py \
  --config configs/llama_34b_pi_spiral.yaml \
  --mode profiling \
  --profiling.detailed true
```

**Pass Gate:**
- Bounded VRAM (no growth with length)
- Stable retrieval at depth
- Throughput can be slow (acceptable)

---

### Phase 6: Ablations That Decide Authorship Claims

**Goal:** Separate effects of non-periodicity, attractor, and alpha policy.

**Tasks:**

#### 6.1 Irrational Choice
```bash
# Compare π, e, √2, φ, PRNG
python scripts/evaluate.py \
  --config configs/ablation_irrational.yaml
```

**Analysis:**
- If PRNG ≈ π: Claim is "non-periodic encoding"
- If π significantly better: Claim is "π-based encoding"

#### 6.2 Attractor On vs Off
```bash
# Compare with and without attractor
python scripts/evaluate.py \
  --config configs/ablation_attractor.yaml
```

**Analysis:**
- Quantify attractor contribution
- Separate positional encoding vs global context effects

#### 6.3 Decay Schedules
```bash
# Compare alpha policies
python scripts/evaluate.py \
  --config configs/ablation_alpha_policy.yaml
```

**Analysis:**
- Identify optimal decay schedule
- Validate theoretical motivation (exp(-π/N))

#### 6.4 Hybrid Threshold K
```bash
# Sweep K values
python scripts/evaluate.py \
  --config configs/ablation_hybrid_k.yaml
```

**Analysis:**
- Find optimal transition point
- Verify short-range preservation

**Pass Gate:**
- Clear ranking of ablation variants
- Statistical significance (multiple seeds)
- Actionable insights for claims

**Validation:**
```bash
# Statistical analysis
python scripts/analyze_ablations.py \
  --results ./results/ablations \
  --num_seeds 3 \
  --confidence 0.95
```

---

### Phase 7: Aliasing and Stability Diagnostics

**Goal:** Show why it works, not just that it works.

**Tasks:**

#### 7.1 Positional Collision
```bash
# Analyze cosine similarity at large offsets
python scripts/diagnose.py \
  --config configs/qwen_1.5b_pi_spiral.yaml \
  --mode encoding_analysis \
  --analysis.type collision
```

#### 7.2 Spectral Analysis
```bash
# FFT of positional sequences
python scripts/diagnose.py \
  --config configs/qwen_1.5b_pi_spiral.yaml \
  --mode encoding_analysis \
  --analysis.type spectral
```

#### 7.3 Perplexity Drift
```bash
# Per-token loss vs position
python scripts/diagnose.py \
  --config configs/qwen_1.5b_pi_spiral.yaml \
  --mode perplexity_drift
```

#### 7.4 Attention Visualizations
```bash
# Attention heatmaps
python scripts/diagnose.py \
  --config configs/qwen_1.5b_pi_spiral.yaml \
  --mode attention_viz
```

**Pass Gate:**
- Clear reduction of periodic artifacts vs RoPE
- Flat spectrum for π-Spiral
- No oscillations in perplexity

---

### Phase 8: Cost and Reliability

**Goal:** Operational viability.

**Tasks:**

#### 8.1 Cost Curves
```bash
# Generate cost curves
python scripts/diagnose.py \
  --config configs/qwen_1.5b_pi_spiral.yaml \
  --mode profiling \
  --profiling.generate_curves true
```

#### 8.2 Stability (Multiple Seeds)
```bash
# Run with 3 different seeds
for seed in 42 123 456; do
  python scripts/evaluate.py \
    --config configs/qwen_1.5b_pi_spiral.yaml \
    --training.seed $seed \
    --run_name seed_$seed
done

# Aggregate results
python scripts/aggregate_seeds.py \
  --results ./results/qwen_1.5b_pi_spiral \
  --seeds 42,123,456
```

#### 8.3 Failure Cases
```bash
# Collect failure cases
python scripts/analyze_failures.py \
  --results ./results/qwen_1.5b_pi_spiral \
  --threshold 0.5
```

**Outputs:**
- Mean and 95% CI for all metrics
- Failure case annotations
- Troubleshooting notes

---

### Phase 9: Results Packaging

**Goal:** Paper-ready artifacts.

**Tasks:**

#### 9.1 Generate Tables
```bash
# Create all tables
python scripts/generate_tables.py \
  --results ./results \
  --output ./figures/tables
```

**Tables:**
- NIAH accuracy (length × depth × method × model)
- RULER per-task accuracy
- System metrics summary
- Ablation results

#### 9.2 Generate Figures
```bash
# Create all figures
python scripts/generate_figures.py \
  --results ./results \
  --output ./figures
```

**Figures:**
- Retrieval heatmaps
- Length-sweep curves
- Radar plots
- Spectral plots
- Cost curves
- Ablation grids

#### 9.3 Reproducibility Kit
```bash
# Package reproducibility kit
python scripts/package_repro_kit.py \
  --output ./repro_kit
```

**Contents:**
- All configuration YAMLs
- Exact commit hash
- Environment dump (`pip list`)
- Run scripts
- Results summary

---

### Phase 10: Decision Tree and Next Actions

**Goal:** Interpret results and decide next steps.

**Decision Tree:**

```
All pass gates hold?
├─ YES → Release code and checkpoints
│         Document all claims
│         Prepare publication
│
├─ π vs PRNG indistinguishable?
│  └─ Reframe to "non-periodic spiral encodings"
│     Keep contributions precise
│     Still submit
│
├─ Short-range regressions persist?
│  └─ Ship Hybrid as default
│     Report K-sweep results
│     Highlight no-regression setting
│
└─ Attractor provides most gains?
   └─ Emphasize O(1) global context
      Position encoding as supporting mechanism
      Highlight combined approach
```

**Validation Script:**
```bash
# Run decision tree analysis
python scripts/decision_tree.py \
  --results ./results \
  --pass_gates ./configs/pass_gates.yaml
```

**Outputs:**
- Pass/fail status for each gate
- Recommended next actions
- Claims to emphasize
- Potential concerns

---

## Running Experiments

### Sequential Execution

Run phases in order:

```bash
# Phase 0-1: Setup (manual)
# Phase 2: Sanity runs
bash scripts/run_phase2.sh

# Phase 3: Core benchmarks
bash scripts/run_phase3.sh

# Phase 4: Medium model
bash scripts/run_phase4.sh

# Phase 5: Heavyweight demo
bash scripts/run_phase5.sh

# Phase 6: Ablations
bash scripts/run_phase6.sh

# Phase 7: Diagnostics
bash scripts/run_phase7.sh

# Phase 8: Cost and reliability
bash scripts/run_phase8.sh

# Phase 9: Packaging
bash scripts/run_phase9.sh

# Phase 10: Decision tree
bash scripts/run_phase10.sh
```

### Parallel Execution

Some phases can run in parallel:

```bash
# Phase 6 ablations can run in parallel
python scripts/evaluate.py --config configs/ablation_irrational.yaml &
python scripts/evaluate.py --config configs/ablation_attractor.yaml &
python scripts/evaluate.py --config configs/ablation_alpha_policy.yaml &
python scripts/evaluate.py --config configs/ablation_hybrid_k.yaml &
wait

# Phase 7 diagnostics can run in parallel
python scripts/diagnose.py --mode encoding_analysis &
python scripts/diagnose.py --mode attention_viz &
python scripts/diagnose.py --mode profiling &
wait
```

### Resuming Experiments

If interrupted:

```bash
# Check last completed phase
python scripts/check_progress.py --results ./results

# Resume from specific phase
bash scripts/run_phase3.sh --resume --checkpoint ./results/phase3/checkpoint-1000
```

---

## Pass Gates and Validation

### Automated Validation

```bash
# Validate all pass gates
python scripts/validate_all_phases.py \
  --results ./results \
  --pass_gates ./configs/pass_gates.yaml
```

### Manual Validation

Check each pass gate manually:

**Phase 2:**
- [ ] Hybrid ≤ 1% regression on 4k-32k
- [ ] Flat VRAM 64k to 1M
- [ ] Stable tokens/sec

**Phase 3:**
- [ ] ≥ 30% gain over RoPE at 256k+
- [ ] Smooth accuracy decay
- [ ] No periodic oscillations

**Phase 4:**
- [ ] Same gains as Phase 3
- [ ] No short-range regression

**Phase 5:**
- [ ] Bounded VRAM
- [ ] Stable retrieval

**Phase 6:**
- [ ] Clear ablation rankings
- [ ] Statistical significance

**Phase 7:**
- [ ] Reduced periodic artifacts
- [ ] Flat spectrum

---

## Results Analysis

### Comparing with Baselines

```bash
# Generate comparison report
python scripts/compare_with_baselines.py \
  --experiment ./results/qwen_1.5b_pi_spiral \
  --baselines ./results/baseline_rope,./results/baseline_rope_ntk,./results/baseline_alibi \
  --output ./reports/comparison.pdf
```

### Statistical Analysis

```bash
# Statistical significance testing
python scripts/statistical_analysis.py \
  --results ./results \
  --test t_test \
  --confidence 0.95
```

### Visualization

```bash
# Generate all visualizations
python scripts/visualize_all.py \
  --results ./results \
  --output ./figures
```

---

## Troubleshooting

### Common Issues

**Out of Memory:**
- Reduce batch size
- Enable gradient checkpointing
- Use smaller window size
- Enable CPU offload (Phase 5)

**Slow Training:**
- Enable flash attention
- Use mixed precision
- Reduce logging frequency
- Use smaller models for testing

**Poor Results:**
- Check configuration
- Verify data quality
- Try different hyperparameters
- Run ablations to isolate issues

**Reproducibility Issues:**
- Set all random seeds
- Enable deterministic mode
- Document environment exactly
- Save all configurations

### Getting Help

1. Check documentation
2. Review example scripts
3. Examine test cases
4. Open GitHub issue
5. Contact maintainers

---

## Logging Schema

All experiments log to standardized formats:

**metrics.jsonl:**
```json
{"task": "niah", "length": 128000, "depth": 0.5, "mode": "hybrid", 
 "accuracy": 0.85, "em": 0.82, "f1": 0.87, "seed": 42, "runtime": 120.5}
```

**system.csv:**
```csv
step,length,vram_mb,tokens_per_sec,window,state_size
100,64000,12500,45.2,8000,262144
```

**positional_diag.csv:**
```csv
offset,cosine_sim_pi_spiral,cosine_sim_rope
1000,0.023,0.876
5000,0.012,0.765
```

---

## Best Practices

1. **Always run baselines first** - Establish performance ceiling
2. **Use multiple seeds** - Ensure statistical validity
3. **Document everything** - Configurations, commands, observations
4. **Save checkpoints** - Enable resumption
5. **Monitor resources** - Track memory and compute usage
6. **Validate incrementally** - Check pass gates after each phase
7. **Compare fairly** - Same evaluation protocol for all methods
8. **Visualize early** - Catch issues before full runs
9. **Archive results** - Keep all experimental data
10. **Share findings** - Document insights and failures

---

**Document Version:** 1.0  
**Last Updated:** 2024-10-23  
**Maintainer:** π-Spiral Team
