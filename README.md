# π-Spiral Positional Encoding Implementation

A complete Python implementation of π-Spiral Positional Encoding for long-context transformer models, featuring O(1) memory complexity through attractor state mechanisms.
# π-Spiral: End-to-End Experiment Plan  

**Target:** Kaggle A100 (16 GB VRAM) | Models: Qwen2.5-1.5B, Llama-3-8B, Llama-3-34B (CPU offload)  
**Focus:** NIAH (sanity), RULER, InfiniteBench, LongBench subsets  
**Internet:** Enabled


## Overview

This package implements the π-Spiral positional encoding scheme, which uses irrational number properties to create non-periodic positional representations. This approach addresses the periodic aliasing issues in standard positional encodings like RoPE, enabling better performance on long-context tasks.

### Key Features

- **π-Spiral Positional Encoding**: Non-periodic encoding using irrational constants (π, e, √2, φ)
- **O(1) Attractor State**: Constant memory complexity for global context compression
- **Hybrid Encoding**: Smooth blending of RoPE (short-range) and π-Spiral (long-range)
- **Pre-trained Model Adapters**: Easy integration with Qwen, Llama, and other transformers
- **Comprehensive Benchmarks**: Support for NIAH, RULER, InfiniteBench, and LongBench
- **Flash Attention Support**: Efficient attention computation for long sequences
- 
### Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- PyTorch 2.0+

---

## Phase 0 – Repo, Environment, Data

**Goal:** Clean, reproducible foundation.



## Installation


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

### 2. Environment
- **Base:** Kaggle/Colab + PyTorch 2.x, CUDA 12.1
- **Key Packages:** `flash-attn>=2.6`, `bitsandbytes>=0.43.3`, `transformers>=4.41`, `accelerate`, `vLLM` (optional)
- **Pin versions** → `requirements.txt` + `pip list > env.txt`

### Checklist
- [ ] `torch.manual_seed(42)` + `numpy.random.seed(42)` globally  
- [ ] `torch.backends.cudnn.deterministic = True`  
- [ ] 4-bit + flash-attn load **without error** on all 3 models

---

## Phase 1 – π-Spiral Module & Hybrid Gate

**Goal:** Drop-in positional encoder with **O(1)** attractor memory.

### Core Components
| Component | Spec |
|--------|------|
| `PiSpiralPositional` | `e_n = [cos(2π {nπ}), sin(2π {nπ})]` |
| `AttractorState` | `C_t = α C_{t-1} + e_t ⊗ h_t` |
| Injection | Cross-attn to `C_t` in last `N_l` layers |
| Hybrid Gate | RoPE for ≤ K, sigmoid blend → π-Spiral |

### Config Flags
```yaml
--pos: rope | rope_ntk | alibi | pi_spiral | hybrid
--window: 8192
--alpha_policy: exp_c_over_N | fixed | learned
--irrational: pi | e | sqrt2 | phi | prng
--inject_layers: lastN
--hybrid_K: 16000
```

### Checklist
- [ ] Unit tests: shape, determinism, incremental = full recompute  
- [ ] Microbench: **State size = O(1)**, **VRAM flat** beyond 64k

---

## Phase 2 – Sanity Runs (Qwen2.5-1.5B)

**Goal:** No breakage, bounded memory.

### 1. Short Tasks (4k–32k)
- Tasks: Mini-NIAH, QA  
- Modes: RoPE, π-Spiral, Hybrid  
- **Pass:** ≤1% drop vs RoPE on 4k–32k

### 2. Long Stream Smoke Test
- 1M dummy tokens, `W=8k`  
- Log: **VRAM**, **tokens/sec**, **speed stability**

**Pass Gate:**  
- ≤1% delta on short tasks (Hybrid)  
- **Flat VRAM** from 64k → 1M

---

## Phase 3 – Core Benchmarks (Qwen2.5-1.5B)

**Goal:** Long-context credibility.

### 1. NIAH Sweep
- Lengths: 32k, 64k, 128k, 256k, 512k, 1M  
- Depths: 0.1 → 0.9  
- Output: **Accuracy heatmap**, **AUC vs depth**

### 2. RULER Subset
- Tasks: counting, multi-needle, multi-hop, aggregation (≤256k)  
- Output: **Per-task curves**, **radar plot**

### 3. InfiniteBench Subset
- 1–2 tasks @ 100k, 256k, 512k  
- Output: **Score vs length**

### 4. System Metrics
- VRAM, tokens/sec, state size (must be **constant**)

**Pass Gate:**  
- **≥30% absolute gain** over RoPE/NTK on deep NIAH ≥256k  
- No periodic accuracy oscillations

---

## Phase 4 – Lift to Llama-3-8B (16 GB)

**Goal:** Scale within memory budget.

### Settings
- 4-bit quant, `bf16`, `W=4k`  
- Repeat: NIAH (≤512k), RULER (≤256k), LongBench QA (128k)

### Short-Range Guard
- ≤1% drop @ 4k–32k (Hybrid)

**Pass Gate:**  
- Same qualitative gains  
- No short-range regression

---

## Phase 5 – Heavyweight Demo (Llama-3-34B)

**Goal:** Prove O(1) memory scales.

### Setup
- 4-bit + **CPU offload**, `W=2k`, batch=1  
- Run: NIAH @ 128k, InfiniteBench @ 100k

### Logs
- VRAM trace, throughput

**Pass Gate:**  
- Bounded VRAM  
- Stable retrieval at depth (throughput may be slow)

---

## Phase 6 – Ablations (Qwen2.5-1.5B)

**Goal:** Isolate contributions.

| Ablation | Variants |
|--------|----------|
| Irrational | π, e, √2, φ, PRNG |
| Attractor | On vs Off |
| α Policy | exp(−π/N), fixed, learned |
| Hybrid K | 8k, 16k, 32k, 64k |

**Pass Gate:**  
- If **PRNG ≈ π > RoPE** → claim: *non-periodic + O(1) attractor*  
- If attractor dominates → highlight combo

---

## Phase 7 – Diagnostics (Why It Works)

**Goal:** Mechanistic evidence.

1. **Positional Collision**  
   → Cosine sim of vectors at large offsets  
2. **Spectral FFT**  
   → RoPE: periodic spikes | π-Spiral: flat  
3. **Perplexity Drift**  
   → Loss vs position on long concat  
4. **Attention Heatmaps**  
   → Distant token focus at high depth

**Pass Gate:**  
- **Clear reduction** in periodic artifacts vs RoPE/NTK

---

## Phase 8 – Cost & Reliability

**Goal:** Operational readiness.

1. **Cost Curves**  
   - VRAM vs length  
   - Tokens/sec vs length  
2. **Reproducibility**  
   - 3 seeds per key config  
   - Report mean ± 95% CI  
3. **Failure Cases**  
   - Collect + annotate underperformance

---

## Phase 9 – Results Packaging

**Goal:** Paper-ready artifacts.

### Tables
- NIAH accuracy (length × depth × method)  
- RULER per-task  
- System metrics summary

### Figures
- Retrieval heatmaps  
- Length-sweep curves  
- Radar plot  
- Spectral FFT  
- Cost curves  
- Ablation grid

### Repro Kit
- Config YAMLs  
- Commit hashes  
- `env.txt`  
- Run scripts

---

## Phase 10 – Decision Tree

| Outcome | Action |
|-------|--------|
| All gates pass | Release code + checkpoints |
| π ≈ PRNG | Reframe: *non-periodic spiral encodings* |
| Short-range regression | Ship **Hybrid default**, report K-sweep |

---

## Run Order & Time Budget

```
0 → 1 (tests) → 2 (sanity)
3 (small model full) → 4 (8B subset) → 6 (ablations) → 7 (diagnostics)
5 (34B demo) → 8 (cost) → 9 (packaging)
```

---

## Logging Schema

```json
// metrics.jsonl
{ "task": "niah", "length": 128000, "depth": 0.7, "mode": "hybrid", "acc": 0.78, "seed": 42, "runtime_s": 120 }

// system.csv
step,length,vram_mb,tokens_per_sec,window,state_size

// positional_diag.csv
offset,cosine_sim_rope,cosine_sim_pi

// spectral_fft.csv
freq,magnitude_rope,magnitude_pi
```

Plots auto-saved: `figures/niah_heatmap_hybrid_128k.png`

---

## Final Notes

- **Hybrid default K:**  
  - 1.5B: `K=8k`  
  - 8B: `K=16k`  
- **InfiniteBench priority (first pass):**  
  - `PassKey` (sanity)  
  - `NumberString` (counting)  
  - `Retrieval` (long-doc)

---

**Ready to run. No detours.**

