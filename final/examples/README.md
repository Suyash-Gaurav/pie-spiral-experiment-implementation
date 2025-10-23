# Examples Directory

This directory contains practical, self-contained example scripts demonstrating how to use the π-Spiral Positional Encoding implementation. Each example is designed to be run independently and showcases different aspects of the system.

## Table of Contents

- [Quick Start](#quick-start)
- [Example Scripts](#example-scripts)
- [Running Examples](#running-examples)
- [Example Outputs](#example-outputs)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

All examples can be run directly from the project root:

```bash
# Make sure you're in the project root (final/)
cd final

# Run any example
python examples/example_train_simple.py
python examples/example_evaluate_niah.py
python examples/example_diagnostic.py
python examples/example_custom_encoding.py
python examples/example_pretrained_adapter.py
```

**Prerequisites:**
- Python 3.8+
- All dependencies installed: `pip install -r requirements.txt`
- CUDA-capable GPU (recommended, but CPU works for small examples)

---

## Example Scripts

### 1. `example_train_simple.py`

**Purpose:** Demonstrates basic training with π-Spiral encoding from scratch.

**What it does:**
- Loads a base configuration
- Creates a simple training dataset
- Initializes a small transformer model with π-Spiral encoding
- Trains for a few steps
- Saves the trained model

**Key Features:**
- Self-contained training loop
- Small model for quick execution
- Demonstrates hybrid encoding setup
- Shows attractor state configuration

**Estimated Runtime:** 2-5 minutes (GPU), 10-15 minutes (CPU)

**Output:**
- Training logs in `./logs/`
- Trained model in `./results/simple_training/`
- Loss curves and metrics

**Usage:**
```bash
python examples/example_train_simple.py
```

**Learning Points:**
- How to configure π-Spiral encoding
- How to initialize models with custom positional encodings
- Basic training workflow
- Model saving and checkpointing

---

### 2. `example_evaluate_niah.py`

**Purpose:** Demonstrates NIAH (Needle in a Haystack) evaluation.

**What it does:**
- Loads a pre-trained model (Qwen2.5-1.5B)
- Injects π-Spiral encoding using the adapter
- Creates NIAH test cases with varying context lengths and depths
- Evaluates retrieval accuracy
- Generates summary statistics

**Key Features:**
- Pre-trained model adaptation
- NIAH dataset generation
- Evaluation across multiple context lengths
- Accuracy analysis by length and depth

**Estimated Runtime:** 5-10 minutes (GPU), 20-30 minutes (CPU)

**Output:**
- Evaluation results in `./results/niah_evaluation/`
- JSON file with detailed results
- Accuracy statistics by length and depth

**Usage:**
```bash
python examples/example_evaluate_niah.py
```

**Learning Points:**
- How to adapt pre-trained models
- NIAH benchmark methodology
- Evaluation workflow
- Results analysis and interpretation

---

### 3. `example_diagnostic.py`

**Purpose:** Demonstrates diagnostic analyses comparing π-Spiral and RoPE.

**What it does:**
- Analyzes positional collision (cosine similarity at large offsets)
- Performs spectral analysis (FFT of positional sequences)
- Profiles memory usage across sequence lengths
- Visualizes encoding patterns

**Key Features:**
- Positional collision analysis
- Frequency spectrum comparison
- Memory profiling
- Visualization generation

**Estimated Runtime:** 3-5 minutes

**Output:**
- Diagnostic plots in `./results/diagnostics/`
- `spectral_analysis.png` - Frequency spectrum comparison
- `encoding_patterns.png` - Positional encoding heatmaps
- Console output with analysis results

**Usage:**
```bash
python examples/example_diagnostic.py
```

**Learning Points:**
- Why π-Spiral works (non-periodic properties)
- How to analyze positional encodings
- Spectral analysis interpretation
- Memory profiling techniques

---

### 4. `example_custom_encoding.py`

**Purpose:** Demonstrates how to create and use custom encoding configurations.

**What it does:**
- Tests different irrational constants (π, e, √2, φ)
- Compares hybrid encoding with different K values
- Shows different alpha decay policies for attractor state
- Creates custom configuration programmatically
- Demonstrates configuration saving and loading

**Key Features:**
- Irrational constant comparison
- Hybrid K tuning
- Alpha policy exploration
- Programmatic configuration creation
- YAML export

**Estimated Runtime:** 1-2 minutes

**Output:**
- Custom configuration file in `./configs/my_custom_config.yaml`
- Console output with comparison results
- Configuration examples

**Usage:**
```bash
python examples/example_custom_encoding.py
```

**Learning Points:**
- How to customize positional encodings
- Configuration system usage
- Parameter tuning strategies
- Creating experiment variants

---

### 5. `example_pretrained_adapter.py`

**Purpose:** Demonstrates adapting pre-trained models with π-Spiral encoding.

**What it does:**
- Loads a pre-trained model from Hugging Face
- Injects π-Spiral encoding using the adapter interface
- Tests with short and long contexts
- Compares different adapter configurations
- Shows generation with adapted models

**Key Features:**
- Pre-trained model loading
- Adapter injection
- Short vs long context handling
- Configuration comparison
- Text generation

**Estimated Runtime:** 3-5 minutes (GPU), 10-15 minutes (CPU)

**Output:**
- Generated text samples
- Console output with configuration comparisons
- Demonstration of adapter flexibility

**Usage:**
```bash
python examples/example_pretrained_adapter.py
```

**Learning Points:**
- How to adapt existing models
- Adapter interface usage
- Hybrid encoding behavior
- Model integration strategies

---

## Running Examples

### Basic Execution

Run any example directly:

```bash
python examples/example_train_simple.py
```

### With Custom Parameters

Some examples accept command-line arguments (check the script for details):

```bash
# Example with custom model
python examples/example_pretrained_adapter.py --model meta-llama/Llama-3-8B

# Example with custom config
python examples/example_train_simple.py --config configs/my_config.yaml
```

### Running All Examples

To run all examples in sequence:

```bash
for script in examples/example_*.py; do
    echo "Running $script..."
    python "$script"
    echo "---"
done
```

### Running in Jupyter Notebook

Examples can be adapted for Jupyter notebooks:

```python
# In a Jupyter cell
%run examples/example_diagnostic.py
```

Or copy the code into notebook cells for interactive exploration.

---

## Example Outputs

### Expected Directory Structure After Running Examples

```
final/
├── examples/
│   ├── *.py (example scripts)
│   └── README.md (this file)
├── results/
│   ├── simple_training/
│   │   └── final_model/
│   ├── niah_evaluation/
│   │   └── results.json
│   └── diagnostics/
│       ├── spectral_analysis.png
│       └── encoding_patterns.png
├── logs/
│   └── (training logs)
└── configs/
    └── my_custom_config.yaml
```

### Sample Output: NIAH Evaluation

```
Overall Accuracy: 85.0%

Accuracy by Context Length:
   4000 tokens: 100.0% (5/5)
   8000 tokens: 90.0% (9/10)
  16000 tokens: 75.0% (15/20)

Accuracy by Depth:
  10.0%: 95.0% (19/20)
  50.0%: 85.0% (17/20)
  90.0%: 75.0% (15/20)
```

### Sample Output: Diagnostic Analysis

```
Positional Collision Analysis:
Offset     π-Spiral        RoPE
----------------------------------------
1000       0.0234          0.8765
5000       0.0123          0.7654
10000      0.0089          0.6543

Interpretation:
  - Lower similarity = better positional discrimination
  - π-Spiral shows lower similarity at large offsets
  - RoPE shows periodic patterns (aliasing)
```

---

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Make sure you're in the project root
cd final

# Run from project root
python examples/example_train_simple.py
```

#### 2. Out of Memory (OOM)

**Problem:** CUDA out of memory error

**Solutions:**
- Use smaller models or batch sizes
- Reduce sequence lengths in examples
- Enable CPU offload
- Use quantization (4-bit/8-bit)

```python
# Modify example to use smaller settings
config.training.batch_size = 1
config.data.max_seq_length = 2048
```

#### 3. Model Download Issues

**Problem:** Cannot download pre-trained models

**Solutions:**
- Check internet connection
- Set Hugging Face cache directory: `export HF_HOME=/path/to/cache`
- Use local model path if already downloaded
- Check Hugging Face authentication for gated models

```bash
# Login to Hugging Face (for gated models like Llama)
huggingface-cli login
```

#### 4. Slow Execution

**Problem:** Examples run very slowly

**Solutions:**
- Use GPU instead of CPU
- Reduce dataset sizes in examples
- Enable flash attention
- Use smaller models for testing

```python
# Modify for faster execution
num_samples = 5  # Reduce from default
niah_lengths = [4000, 8000]  # Use shorter contexts
```

#### 5. Missing Dependencies

**Problem:** Import errors for specific packages

**Solution:**
```bash
# Install all dependencies
pip install -r requirements.txt

# Install flash-attention separately (requires CUDA)
pip install flash-attn --no-build-isolation

# Install optional dependencies
pip install wandb tensorboard plotly
```

---

## Customizing Examples

### Modifying Parameters

All examples can be easily customized by editing the script:

```python
# In example_evaluate_niah.py
# Change these lines:
niah_lengths = [4000, 8000, 16000]  # Add more lengths
niah_depths = [0.1, 0.5, 0.9]       # Add more depths
num_samples = 5                      # Increase for more robust results
```

### Using Different Models

```python
# In example_pretrained_adapter.py
# Change model:
model_name = "meta-llama/Llama-3-8B"  # Instead of Qwen
```

### Saving Results

Examples automatically save results, but you can customize:

```python
# Custom output directory
output_dir = './my_results/experiment_1'
os.makedirs(output_dir, exist_ok=True)

# Save with custom name
results_path = os.path.join(output_dir, 'my_results.json')
```

---

## Integration with Main Scripts

Examples demonstrate concepts that can be used with main scripts:

### From Example to Production

**Example:**
```bash
python examples/example_train_simple.py
```

**Production:**
```bash
python scripts/train.py --config configs/qwen_1.5b_pi_spiral.yaml
```

**Example:**
```bash
python examples/example_evaluate_niah.py
```

**Production:**
```bash
python scripts/evaluate.py --config configs/qwen_1.5b_pi_spiral.yaml
```

**Example:**
```bash
python examples/example_diagnostic.py
```

**Production:**
```bash
python scripts/diagnose.py --config configs/qwen_1.5b_pi_spiral.yaml --mode all
```

---

## Learning Path

Recommended order for learning:

1. **Start with:** `example_custom_encoding.py`
   - Understand configuration options
   - Learn about different encoding types
   - Explore parameter effects

2. **Then try:** `example_pretrained_adapter.py`
   - Learn model adaptation
   - See practical usage
   - Understand hybrid encoding behavior

3. **Next:** `example_diagnostic.py`
   - Understand why π-Spiral works
   - Learn analysis techniques
   - Interpret results

4. **Then:** `example_evaluate_niah.py`
   - Learn evaluation methodology
   - Understand benchmarks
   - Analyze performance

5. **Finally:** `example_train_simple.py`
   - Complete training workflow
   - End-to-end understanding
   - Ready for full experiments

---

## Additional Resources

- **Main README:** `../README.md` - Complete project documentation
- **Configuration Guide:** `../configs/README.md` - All configuration options
- **Experiment Plan:** `../../exp plan.md` - Full experiment methodology
- **API Documentation:** `../docs/` - Detailed API reference

---

## Contributing Examples

To add new examples:

1. Create a new file: `example_your_feature.py`
2. Follow the existing structure:
   - Clear docstring at the top
   - Step-by-step execution with numbered sections
   - Informative console output
   - Save results to appropriate directory
   - Summary and next steps at the end
3. Update this README with your example
4. Test thoroughly before committing

---

## Questions or Issues?

- Check the main README for general documentation
- Review the configuration guide for parameter details
- Examine the source code for implementation details
- Open an issue on GitHub for bugs or questions

---

**Happy Experimenting with π-Spiral Positional Encoding!** 🎯
