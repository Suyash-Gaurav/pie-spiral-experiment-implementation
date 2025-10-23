Here is a crisp, end-to-end experiment plan you can follow without getting lost. It is structured as phases with explicit deliverables, pass–fail gates, and checklists. It assumes Kaggle A100 16 GB for Qwen2.5-1.5B and Llama-3-8B, plus a single Llama-3-34B run with CPU offload for a heavyweight demo. I have internet access,

\# Phase 0. Repo, environments, data

_\*\*Goal:\*\* a clean, reproducible skeleton before touching experiments._

_1\. Repo layout_

_\`\`\`_

_pi\_spiral/_

_src/ # pi-spiral module and integration shims_

_configs/ # YAMLs for models, tasks, ablations_

_benches/ # NIAH, RULER subset, LongBench subset, InfiniteBench subset_

_scripts/ # run\_\*.py and plot\_\*.py_

_results/ # auto-populated by runs_

_figures/ # generated plots_

_docs/ # experiment log, paper ou_tline

\`\`\`

2\. Environments

\* Kaggle or Colab image with: PyTorch 2.x, flash-attn v2, bitsandbytes, transformers, vLLM or similar runtime, optional AutoGPTQ/AWQ, CUDA toolkit.

\* Pin versions in \`requirements.txt\` and save \`pip list\` after install.

\*\*Checklist\*\*

\* \[ \] Deterministic seeds set globally.

\* \[ \] \`torch.backends.cudnn.deterministic=True\` if needed.

\* \[ \] Confirm flash-attn and 4-bit quantization load successfully for all three models.

\# Phase 1. π-Spiral module and hybrid gate

\*\*Goal:\*\* a drop-in positional module with O(1) global attractor, plus a hybrid switch for local span.

1\. Implement

\* \`PiSpiralPositional\`: generates 2D unit vectors \`e\_n = \[cos(2π frac(nπ)), sin(...)\]\` on the fly.

\* \`AttractorState\`: incremental update \`C\_t = α C\_{t-1} + e\_t ⊗ h\_t\`.

\* Injection: cross-attention to \`C\_t\` in last N\_l layers or as a residual adapter.

\* Hybrid gate: RoPE for indices ≤ K, blend to π-Spiral beyond K with a smooth sigmoid.

2\. Config flags

\* \`--pos=rope|rope\_ntk|alibi|pi\_spiral|hybrid\`

\* \`--window=W\` sliding window for standard attention

\* \`--alpha\_policy=exp\_c\_over\_N|fixed|learned\`

\* \`--irrational=pi|e|sqrt2|phi|prng\`

\* \`--inject\_layers=lastN\`

\* \`--hybrid\_K=16000\` (default)

\*\*Checklist\*\*

\* \[ \] Unit tests: shape checks, determinism, incremental update matches full recompute.

\* \[ \] Microbenchmarks: O(1) state size confirmed, no VRAM growth with length.

\# Phase 2. Sanity runs on Qwen2.5-1.5B

\*\*Goal:\*\* prove nothing breaks and memory stays bounded.

1\. Short tasks at 4k to 32k

\* Tasks: tiny NIAH, simple QA.

\* Modes: RoPE, π-Spiral, Hybrid.

\* Verify no short-range regression beyond 1 percent for π-Spiral and none for Hybrid.

2\. Long stream smoke test

\* 1M tokens dummy stream with sliding window W=8k.

\* Log peak VRAM, tokens per second, and stable speed.

\*\*Pass gate\*\*

\* ≤ 1 percent delta on 4k to 32k tasks with Hybrid.

\* Flat VRAM across 64k to 1M length.

\# Phase 3. Core benchmarks on Qwen2.5-1.5B

\*\*Goal:\*\* establish long-context credibility on a small model.

1\. NIAH sweep

\* Lengths: 32k, 64k, 128k, 256k, 512k, 1M.

\* Depth grid: 0.1 to 0.9 of length.

\* Outputs: accuracy heatmaps, AUC vs depth.

2\. RULER subset

\* Tasks: counting, multi-needle, multi-hop, aggregation up to 256k.

\* Outputs: per-task accuracy curves and a radar plot.

3\. InfiniteBench subset

\* One or two tasks at 100k, 256k, 512k.

\* Outputs: task score vs length.

4\. System metrics

\* VRAM vs length, tokens per second vs length, state size constant with length.

\*\*Pass gate\*\*

\* ≥ 30 percent absolute gain over RoPE or RoPE-NTK on deep NIAH at ≥ 256k.

\* Smooth accuracy decay with length, no periodic oscillations.

\# Phase 4. Lift to Llama-3-8B

\*\*Goal:\*\* repeat the winning subset at 7 to 8B within 16 GB.

1\. Settings

\* 4-bit quant, bf16 activations if available, W=4k for stability.

\* Repeat NIAH to 512k and RULER to 256k, plus a LongBench-style multi-doc QA at 128k.

2\. Recheck short-range guard

\* ≤ 1 percent drop at 4k to 32k with Hybrid.

\*\*Pass gate\*\*

\* Same qualitative gains as small model and no short-range regressions.

\# Phase 5. One heavyweight demo on Llama-3-34B

\*\*Goal:\*\* show the O(1) memory story scales.

1\. Setup

\* 4-bit weights with CPU offload, W=2k, batch=1.

\* Single long NIAH at 128k and a 100k InfiniteBench task.

2\. Log memory traces and throughput.

\*\*Pass gate\*\*

\* Bounded VRAM and stable retrieval at depth. Throughput can be slow.

\# Phase 6. Ablations that decide authorship claims

\*\*Goal:\*\* separate the effect of non-periodicity, the attractor, and α policy.

1\. Irrational choice

\* π vs e vs √2 vs φ vs PRNG on Qwen 1.5B for NIAH 512k and one RULER task at 256k.

2\. Attractor on vs off

\* π-Spiral positions alone vs positions plus C\_t.

3\. Decay schedules

\* α = exp(−π/N), α = exp(−c) with c tuned, α learned scalar.

4\. Hybrid threshold K

\* K in {8k, 16k, 32k, 64k} with short-range QA guard.

\*\*Pass gate\*\*

\* If PRNG ≈ π and both beat RoPE, claim becomes non-periodic encoding plus O(1) attractor memory.

\* If attractor carries most gains, state the combo clearly.

\# Phase 7. Aliasing and stability diagnostics

\*\*Goal:\*\* show why it works, not just that it works.

1\. Positional collision

\* Cosine similarity of positional vectors at large offsets for RoPE vs π-Spiral.

2\. Spectral analysis

\* FFT of positional phase sequences to reveal periodic spikes in RoPE and flat spectrum for π-Spiral.

3\. Perplexity drift

\* Per-token loss vs position on long concatenations. Look for oscillations in RoPE that are absent in π-Spiral.

4\. Attention visualizations

\* Heatmaps demonstrating distant token addressability at high depths.

\*\*Pass gate\*\*

\* Clear reduction of periodic artifacts relative to RoPE and RoPE-NTK.

\# Phase 8. Cost and reliability

\*\*Goal:\*\* operational viability.

1\. Cost curves

\* VRAM vs length and tokens per second vs length for all modes.

2\. Stability

\* Three random seeds for each key setting.

\* Report mean and 95 percent CI.

3\. Failure cases

\* Collect and annotate cases where π-Spiral underperforms. Add a small troubleshooting note.

\# Phase 9. Results packaging

\*\*Goal:\*\* paper-ready artifacts.

1\. Tables

\* NIAH accuracy at fixed lengths and depths across methods and models.

\* RULER per-task accuracy.

\* System metrics summary.

2\. Figures

\* Retrieval heatmaps, length-sweep curves, radar plot, spectral plot, cost curves, ablation grid.

3\. Repro kit

\* Config YAMLs, exact commit hashes, environment dump, run scripts.

\# Phase 10. Decision tree and next actions

\*\*If all pass gates hold\*\*

\* Release code and checkpoints for reproducibility.

\*\*If π vs PRNG is indistinguishable\*\*

\* Reframe to “non-periodic spiral encodings”. Keep contributions precise and still submit.

\*\*If short-range regressions persist\*\*

\* Ship Hybrid as default, report K-sweep, and highlight no regression setting.

\# Run order and time budgeting

Phase 0, Phase 1 unit tests, Phase 2 sanity.

Phase 3 full small-model benchmarks.

Phase 4 medium-model subset and short-range guard.

Phase 6 ablations on small model, Phase 7 diagnostics.

Phase 5 single 34B demo, Phase 8 cost and reliability.

Phase 9 packaging and figure generation.

\# Logging schema

\* \`metrics.jsonl\`: task, length, depth, mode, accuracy, em, f1, rouge, ppl, seed, runtime.

\* \`system.csv\`: step, length, vram\_mb, tokens\_per\_sec, window, state\_size.

\* \`positional\_diag.csv\`: offset, cosine\_sim; \`spectral\_fft.csv\`: freq, magnitude.

\* Plots auto-saved under \`figures/\` with filenames embedding config.

\* Confirm Hybrid default K. I propose K=16k for 8B and K=8k for 1.5B.

\* Confirm which InfiniteBench tasks to prioritize for the first pass. for sanity check: NIAH

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML