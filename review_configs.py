#!/usr/bin/env python3
"""Script to review all configuration files and identify coverage."""

import yaml
from pathlib import Path

def load_yaml(filepath):
    """Load YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def review_configs():
    """Review all configuration files."""
    config_dir = Path('./final/configs')
    config_files = sorted(config_dir.glob('*.yaml'))
    
    print("=" * 80)
    print("CONFIGURATION FILES REVIEW")
    print("=" * 80)
    
    for config_file in config_files:
        print(f"\n{'=' * 80}")
        print(f"File: {config_file.name}")
        print('=' * 80)
        
        config = load_yaml(config_file)
        
        # Print key information
        print(f"Experiment Name: {config.get('experiment_name', 'N/A')}")
        
        # Model info
        if 'pretrained' in config:
            print(f"Model: {config['pretrained'].get('model_name_or_path', 'N/A')}")
            print(f"Quantization: {config['pretrained'].get('load_in_4bit', False)}")
        elif 'model' in config:
            print(f"Model: {config['model'].get('model_name', 'N/A')}")
        
        # Positional encoding
        if 'model' in config and 'pos_encoding' in config['model']:
            pos = config['model']['pos_encoding']
            print(f"Pos Encoding: {pos.get('type', 'N/A')}")
            print(f"Irrational: {pos.get('irrational', 'N/A')}")
            print(f"Hybrid K: {pos.get('hybrid_K', 'N/A')}")
        
        # Attractor
        if 'model' in config and 'attractor' in config['model']:
            att = config['model']['attractor']
            print(f"Use Attractor: {att.get('use_attractor', False)}")
            print(f"Alpha Policy: {att.get('alpha_policy', 'N/A')}")
        
        # Evaluation
        if 'evaluation' in config:
            eval_cfg = config['evaluation']
            print(f"Benchmarks: {eval_cfg.get('benchmarks', [])}")
            if 'niah_lengths' in eval_cfg:
                lengths = eval_cfg['niah_lengths']
                print(f"NIAH Lengths: {lengths}")
        
        # Ablation mode
        if 'ablation_mode' in config and config['ablation_mode']:
            print(f"Ablation Mode: {config['ablation_mode']}")
            print(f"Ablation Values: {config.get('ablation_values', [])}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT PHASE COVERAGE ANALYSIS")
    print("=" * 80)
    
    phases = {
        'Phase 0': 'Setup - base_config.yaml',
        'Phase 1': 'Module Development - Covered in code',
        'Phase 2': 'Sanity Runs (Qwen 1.5B) - qwen_1.5b_pi_spiral.yaml',
        'Phase 3': 'Core Benchmarks (Qwen 1.5B) - qwen_1.5b_pi_spiral.yaml',
        'Phase 4': 'Llama-3-8B - llama_8b_pi_spiral.yaml',
        'Phase 5': 'Llama-3-34B Demo - llama_34b_pi_spiral.yaml',
        'Phase 6': 'Ablations - ablation_irrational.yaml',
        'Phase 7': 'Diagnostics - Covered in diagnostic tools',
        'Phase 8': 'Cost and Reliability - Covered in profiling',
        'Phase 9': 'Results Packaging - Need documentation',
        'Phase 10': 'Decision Tree - Need documentation',
    }
    
    for phase, coverage in phases.items():
        print(f"{phase}: {coverage}")
    
    print("\n" + "=" * 80)
    print("MISSING CONFIGURATIONS")
    print("=" * 80)
    
    missing = []
    
    # Check for specific ablation configs
    config_names = [f.stem for f in config_files]
    
    if 'ablation_attractor' not in config_names:
        missing.append("ablation_attractor.yaml - Attractor on/off comparison")
    
    if 'ablation_alpha_policy' not in config_names:
        missing.append("ablation_alpha_policy.yaml - Alpha policy comparison")
    
    if 'ablation_hybrid_k' not in config_names:
        missing.append("ablation_hybrid_k.yaml - Hybrid threshold sweep")
    
    if 'phase2_sanity.yaml' not in config_names:
        missing.append("phase2_sanity.yaml - Dedicated Phase 2 sanity runs config")
    
    if 'baseline_rope_ntk' not in config_names:
        missing.append("baseline_rope_ntk.yaml - RoPE-NTK baseline")
    
    if 'baseline_alibi' not in config_names:
        missing.append("baseline_alibi.yaml - ALiBi baseline")
    
    for item in missing:
        print(f"- {item}")
    
    if not missing:
        print("All essential configurations are present!")

if __name__ == "__main__":
    review_configs()
