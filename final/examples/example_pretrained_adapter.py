#!/usr/bin/env python3
"""
Pre-trained Model Adapter Example

This example demonstrates how to adapt pre-trained models (Qwen, Llama, etc.)
with π-Spiral positional encoding using the adapter interface.

Usage:
    python examples/example_pretrained_adapter.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.adapters import inject_pi_spiral_encoding, PiSpiralAdapter

def main():
    print("=" * 80)
    print("Pre-trained Model Adapter Example")
    print("=" * 80)
    
    # Configuration
    model_name = "Qwen/Qwen2.5-1.5B"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    
    # Example 1: Basic Adapter Injection
    print("\n" + "=" * 80)
    print("[1/4] Basic Adapter Injection")
    print("=" * 80)
    
    print("\nLoading pre-trained model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"  Model loaded: {model.__class__.__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nInjecting π-Spiral encoding...")
    adapter = inject_pi_spiral_encoding(
        model,
        pos_encoding='hybrid',
        inject_layers='last_N',
        N_inject=4,
        use_attractor=True,
        hybrid_K=16000,
        irrational='pi',
    )
    
    print(f"  Adapter injected: {adapter.__class__.__name__}")
    print(f"  Encoding type: hybrid")
    print(f"  Attractor: enabled")
    print(f"  Injected layers: last 4")
    
    # Example 2: Testing with Short Context
    print("\n" + "=" * 80)
    print("[2/4] Testing with Short Context")
    print("=" * 80)
    
    print("\nGenerating text with short context (should use RoPE)...")
    
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print(f"  Prompt: '{prompt}'")
    print(f"  Input length: {inputs['input_ids'].shape[1]} tokens")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Generated: '{generated_text}'")
    print(f"  Note: Short context uses RoPE (position < K=16000)")
    
    # Example 3: Testing with Long Context
    print("\n" + "=" * 80)
    print("[3/4] Testing with Long Context")
    print("=" * 80)
    
    print("\nGenerating text with long context (should use π-Spiral)...")
    
    # Create a long context by repeating text
    long_context = "This is a test sentence. " * 1000  # ~5000 tokens
    long_prompt = long_context + "\n\nQuestion: What is this text about?\nAnswer:"
    
    inputs = tokenizer(long_prompt, return_tensors="pt", truncation=True, max_length=20000)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print(f"  Context length: {inputs['input_ids'].shape[1]} tokens")
    print(f"  Note: Long context uses hybrid (RoPE + π-Spiral)")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    print(f"  Generated: '{generated_text[:100]}...'")
    
    # Example 4: Different Adapter Configurations
    print("\n" + "=" * 80)
    print("[4/4] Different Adapter Configurations")
    print("=" * 80)
    
    print("\nComparing different adapter configurations:")
    
    configs = [
        {
            'name': 'Pure π-Spiral',
            'pos_encoding': 'pi_spiral',
            'use_attractor': True,
            'irrational': 'pi',
        },
        {
            'name': 'Hybrid (K=8000)',
            'pos_encoding': 'hybrid',
            'hybrid_K': 8000,
            'use_attractor': True,
            'irrational': 'pi',
        },
        {
            'name': 'Hybrid (K=16000)',
            'pos_encoding': 'hybrid',
            'hybrid_K': 16000,
            'use_attractor': True,
            'irrational': 'pi',
        },
        {
            'name': 'π-Spiral without Attractor',
            'pos_encoding': 'pi_spiral',
            'use_attractor': False,
            'irrational': 'pi',
        },
        {
            'name': 'Golden Ratio (φ)',
            'pos_encoding': 'hybrid',
            'hybrid_K': 16000,
            'use_attractor': True,
            'irrational': 'phi',
        },
    ]
    
    print(f"\n{'Configuration':<30} {'Encoding':<15} {'Attractor':<12} {'Special':<20}")
    print("-" * 80)
    
    for config in configs:
        encoding = config['pos_encoding']
        attractor = 'Yes' if config.get('use_attractor', False) else 'No'
        
        special = []
        if 'hybrid_K' in config:
            special.append(f"K={config['hybrid_K']}")
        if config.get('irrational') != 'pi':
            special.append(f"irrational={config['irrational']}")
        special_str = ', '.join(special) if special else '-'
        
        print(f"{config['name']:<30} {encoding:<15} {attractor:<12} {special_str:<20}")
    
    print("\nNote: Each configuration can be tested by modifying the inject_pi_spiral_encoding call.")
    
    # Example: Switching configuration
    print("\nExample: Switching to Golden Ratio (φ) encoding...")
    
    # Reload model (in practice, you'd use a fresh model)
    print("  (In practice, reload the model to switch configurations)")
    
    # Show how to inject with different config
    print("\n  Code example:")
    print("  ```python")
    print("  adapter = inject_pi_spiral_encoding(")
    print("      model,")
    print("      pos_encoding='hybrid',")
    print("      hybrid_K=16000,")
    print("      use_attractor=True,")
    print("      irrational='phi',  # Use golden ratio")
    print("      inject_layers='last_N',")
    print("      N_inject=4,")
    print("  )")
    print("  ```")
    
    # Summary
    print("\n" + "=" * 80)
    print("Pre-trained Model Adapter Complete!")
    print("=" * 80)
    
    print("\nKey Features:")
    print("  1. Easy integration with Hugging Face models")
    print("  2. Minimal code changes required")
    print("  3. Flexible configuration options")
    print("  4. Automatic handling of short and long contexts")
    print("  5. Compatible with generation, fine-tuning, and evaluation")
    
    print("\nAdapter Benefits:")
    print("  ✓ Preserves pre-trained weights")
    print("  ✓ Adds π-Spiral encoding without retraining")
    print("  ✓ Hybrid mode ensures no short-range regression")
    print("  ✓ Attractor provides O(1) global context")
    print("  ✓ Works with quantization (4-bit, 8-bit)")
    
    print("\nSupported Models:")
    print("  - Qwen/Qwen2.5 series")
    print("  - meta-llama/Llama-3 series")
    print("  - Any transformer with compatible architecture")
    
    print("\nNext steps:")
    print("  1. Try different models: change model_name")
    print("  2. Experiment with configurations: modify inject_pi_spiral_encoding params")
    print("  3. Evaluate on benchmarks: python scripts/evaluate.py")
    print("  4. Fine-tune adapted model: python scripts/train.py")
    print("  5. Compare with baselines: use baseline configs")
    
    print("\nUseful Commands:")
    print("  # Evaluate adapted model on NIAH")
    print("  python scripts/evaluate.py --config configs/qwen_1.5b_pi_spiral.yaml")
    print()
    print("  # Run diagnostics")
    print("  python scripts/diagnose.py --config configs/qwen_1.5b_pi_spiral.yaml")
    print()
    print("  # Compare with baseline")
    print("  python scripts/evaluate.py --config configs/baseline_rope.yaml")

if __name__ == "__main__":
    main()
