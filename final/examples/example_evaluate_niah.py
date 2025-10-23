#!/usr/bin/env python3
"""
NIAH (Needle in a Haystack) Evaluation Example

This example demonstrates how to evaluate a model on the NIAH benchmark,
which tests the ability to retrieve specific information from long contexts.

Usage:
    python examples/example_evaluate_niah.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.config import ExperimentConfig
from src.models.adapters import inject_pi_spiral_encoding
from src.data_utils import NIAHDataset, evaluate_niah
import numpy as np

def main():
    print("=" * 80)
    print("NIAH Evaluation Example - π-Spiral Positional Encoding")
    print("=" * 80)
    
    # 1. Configuration
    print("\n[1/5] Setting up configuration...")
    config = ExperimentConfig.from_yaml('configs/qwen_1.5b_pi_spiral.yaml')
    
    # Use smaller lengths for quick demo
    niah_lengths = [4000, 8000, 16000]
    niah_depths = [0.1, 0.5, 0.9]
    num_samples = 5  # Small number for demo
    
    print(f"   Context lengths: {niah_lengths}")
    print(f"   Depths: {niah_depths}")
    print(f"   Samples per config: {num_samples}")
    
    # 2. Load model and tokenizer
    print("\n[2/5] Loading model and tokenizer...")
    model_name = "Qwen/Qwen2.5-1.5B"
    
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
    
    print(f"   Model: {model_name}")
    print(f"   Device: {model.device}")
    
    # 3. Inject π-Spiral encoding
    print("\n[3/5] Injecting π-Spiral encoding...")
    adapter = inject_pi_spiral_encoding(
        model,
        pos_encoding='hybrid',
        inject_layers='last_N',
        N_inject=4,
        use_attractor=True,
        hybrid_K=16000,
        irrational='pi',
    )
    
    print(f"   Encoding type: hybrid")
    print(f"   Hybrid K: 16000")
    print(f"   Attractor: enabled")
    print(f"   Injected layers: last 4")
    
    # 4. Create NIAH dataset
    print("\n[4/5] Creating NIAH dataset...")
    dataset = NIAHDataset(
        num_samples=num_samples,
        context_lengths=niah_lengths,
        depths=niah_depths,
        tokenizer=tokenizer,
        seed=42
    )
    
    print(f"   Total samples: {len(dataset)}")
    print(f"   Configurations: {len(niah_lengths)} lengths × {len(niah_depths)} depths")
    
    # Show example
    sample = dataset[0]
    print(f"\n   Example sample:")
    print(f"     Context length: {sample['context_length']} tokens")
    print(f"     Depth: {sample['depth']:.1%}")
    print(f"     Question: {sample['question'][:80]}...")
    print(f"     Answer: {sample['answer']}")
    
    # 5. Evaluate
    print("\n[5/5] Running evaluation...")
    print("-" * 80)
    
    results = []
    
    for i, sample in enumerate(dataset):
        print(f"\nSample {i+1}/{len(dataset)}")
        print(f"  Length: {sample['context_length']}, Depth: {sample['depth']:.1%}")
        
        # Prepare input
        prompt = sample['context'] + "\n\n" + sample['question']
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=sample['context_length'])
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,  # Low temperature for deterministic output
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Check if answer is in generated text
        correct = sample['answer'].lower() in generated_text.lower()
        
        results.append({
            'length': sample['context_length'],
            'depth': sample['depth'],
            'correct': correct,
            'generated': generated_text[:100]
        })
        
        print(f"  Generated: {generated_text[:80]}...")
        print(f"  Expected: {sample['answer']}")
        print(f"  Correct: {'✓' if correct else '✗'}")
    
    # 6. Summarize results
    print("\n" + "=" * 80)
    print("Evaluation Results Summary")
    print("=" * 80)
    
    # Overall accuracy
    overall_acc = np.mean([r['correct'] for r in results])
    print(f"\nOverall Accuracy: {overall_acc:.1%}")
    
    # By length
    print("\nAccuracy by Context Length:")
    for length in niah_lengths:
        length_results = [r for r in results if r['length'] == length]
        if length_results:
            acc = np.mean([r['correct'] for r in length_results])
            print(f"  {length:>6} tokens: {acc:.1%} ({sum(r['correct'] for r in length_results)}/{len(length_results)})")
    
    # By depth
    print("\nAccuracy by Depth:")
    for depth in niah_depths:
        depth_results = [r for r in results if r['depth'] == depth]
        if depth_results:
            acc = np.mean([r['correct'] for r in depth_results])
            print(f"  {depth:.1%}: {acc:.1%} ({sum(r['correct'] for r in depth_results)}/{len(depth_results)})")
    
    # Save results
    output_dir = './results/niah_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    import json
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_dir}/results.json")
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Try longer contexts: modify niah_lengths")
    print("  2. Compare with baseline: use baseline_rope.yaml")
    print("  3. Run full evaluation: python scripts/evaluate.py")
    print("  4. Visualize results: python examples/example_diagnostic.py")

if __name__ == "__main__":
    main()
