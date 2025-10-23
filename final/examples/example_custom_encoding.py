#!/usr/bin/env python3
"""
Custom Encoding Configuration Example

This example demonstrates how to create and use custom positional encoding
configurations, including different irrational constants and hybrid settings.

Usage:
    python examples/example_custom_encoding.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.encodings import PiSpiralPositional, HybridPositionalEncoding, AttractorState
from src.config import ExperimentConfig, PositionalEncodingConfig, AttractorConfig

def main():
    print("=" * 80)
    print("Custom Encoding Configuration Example")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    d_model = 512
    
    # Example 1: Different Irrational Constants
    print("\n" + "=" * 80)
    print("[1/5] Testing Different Irrational Constants")
    print("=" * 80)
    
    irrational_constants = {
        'pi': 3.14159265359,
        'e': 2.71828182846,
        'sqrt2': 1.41421356237,
        'phi': 1.61803398875,  # Golden ratio
    }
    
    print("\nComparing positional encodings with different irrational constants:")
    print(f"{'Constant':<10} {'Value':<15} {'Encoding Sample':<30}")
    print("-" * 60)
    
    for name, value in irrational_constants.items():
        encoder = PiSpiralPositional(
            d_model=d_model,
            max_len=100000,
            irrational=name
        )
        
        # Get encoding for position 1000
        pos = torch.tensor([1000])
        encoding = encoder.get_encoding(pos)
        
        # Show first 3 values
        sample = encoding[0, :3].tolist()
        sample_str = f"[{sample[0]:.3f}, {sample[1]:.3f}, {sample[2]:.3f}]"
        
        print(f"{name:<10} {value:<15.11f} {sample_str:<30}")
    
    print("\nNote: Different constants produce different non-periodic patterns.")
    print("Experiment with ablation_irrational.yaml to compare performance.")
    
    # Example 2: Hybrid Encoding with Different K Values
    print("\n" + "=" * 80)
    print("[2/5] Hybrid Encoding with Different K Values")
    print("=" * 80)
    
    k_values = [4000, 8000, 16000, 32000]
    
    print("\nTesting hybrid transition at different K values:")
    print(f"{'K Value':<10} {'Blend at K-1000':<20} {'Blend at K':<20} {'Blend at K+1000':<20}")
    print("-" * 70)
    
    for K in k_values:
        hybrid = HybridPositionalEncoding(
            d_model=d_model,
            hybrid_K=K,
            transition_width=1000,
            irrational='pi'
        )
        
        # Test blend weights around K
        pos_before = torch.tensor([K - 1000])
        pos_at = torch.tensor([K])
        pos_after = torch.tensor([K + 1000])
        
        weight_before = hybrid.get_blend_weight(pos_before.item())
        weight_at = hybrid.get_blend_weight(pos_at.item())
        weight_after = hybrid.get_blend_weight(pos_after.item())
        
        print(f"{K:<10} {weight_before:<20.3f} {weight_at:<20.3f} {weight_after:<20.3f}")
    
    print("\nNote: Blend weight 0.0 = pure RoPE, 1.0 = pure π-Spiral")
    print("Smooth transition ensures no discontinuities in positional information.")
    
    # Example 3: Attractor State with Different Alpha Policies
    print("\n" + "=" * 80)
    print("[3/5] Attractor State with Different Alpha Policies")
    print("=" * 80)
    
    print("\nComparing alpha decay policies:")
    
    # Fixed alpha
    print("\n1. Fixed Alpha (α = 0.99):")
    attractor_fixed = AttractorState(
        d_model=d_model,
        alpha_policy='fixed',
        alpha_value=0.99
    )
    print(f"   Alpha value: {attractor_fixed.get_alpha(N=10000):.6f}")
    print(f"   Same for any sequence length")
    
    # Exponential decay with c/N
    print("\n2. Exponential Decay (α = exp(-π/N)):")
    attractor_exp = AttractorState(
        d_model=d_model,
        alpha_policy='exp_c_over_N',
        c_value=3.14159
    )
    for N in [1000, 10000, 100000]:
        alpha = attractor_exp.get_alpha(N=N)
        print(f"   N={N:>6}: α = {alpha:.6f}")
    
    # Learned alpha
    print("\n3. Learned Alpha:")
    attractor_learned = AttractorState(
        d_model=d_model,
        alpha_policy='learned'
    )
    print(f"   Initial alpha: {attractor_learned.alpha.item():.6f}")
    print(f"   Will be optimized during training")
    
    print("\nNote: Different policies adapt differently to sequence length.")
    print("Use ablation_alpha_policy.yaml to compare performance.")
    
    # Example 4: Creating Custom Configuration
    print("\n" + "=" * 80)
    print("[4/5] Creating Custom Configuration")
    print("=" * 80)
    
    print("\nCreating a custom configuration programmatically:")
    
    # Create custom config
    custom_config = ExperimentConfig(
        experiment_name='my_custom_experiment',
        output_dir='./results/custom',
        logging_dir='./logs/custom',
    )
    
    # Configure positional encoding
    custom_config.model.pos_encoding = PositionalEncodingConfig(
        type='hybrid',
        irrational='phi',  # Use golden ratio
        hybrid_K=12000,    # Custom transition point
        transition_width=2000,
        max_seq_len=200000,
    )
    
    # Configure attractor
    custom_config.model.attractor = AttractorConfig(
        use_attractor=True,
        alpha_policy='exp_c_over_N',
        c_value=2.71828,  # Use e instead of π
        inject_layers='last_N',
        N_inject=6,  # Inject in last 6 layers
        attractor_inject='cross_attn',
    )
    
    print("\nCustom Configuration:")
    print(f"  Experiment: {custom_config.experiment_name}")
    print(f"  Encoding: {custom_config.model.pos_encoding.type}")
    print(f"  Irrational: {custom_config.model.pos_encoding.irrational}")
    print(f"  Hybrid K: {custom_config.model.pos_encoding.hybrid_K}")
    print(f"  Attractor: {custom_config.model.attractor.use_attractor}")
    print(f"  Alpha policy: {custom_config.model.attractor.alpha_policy}")
    print(f"  Inject layers: {custom_config.model.attractor.N_inject}")
    
    # Save configuration
    output_dir = './configs'
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'my_custom_config.yaml')
    custom_config.to_yaml(config_path)
    
    print(f"\nConfiguration saved to: {config_path}")
    
    # Example 5: Using Custom Encoding in Model
    print("\n" + "=" * 80)
    print("[5/5] Using Custom Encoding in Model")
    print("=" * 80)
    
    print("\nInitializing model with custom encoding:")
    
    from src.models import PiSpiralTransformer
    
    model = PiSpiralTransformer(
        vocab_size=50000,
        d_model=512,
        num_layers=6,
        num_heads=8,
        pos_encoding='hybrid',
        use_attractor=True,
        hybrid_K=12000,
        irrational='phi',
        alpha_policy='exp_c_over_N',
        c_value=2.71828,
        inject_layers='last_N',
        N_inject=6,
    )
    
    print(f"  Model: {model.__class__.__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Positional encoding: hybrid (φ-based)")
    print(f"  Attractor: enabled with e-based decay")
    
    # Test forward pass
    print("\nTesting forward pass:")
    batch_size = 2
    seq_len = 1000
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {outputs.shape}")
    print(f"  Success! ✓")
    
    # Summary
    print("\n" + "=" * 80)
    print("Custom Encoding Configuration Complete!")
    print("=" * 80)
    
    print("\nKey Takeaways:")
    print("  1. Different irrational constants produce different non-periodic patterns")
    print("  2. Hybrid K controls the transition point from RoPE to π-Spiral")
    print("  3. Alpha policies control attractor decay behavior")
    print("  4. Configurations can be created programmatically or via YAML")
    print("  5. Custom encodings integrate seamlessly with the model")
    
    print("\nNext steps:")
    print("  1. Experiment with different irrational constants")
    print("  2. Tune hybrid K for your use case")
    print("  3. Compare alpha policies on your data")
    print("  4. Run ablation studies to find optimal settings")
    print(f"  5. Use your custom config: python scripts/train.py --config {config_path}")

if __name__ == "__main__":
    main()
