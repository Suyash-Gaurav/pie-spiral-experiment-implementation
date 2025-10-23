#!/usr/bin/env python3
"""
Simple Training Example for π-Spiral Positional Encoding

This example demonstrates how to train a model with π-Spiral encoding
on a simple task. It's designed to be self-contained and easy to understand.

Usage:
    python examples/example_train_simple.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoTokenizer
from src.config import ExperimentConfig
from src.models import inject_pi_spiral_encoding
from src.training import Trainer
from src.data_utils import create_simple_dataset

def main():
    print("=" * 80)
    print("Simple Training Example - π-Spiral Positional Encoding")
    print("=" * 80)
    
    # 1. Load configuration
    print("\n[1/6] Loading configuration...")
    config = ExperimentConfig.from_yaml('configs/base_config.yaml')
    
    # Modify for quick training
    config.training.num_epochs = 1
    config.training.max_steps = 100
    config.training.batch_size = 1
    config.training.logging_steps = 10
    config.data.max_seq_length = 4096  # Short sequences for quick training
    
    print(f"   Experiment: {config.experiment_name}")
    print(f"   Max steps: {config.training.max_steps}")
    print(f"   Batch size: {config.training.batch_size}")
    
    # 2. Load tokenizer
    print("\n[2/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",  # Using GPT-2 tokenizer for simplicity
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"   Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"   Vocab size: {len(tokenizer)}")
    
    # 3. Create simple dataset
    print("\n[3/6] Creating training dataset...")
    train_dataset = create_simple_dataset(
        num_samples=50,
        seq_length=512,
        vocab_size=len(tokenizer),
        task='language_modeling'
    )
    print(f"   Dataset size: {len(train_dataset)} samples")
    print(f"   Sequence length: 512 tokens")
    
    # 4. Initialize model with π-Spiral encoding
    print("\n[4/6] Initializing model with π-Spiral encoding...")
    from src.models import PiSpiralTransformer
    
    model = PiSpiralTransformer(
        vocab_size=len(tokenizer),
        d_model=256,  # Small model for quick training
        num_layers=4,
        num_heads=4,
        d_ff=1024,
        pos_encoding='hybrid',  # Use hybrid for best results
        use_attractor=True,
        hybrid_K=2048,
        irrational='pi',
        dropout=0.1,
    )
    
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Positional encoding: hybrid (RoPE + π-Spiral)")
    print(f"   Attractor state: enabled")
    
    # 5. Setup trainer
    print("\n[5/6] Setting up trainer...")
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    print("   Trainer initialized")
    
    # 6. Train
    print("\n[6/6] Starting training...")
    print("-" * 80)
    
    try:
        trainer.train()
        print("\n" + "=" * 80)
        print("Training completed successfully!")
        print("=" * 80)
        
        # Print final metrics
        if hasattr(trainer, 'state') and trainer.state.log_history:
            final_loss = trainer.state.log_history[-1].get('loss', 'N/A')
            print(f"\nFinal training loss: {final_loss}")
        
        # Save model
        output_dir = config.output_dir or './results/simple_training'
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, 'final_model')
        trainer.save_model(model_path)
        print(f"\nModel saved to: {model_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        raise
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Check results in:", config.output_dir or './results/simple_training')
    print("  2. View logs in:", config.logging_dir or './logs')
    print("  3. Try evaluation: python examples/example_evaluate_niah.py")
    print("  4. Run diagnostics: python examples/example_diagnostic.py")

if __name__ == "__main__":
    main()
