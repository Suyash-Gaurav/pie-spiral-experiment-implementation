#!/usr/bin/env python3
"""
Training Script for π-Spiral Positional Encoding Experiment

Trains models with different positional encoding schemes.
Supports:
- Custom models from scratch
- Pre-trained model adaptation (Qwen, Llama)
- Multiple encoding types (RoPE, π-Spiral, Hybrid)
- Distributed training
- Mixed precision training
- Comprehensive logging and checkpointing

Usage:
    python train.py --config configs/qwen_1.5b_pi_spiral.yaml
    python train.py --model-type pi_spiral --encoding pi_spiral --output-dir results/pi_spiral
"""

import argparse
import sys
import logging
from pathlib import Path
import torch
import torch.distributed as dist

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ExperimentConfig
from training import Trainer, setup_logging
from data_utils import NIAHDataset
from models.transformer import PiSpiralTransformer
from models.adapters import adapt_pretrained_model

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train π-Spiral models')
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--experiment-name', type=str, default='pi_spiral_experiment', help='Experiment name')
    
    # Model configuration
    parser.add_argument('--model-type', type=str, choices=['custom', 'pretrained'], default='custom',
                       help='Model type: custom or pretrained')
    parser.add_argument('--pretrained-model', type=str, help='Pretrained model name or path')
    parser.add_argument('--encoding', type=str, choices=['rope', 'pi_spiral', 'hybrid'], default='pi_spiral',
                       help='Positional encoding type')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=8, help='Gradient accumulation steps')
    parser.add_argument('--num-epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--max-steps', type=int, help='Maximum training steps')
    
    # Data configuration
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--max-seq-length', type=int, default=100000, help='Maximum sequence length')
    
    # System configuration
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 mixed precision')
    parser.add_argument('--bf16', action='store_true', help='Use BF16 mixed precision')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Use gradient checkpointing')
    
    # Logging
    parser.add_argument('--logging-steps', type=int, default=10, help='Logging frequency')
    parser.add_argument('--eval-steps', type=int, default=100, help='Evaluation frequency')
    parser.add_argument('--save-steps', type=int, default=500, help='Checkpoint save frequency')
    
    # Distributed training
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    
    # Resume training
    parser.add_argument('--resume-from', type=str, help='Resume from checkpoint')
    
    # Seed
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def setup_distributed():
    """Setup distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    
    return 0, 1, 0


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(args):
    """Load configuration from file or create from args"""
    if args.config:
        config = ExperimentConfig.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        # Create config from args
        from config import (
            ModelConfig, TrainingConfig, DataConfig, SystemConfig,
            PositionalEncodingConfig, AttractorConfig
        )
        
        pos_encoding = PositionalEncodingConfig(type=args.encoding)
        attractor = AttractorConfig(use_attractor=(args.encoding == 'pi_spiral'))
        
        model_config = ModelConfig(
            pos_encoding=pos_encoding,
            attractor=attractor,
        )
        
        training_config = TrainingConfig(
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            max_steps=args.max_steps,
            fp16=args.fp16,
            bf16=args.bf16,
            logging_steps=args.logging_steps,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            seed=args.seed,
        )
        
        data_config = DataConfig(
            data_dir=args.data_dir,
            max_seq_length=args.max_seq_length,
        )
        
        system_config = SystemConfig(
            device=args.device,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        
        config = ExperimentConfig(
            experiment_name=args.experiment_name,
            output_dir=args.output_dir,
            model=model_config,
            training=training_config,
            data=data_config,
            system=system_config,
        )
        
        logger.info("Created config from command line arguments")
    
    return config


def create_model(config, device):
    """Create or load model"""
    if config.pretrained and config.pretrained.model_name_or_path:
        logger.info(f"Loading pretrained model: {config.pretrained.model_name_or_path}")
        model = adapt_pretrained_model(config.pretrained)
    else:
        logger.info("Creating custom model from scratch")
        model = PiSpiralTransformer(config.model)
    
    model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model


def create_dataloaders(config):
    """Create training and evaluation dataloaders"""
    from torch.utils.data import DataLoader
    
    # For now, create dummy dataloaders
    # In practice, load actual datasets
    logger.info("Creating dataloaders...")
    
    # Create NIAH dataset as example
    train_dataset = NIAHDataset(
        num_samples=1000,
        context_lengths=config.data.niah_lengths[:2],  # Use shorter lengths for training
        depths=config.data.niah_depths,
        seed=config.training.seed,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.dataloader_num_workers,
        pin_memory=config.data.dataloader_pin_memory,
    )
    
    # Eval dataloader (optional)
    eval_dataloader = None
    
    logger.info(f"Created train dataloader with {len(train_dataset)} samples")
    
    return train_dataloader, eval_dataloader


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    is_main_process = rank == 0
    
    # Setup logging
    if is_main_process:
        exp_logger = setup_logging(
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            log_level='INFO',
        )
        exp_logger.info("Starting training...")
        exp_logger.info(f"Arguments: {args}")
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args)
    
    # Save config
    if is_main_process:
        config_path = Path(args.output_dir) / 'config.yaml'
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config.to_yaml(str(config_path))
        logger.info(f"Saved config to {config_path}")
    
    # Create model
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = create_model(config, device)
    
    # Create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config,
        device=device,
        output_dir=args.output_dir,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Train
    try:
        training_stats = trainer.train()
        
        if is_main_process:
            logger.info("Training completed successfully!")
            logger.info(f"Training statistics: {training_stats}")
            
            # Save final model
            final_model_path = Path(args.output_dir) / 'final_model'
            trainer.save_checkpoint('final_model')
            logger.info(f"Final model saved to {final_model_path}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if is_main_process:
            trainer.save_checkpoint('interrupted')
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    import os
    main()
