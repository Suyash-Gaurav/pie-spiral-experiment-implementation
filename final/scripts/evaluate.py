#!/usr/bin/env python3
"""
Evaluation Script for π-Spiral Positional Encoding Experiment

Evaluates models on long-context benchmarks:
- NIAH (Needle in a Haystack)
- RULER
- InfiniteBench

Usage:
    python evaluate.py --model-path results/checkpoint-best --benchmark niah
    python evaluate.py --config configs/eval_config.yaml --benchmark all
    python evaluate.py --model-path results/final_model --benchmark ruler --output-dir results/eval
"""

import argparse
import sys
import logging
from pathlib import Path
import torch
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'tests'))

from config import ExperimentConfig
from models.transformer import PiSpiralTransformer
from models.adapters import adapt_pretrained_model
from test_niah import run_niah_benchmark
from test_ruler import run_ruler_benchmark
from test_infinitebench import run_infinitebench

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate π-Spiral models')
    
    # Model configuration
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--model-type', type=str, choices=['custom', 'pretrained'], default='custom',
                       help='Model type')
    
    # Benchmark selection
    parser.add_argument('--benchmark', type=str, required=True,
                       choices=['niah', 'ruler', 'infinitebench', 'all'],
                       help='Benchmark to run')
    
    # Benchmark-specific options
    parser.add_argument('--niah-lengths', type=int, nargs='+',
                       default=[32000, 64000, 128000, 256000, 512000, 1000000],
                       help='Context lengths for NIAH')
    parser.add_argument('--niah-depths', type=float, nargs='+',
                       default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                       help='Depths for NIAH')
    parser.add_argument('--ruler-lengths', type=int, nargs='+',
                       default=[32000, 64000, 128000, 256000],
                       help='Context lengths for RULER')
    parser.add_argument('--infinitebench-lengths', type=int, nargs='+',
                       default=[100000, 256000, 512000],
                       help='Context lengths for InfiniteBench')
    
    # Evaluation configuration
    parser.add_argument('--samples-per-config', type=int, default=10,
                       help='Number of samples per configuration')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--max-new-tokens', type=int, default=50, help='Max tokens to generate')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='./results/evaluation',
                       help='Output directory for results')
    
    # System configuration
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--fp16', action='store_true', help='Use FP16')
    parser.add_argument('--bf16', action='store_true', help='Use BF16')
    
    return parser.parse_args()


def load_model_and_tokenizer(args):
    """Load model and tokenizer from checkpoint"""
    model_path = Path(args.model_path)
    
    # Load config if available
    config_path = model_path.parent / 'config.yaml'
    if config_path.exists():
        config = ExperimentConfig.from_yaml(str(config_path))
        logger.info(f"Loaded config from {config_path}")
    elif args.config:
        config = ExperimentConfig.from_yaml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        logger.warning("No config found, using defaults")
        config = ExperimentConfig()
    
    # Create model
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    if config.pretrained and config.pretrained.model_name_or_path:
        logger.info(f"Loading pretrained model: {config.pretrained.model_name_or_path}")
        model = adapt_pretrained_model(config.pretrained)
        
        # Load tokenizer
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.pretrained.model_name_or_path)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            tokenizer = None
    else:
        logger.info("Loading custom model")
        model = PiSpiralTransformer(config.model)
        tokenizer = None  # Custom tokenizer needed
    
    # Load checkpoint weights
    checkpoint_file = model_path / 'model.pt' if model_path.is_dir() else model_path
    if checkpoint_file.exists():
        logger.info(f"Loading model weights from {checkpoint_file}")
        state_dict = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(state_dict)
    else:
        logger.warning(f"No checkpoint found at {checkpoint_file}, using initialized weights")
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded on {device}")
    
    return model, tokenizer, config


def run_niah_evaluation(model, tokenizer, args, config):
    """Run NIAH benchmark"""
    logger.info("Running NIAH benchmark...")
    
    output_dir = Path(args.output_dir) / 'niah'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config with command line args
    if hasattr(config, 'evaluation'):
        config.evaluation.niah_lengths = args.niah_lengths
        config.evaluation.niah_depths = args.niah_depths
    
    results = run_niah_benchmark(
        model=model,
        tokenizer=tokenizer,
        config=config,
        output_dir=str(output_dir),
    )
    
    logger.info(f"NIAH evaluation completed. Overall accuracy: {results['summary']['overall_accuracy']:.2%}")
    
    return results


def run_ruler_evaluation(model, tokenizer, args, config):
    """Run RULER benchmark"""
    logger.info("Running RULER benchmark...")
    
    output_dir = Path(args.output_dir) / 'ruler'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update config with command line args
    if hasattr(config, 'evaluation'):
        config.evaluation.ruler_max_length = max(args.ruler_lengths)
    
    results = run_ruler_benchmark(
        model=model,
        tokenizer=tokenizer,
        config=config,
        output_dir=str(output_dir),
    )
    
    logger.info(f"RULER evaluation completed. Overall accuracy: {results['summary']['overall']:.2%}")
    
    return results


def run_infinitebench_evaluation(model, tokenizer, args, config):
    """Run InfiniteBench"""
    logger.info("Running InfiniteBench...")
    
    output_dir = Path(args.output_dir) / 'infinitebench'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_infinitebench(
        model=model,
        tokenizer=tokenizer,
        config=config,
        output_dir=str(output_dir),
    )
    
    if 'summary' in results and 'overall' in results['summary']:
        logger.info(f"InfiniteBench evaluation completed. Overall accuracy: {results['summary']['overall']:.2%}")
    else:
        logger.info("InfiniteBench evaluation completed.")
    
    return results


def main():
    """Main evaluation function"""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    logger.info("Starting evaluation...")
    logger.info(f"Arguments: {args}")
    
    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(args)
    
    if tokenizer is None:
        logger.error("No tokenizer available. Cannot run evaluation.")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks
    all_results = {}
    
    if args.benchmark in ['niah', 'all']:
        try:
            niah_results = run_niah_evaluation(model, tokenizer, args, config)
            all_results['niah'] = niah_results
        except Exception as e:
            logger.error(f"NIAH evaluation failed: {e}", exc_info=True)
    
    if args.benchmark in ['ruler', 'all']:
        try:
            ruler_results = run_ruler_evaluation(model, tokenizer, args, config)
            all_results['ruler'] = ruler_results
        except Exception as e:
            logger.error(f"RULER evaluation failed: {e}", exc_info=True)
    
    if args.benchmark in ['infinitebench', 'all']:
        try:
            infinitebench_results = run_infinitebench_evaluation(model, tokenizer, args, config)
            all_results['infinitebench'] = infinitebench_results
        except Exception as e:
            logger.error(f"InfiniteBench evaluation failed: {e}", exc_info=True)
    
    # Save combined results
    combined_results_file = output_dir / 'evaluation_results.json'
    with open(combined_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"All results saved to {combined_results_file}")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    
    for benchmark_name, results in all_results.items():
        logger.info(f"\n{benchmark_name.upper()}:")
        if 'summary' in results:
            if 'overall_accuracy' in results['summary']:
                logger.info(f"  Overall Accuracy: {results['summary']['overall_accuracy']:.2%}")
            elif 'overall' in results['summary']:
                logger.info(f"  Overall Score: {results['summary']['overall']:.2%}")
            
            # Print per-task or per-length breakdown
            for key, value in results['summary'].items():
                if key not in ['overall_accuracy', 'overall']:
                    if isinstance(value, dict):
                        logger.info(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            logger.info(f"    {sub_key}: {sub_value:.2%}" if isinstance(sub_value, float) else f"    {sub_key}: {sub_value}")
                    else:
                        logger.info(f"  {key}: {value:.2%}" if isinstance(value, float) else f"  {key}: {value}")
    
    logger.info("\n" + "="*50)
    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()
