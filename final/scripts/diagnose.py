#!/usr/bin/env python3
"""
Diagnostic Script for π-Spiral Positional Encoding Experiment

Runs diagnostic analyses:
- Positional collision analysis
- Spectral analysis (FFT)
- Attention pattern visualization
- Model profiling
- Perplexity tracking

Usage:
    python diagnose.py --encoding pi_spiral --analysis collision
    python diagnose.py --model-path results/checkpoint-best --analysis all
    python diagnose.py --encoding-comparison rope,pi_spiral,hybrid --output-dir diagnostics
"""

import argparse
import sys
import logging
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import ExperimentConfig, PositionalEncodingConfig
from encodings.pi_spiral import PiSpiralPositional
from encodings.rope import RoPEPositional
from encodings.hybrid import HybridPositionalEncoding
from diagnostics import (
    EncodingAnalyzer,
    compare_multiple_encodings,
    AttentionVisualizer,
    DiagnosticVisualizer,
    ModelProfiler,
    PerplexityTracker,
)
from models.transformer import PiSpiralTransformer

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run diagnostic analyses')
    
    # Analysis selection
    parser.add_argument('--analysis', type=str, required=True,
                       choices=['collision', 'spectral', 'attention', 'profiling', 'perplexity', 'all'],
                       help='Type of analysis to run')
    
    # Encoding configuration
    parser.add_argument('--encoding', type=str, choices=['rope', 'pi_spiral', 'hybrid'],
                       help='Single encoding to analyze')
    parser.add_argument('--encoding-comparison', type=str,
                       help='Comma-separated list of encodings to compare (e.g., rope,pi_spiral,hybrid)')
    parser.add_argument('--irrational', type=str, choices=['pi', 'e', 'sqrt2', 'phi', 'prng'],
                       default='pi', help='Irrational constant for π-Spiral')
    
    # Model configuration
    parser.add_argument('--model-path', type=str, help='Path to model checkpoint for attention/profiling analysis')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    
    # Analysis parameters
    parser.add_argument('--num-positions', type=int, default=10000,
                       help='Number of positions to analyze')
    parser.add_argument('--max-length', type=int, default=100000,
                       help='Maximum sequence length')
    parser.add_argument('--d-model', type=int, default=512,
                       help='Model dimension')
    
    # Profiling parameters
    parser.add_argument('--profile-lengths', type=int, nargs='+',
                       default=[1000, 4000, 16000, 32000, 64000],
                       help='Lengths to profile')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='./diagnostics',
                       help='Output directory for diagnostic results')
    parser.add_argument('--save-plots', action='store_true', help='Generate and save plots')
    
    # System configuration
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    return parser.parse_args()


def create_encoding(encoding_type: str, d_model: int, irrational: str = 'pi'):
    """Create positional encoding module"""
    if encoding_type == 'rope':
        return RoPEPositional(d_model=d_model)
    elif encoding_type == 'pi_spiral':
        return PiSpiralPositional(d_model=d_model, irrational=irrational)
    elif encoding_type == 'hybrid':
        return HybridPositionalEncoding(d_model=d_model, hybrid_K=16000)
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


def run_collision_analysis(args):
    """Run positional collision analysis"""
    logger.info("Running positional collision analysis...")
    
    output_dir = Path(args.output_dir) / 'collision'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.encoding_comparison:
        # Compare multiple encodings
        encoding_names = args.encoding_comparison.split(',')
        encodings = {
            name: create_encoding(name, args.d_model, args.irrational)
            for name in encoding_names
        }
        
        results = compare_multiple_encodings(
            encodings=encodings,
            num_positions=args.num_positions,
            output_dir=str(output_dir),
        )
        
        # Generate comparison plots
        if args.save_plots:
            visualizer = DiagnosticVisualizer(output_dir=str(output_dir))
            visualizer.plot_encoding_comparison(results, metric='mean_similarity')
    
    else:
        # Analyze single encoding
        encoding = create_encoding(args.encoding, args.d_model, args.irrational)
        analyzer = EncodingAnalyzer(encoding, max_length=args.max_length, device=args.device)
        
        results = analyzer.generate_diagnostic_report(
            output_dir=str(output_dir),
            num_positions=args.num_positions,
        )
        
        # Generate plots
        if args.save_plots:
            visualizer = DiagnosticVisualizer(output_dir=str(output_dir))
            visualizer.plot_positional_collision(
                results['collision_analysis'],
                encoding_name=args.encoding,
            )
    
    logger.info(f"Collision analysis completed. Results saved to {output_dir}")
    return results


def run_spectral_analysis(args):
    """Run spectral analysis (FFT)"""
    logger.info("Running spectral analysis...")
    
    output_dir = Path(args.output_dir) / 'spectral'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.encoding_comparison:
        # Compare multiple encodings
        encoding_names = args.encoding_comparison.split(',')
        encodings = {
            name: create_encoding(name, args.d_model, args.irrational)
            for name in encoding_names
        }
        
        results = {}
        for name, encoding in encodings.items():
            analyzer = EncodingAnalyzer(encoding, max_length=args.max_length, device=args.device)
            spectral_result = analyzer.spectral_analysis(num_positions=args.num_positions)
            results[name] = {'spectral': spectral_result}
            
            # Generate plot for each encoding
            if args.save_plots:
                visualizer = DiagnosticVisualizer(output_dir=str(output_dir))
                visualizer.plot_spectral_analysis(spectral_result, encoding_name=name)
        
        # Save comparison
        import json
        comparison_file = output_dir / 'spectral_comparison.json'
        with open(comparison_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    else:
        # Analyze single encoding
        encoding = create_encoding(args.encoding, args.d_model, args.irrational)
        analyzer = EncodingAnalyzer(encoding, max_length=args.max_length, device=args.device)
        
        results = analyzer.spectral_analysis(num_positions=args.num_positions)
        
        # Save results
        import json
        results_file = output_dir / 'spectral_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate plot
        if args.save_plots:
            visualizer = DiagnosticVisualizer(output_dir=str(output_dir))
            visualizer.plot_spectral_analysis(results, encoding_name=args.encoding)
    
    logger.info(f"Spectral analysis completed. Results saved to {output_dir}")
    return results


def run_attention_analysis(args):
    """Run attention pattern analysis"""
    logger.info("Running attention pattern analysis...")
    
    if not args.model_path:
        logger.error("Model path required for attention analysis")
        return None
    
    output_dir = Path(args.output_dir) / 'attention'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Load config if available
    config_path = Path(args.model_path).parent / 'config.yaml'
    if config_path.exists():
        config = ExperimentConfig.from_yaml(str(config_path))
    elif args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()
    
    # Create model
    model = PiSpiralTransformer(config.model)
    
    # Load checkpoint
    checkpoint_file = Path(args.model_path) / 'model.pt' if Path(args.model_path).is_dir() else Path(args.model_path)
    if checkpoint_file.exists():
        state_dict = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    # Create visualizer
    visualizer = AttentionVisualizer(model, device=device)
    
    # Create dummy input
    input_ids = torch.randint(0, 50000, (1, 1000), device=device)
    
    # Generate attention report
    results = visualizer.generate_attention_report(
        input_ids=input_ids,
        output_dir=str(output_dir),
    )
    
    logger.info(f"Attention analysis completed. Results saved to {output_dir}")
    return results


def run_profiling_analysis(args):
    """Run model profiling"""
    logger.info("Running model profiling...")
    
    if not args.model_path:
        logger.error("Model path required for profiling analysis")
        return None
    
    output_dir = Path(args.output_dir) / 'profiling'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Load config
    config_path = Path(args.model_path).parent / 'config.yaml'
    if config_path.exists():
        config = ExperimentConfig.from_yaml(str(config_path))
    elif args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()
    
    # Create model
    model = PiSpiralTransformer(config.model)
    
    # Load checkpoint
    checkpoint_file = Path(args.model_path) / 'model.pt' if Path(args.model_path).is_dir() else Path(args.model_path)
    if checkpoint_file.exists():
        state_dict = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    # Run profiling
    from diagnostics.profiling import profile_model_comprehensive
    
    results = profile_model_comprehensive(
        model=model,
        test_lengths=args.profile_lengths,
        output_dir=str(output_dir),
        device=device,
    )
    
    logger.info(f"Profiling completed. Results saved to {output_dir}")
    return results


def run_perplexity_analysis(args):
    """Run perplexity tracking analysis"""
    logger.info("Running perplexity analysis...")
    
    if not args.model_path:
        logger.error("Model path required for perplexity analysis")
        return None
    
    output_dir = Path(args.output_dir) / 'perplexity'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # Load config
    config_path = Path(args.model_path).parent / 'config.yaml'
    if config_path.exists():
        config = ExperimentConfig.from_yaml(str(config_path))
    elif args.config:
        config = ExperimentConfig.from_yaml(args.config)
    else:
        config = ExperimentConfig()
    
    # Create model
    model = PiSpiralTransformer(config.model)
    
    # Load checkpoint
    checkpoint_file = Path(args.model_path) / 'model.pt' if Path(args.model_path).is_dir() else Path(args.model_path)
    if checkpoint_file.exists():
        state_dict = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    # Create tracker
    tracker = PerplexityTracker()
    
    # Create dummy input
    input_ids = torch.randint(0, 50000, (1, 5000), device=device)
    
    # Compute perplexity by position
    results = tracker.compute_perplexity_by_position(
        model=model,
        input_ids=input_ids,
        device=device,
    )
    
    # Save results
    import json
    results_file = output_dir / 'perplexity_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Perplexity analysis completed. Results saved to {output_dir}")
    return results


def main():
    """Main diagnostic function"""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    logger.info("Starting diagnostic analysis...")
    logger.info(f"Arguments: {args}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analyses
    results = {}
    
    if args.analysis in ['collision', 'all']:
        try:
            collision_results = run_collision_analysis(args)
            results['collision'] = collision_results
        except Exception as e:
            logger.error(f"Collision analysis failed: {e}", exc_info=True)
    
    if args.analysis in ['spectral', 'all']:
        try:
            spectral_results = run_spectral_analysis(args)
            results['spectral'] = spectral_results
        except Exception as e:
            logger.error(f"Spectral analysis failed: {e}", exc_info=True)
    
    if args.analysis in ['attention', 'all']:
        try:
            attention_results = run_attention_analysis(args)
            results['attention'] = attention_results
        except Exception as e:
            logger.error(f"Attention analysis failed: {e}", exc_info=True)
    
    if args.analysis in ['profiling', 'all']:
        try:
            profiling_results = run_profiling_analysis(args)
            results['profiling'] = profiling_results
        except Exception as e:
            logger.error(f"Profiling analysis failed: {e}", exc_info=True)
    
    if args.analysis in ['perplexity', 'all']:
        try:
            perplexity_results = run_perplexity_analysis(args)
            results['perplexity'] = perplexity_results
        except Exception as e:
            logger.error(f"Perplexity analysis failed: {e}", exc_info=True)
    
    # Save combined results
    import json
    combined_results_file = output_dir / 'diagnostic_results.json'
    with open(combined_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"All diagnostic results saved to {combined_results_file}")
    logger.info("Diagnostic analysis completed successfully!")


if __name__ == '__main__':
    main()
