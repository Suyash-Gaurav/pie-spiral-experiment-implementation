"""
NIAH (Needle in a Haystack) Benchmark Tests

Implements evaluation for the Needle in a Haystack benchmark to test
long-context retrieval capabilities of models with different positional encodings.

Based on: https://github.com/gkamradt/LLMTest_NeedleInAHaystack
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class NIAHEvaluator:
    """
    Evaluator for Needle in a Haystack (NIAH) benchmark
    
    Tests model's ability to retrieve specific information (needle)
    from long contexts (haystack) at various depths and lengths.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        device: Device for evaluation
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        self.model.to(device)
        self.model.eval()
    
    def create_niah_sample(
        self,
        context_length: int,
        depth: float,
        needle: str = "The special magic number is 42.",
        haystack_text: str = "The grass is green. The sky is blue. The sun is yellow. ",
        question: str = "What is the special magic number mentioned in the context?",
        answer: str = "42",
    ) -> Dict[str, Any]:
        """
        Create a single NIAH test sample
        
        Args:
            context_length: Target context length in tokens
            depth: Depth position (0.0 = start, 1.0 = end)
            needle: Text to hide in context
            haystack_text: Filler text for context
            question: Question about the needle
            answer: Expected answer
        
        Returns:
            Dictionary with sample data
        """
        # Tokenize components
        needle_tokens = self.tokenizer.encode(needle, add_special_tokens=False)
        haystack_tokens = self.tokenizer.encode(haystack_text, add_special_tokens=False)
        question_tokens = self.tokenizer.encode(question, add_special_tokens=False)
        
        # Calculate needle position
        needle_position = int(context_length * depth)
        
        # Build context by repeating haystack
        context_tokens = []
        while len(context_tokens) < context_length:
            context_tokens.extend(haystack_tokens)
        
        # Trim to exact length minus needle
        context_tokens = context_tokens[:context_length - len(needle_tokens)]
        
        # Insert needle at specified depth
        context_with_needle = (
            context_tokens[:needle_position] +
            needle_tokens +
            context_tokens[needle_position:]
        )
        
        # Trim to exact length
        context_with_needle = context_with_needle[:context_length]
        
        # Create full prompt
        full_tokens = context_with_needle + question_tokens
        
        return {
            'input_ids': full_tokens,
            'context_length': context_length,
            'depth': depth,
            'needle_position': needle_position,
            'needle': needle,
            'question': question,
            'answer': answer,
        }
    
    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        max_new_tokens: int = 50,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Evaluate a single NIAH sample
        
        Args:
            sample: Sample dictionary from create_niah_sample
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Dictionary with evaluation results
        """
        input_ids = torch.tensor([sample['input_ids']], device=self.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            # Generate response
            if hasattr(self.model, 'generate'):
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                )
            else:
                # Fallback: just get logits for next token
                outputs = self.model(input_ids)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs[0]
                
                # Get most likely next token
                next_token = torch.argmax(logits[:, -1, :], dim=-1)
                output_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        inference_time = time.time() - start_time
        
        # Decode response
        response = self.tokenizer.decode(
            output_ids[0, len(sample['input_ids']):],
            skip_special_tokens=True
        )
        
        # Check if answer is in response
        expected_answer = sample['answer'].lower().strip()
        response_lower = response.lower().strip()
        
        is_correct = expected_answer in response_lower
        
        return {
            'context_length': sample['context_length'],
            'depth': sample['depth'],
            'response': response,
            'expected_answer': sample['answer'],
            'is_correct': is_correct,
            'inference_time': inference_time,
        }
    
    def evaluate_grid(
        self,
        lengths: List[int],
        depths: List[float],
        samples_per_config: int = 10,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate NIAH on a grid of lengths and depths
        
        Args:
            lengths: List of context lengths to test
            depths: List of depth positions to test
            samples_per_config: Number of samples per (length, depth) pair
            output_dir: Optional directory to save results
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Starting NIAH evaluation grid: {len(lengths)} lengths x {len(depths)} depths")
        
        results = {
            'lengths': lengths,
            'depths': depths,
            'samples_per_config': samples_per_config,
            'results': [],
            'accuracy_matrix': np.zeros((len(lengths), len(depths))),
        }
        
        for i, length in enumerate(lengths):
            for j, depth in enumerate(depths):
                logger.info(f"Evaluating length={length}, depth={depth:.2f}")
                
                correct_count = 0
                config_results = []
                
                for sample_idx in range(samples_per_config):
                    # Create sample
                    sample = self.create_niah_sample(length, depth)
                    
                    # Evaluate
                    try:
                        result = self.evaluate_sample(sample)
                        config_results.append(result)
                        
                        if result['is_correct']:
                            correct_count += 1
                    
                    except Exception as e:
                        logger.error(f"Error evaluating sample: {e}")
                        continue
                
                # Calculate accuracy for this config
                accuracy = correct_count / samples_per_config if samples_per_config > 0 else 0
                results['accuracy_matrix'][i, j] = accuracy
                
                results['results'].append({
                    'length': length,
                    'depth': depth,
                    'accuracy': accuracy,
                    'correct_count': correct_count,
                    'total_count': samples_per_config,
                    'sample_results': config_results,
                })
                
                logger.info(f"  Accuracy: {accuracy:.2%}")
        
        # Compute summary statistics
        results['summary'] = {
            'overall_accuracy': float(np.mean(results['accuracy_matrix'])),
            'accuracy_by_length': {
                length: float(np.mean(results['accuracy_matrix'][i, :]))
                for i, length in enumerate(lengths)
            },
            'accuracy_by_depth': {
                depth: float(np.mean(results['accuracy_matrix'][:, j]))
                for j, depth in enumerate(depths)
            },
        }
        
        logger.info(f"Overall accuracy: {results['summary']['overall_accuracy']:.2%}")
        
        # Save results if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_file = output_path / 'niah_results.json'
            
            # Convert numpy arrays to lists for JSON serialization
            results_to_save = results.copy()
            results_to_save['accuracy_matrix'] = results['accuracy_matrix'].tolist()
            
            with open(results_file, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
        
        return results
    
    def evaluate_single_length(
        self,
        length: int,
        num_samples: int = 100,
        depth_range: Tuple[float, float] = (0.1, 0.9),
    ) -> Dict[str, Any]:
        """
        Evaluate NIAH at a single length with random depths
        
        Args:
            length: Context length to test
            num_samples: Number of samples to evaluate
            depth_range: Range of depths to sample from
        
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating NIAH at length {length} with {num_samples} samples")
        
        results = {
            'length': length,
            'num_samples': num_samples,
            'depth_range': depth_range,
            'samples': [],
        }
        
        correct_count = 0
        
        for i in range(num_samples):
            # Random depth
            depth = np.random.uniform(depth_range[0], depth_range[1])
            
            # Create and evaluate sample
            sample = self.create_niah_sample(length, depth)
            
            try:
                result = self.evaluate_sample(sample)
                results['samples'].append(result)
                
                if result['is_correct']:
                    correct_count += 1
            
            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}")
                continue
        
        results['accuracy'] = correct_count / num_samples if num_samples > 0 else 0
        results['correct_count'] = correct_count
        
        logger.info(f"Accuracy at length {length}: {results['accuracy']:.2%}")
        
        return results


def run_niah_benchmark(
    model: torch.nn.Module,
    tokenizer: Any,
    config: Optional[Any] = None,
    output_dir: str = './results/niah',
) -> Dict[str, Any]:
    """
    Run complete NIAH benchmark
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        config: Optional configuration object
        output_dir: Directory to save results
    
    Returns:
        Dictionary with benchmark results
    """
    evaluator = NIAHEvaluator(model, tokenizer)
    
    # Get lengths and depths from config or use defaults
    if config and hasattr(config, 'evaluation'):
        lengths = config.evaluation.niah_lengths
        depths = config.evaluation.niah_depths
    else:
        lengths = [32000, 64000, 128000, 256000]
        depths = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # Run evaluation
    results = evaluator.evaluate_grid(
        lengths=lengths,
        depths=depths,
        samples_per_config=10,
        output_dir=output_dir,
    )
    
    return results


def compare_niah_results(
    results_dict: Dict[str, Dict[str, Any]],
    output_dir: str = './results/niah_comparison',
) -> Dict[str, Any]:
    """
    Compare NIAH results across multiple models/encodings
    
    Args:
        results_dict: Dictionary mapping model names to their NIAH results
        output_dir: Directory to save comparison
    
    Returns:
        Dictionary with comparison results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison = {
        'models': list(results_dict.keys()),
        'overall_accuracy': {},
        'accuracy_by_length': {},
        'accuracy_by_depth': {},
    }
    
    for model_name, results in results_dict.items():
        if 'summary' in results:
            comparison['overall_accuracy'][model_name] = results['summary']['overall_accuracy']
            comparison['accuracy_by_length'][model_name] = results['summary']['accuracy_by_length']
            comparison['accuracy_by_depth'][model_name] = results['summary']['accuracy_by_depth']
    
    # Save comparison
    comparison_file = output_path / 'niah_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Comparison saved to {comparison_file}")
    
    return comparison
