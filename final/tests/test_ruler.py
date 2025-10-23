"""
RULER Benchmark Tests

Implements evaluation for the RULER benchmark to test long-context
modeling capabilities across multiple task categories.

Based on: https://github.com/NVIDIA/RULER
Reference: https://arxiv.org/abs/2404.06654

RULER evaluates 4 task categories with 13 total tasks:
- Retrieval tasks
- Multi-needle tasks
- Multi-hop reasoning
- Aggregation tasks
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import json
import time
import random

logger = logging.getLogger(__name__)


class RULERTask:
    """Base class for RULER tasks"""
    
    def __init__(self, name: str, max_length: int = 256000):
        self.name = name
        self.max_length = max_length
    
    def generate_sample(self, length: int) -> Dict[str, Any]:
        """Generate a single test sample"""
        raise NotImplementedError
    
    def evaluate_response(self, response: str, expected: Any) -> bool:
        """Evaluate if response is correct"""
        raise NotImplementedError


class CountingTask(RULERTask):
    """
    Counting task: Count occurrences of a specific pattern in context
    """
    
    def __init__(self, max_length: int = 256000):
        super().__init__("counting", max_length)
    
    def generate_sample(self, length: int) -> Dict[str, Any]:
        """Generate counting task sample"""
        # Target word to count
        target_word = "MARKER"
        
        # Random number of occurrences
        num_occurrences = random.randint(5, 20)
        
        # Filler words
        filler_words = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
                       "cat", "bird", "tree", "house", "car", "book", "water", "sun"]
        
        # Build context with target word scattered throughout
        context_words = []
        positions = sorted(random.sample(range(length // 2), num_occurrences))
        
        word_idx = 0
        for pos in positions:
            # Add filler words until position
            while word_idx < pos:
                context_words.append(random.choice(filler_words))
                word_idx += 1
            # Add target word
            context_words.append(target_word)
            word_idx += 1
        
        # Fill remaining with filler words
        while len(context_words) < length // 2:
            context_words.append(random.choice(filler_words))
        
        context = " ".join(context_words)
        question = f"How many times does the word '{target_word}' appear in the context?"
        
        return {
            'context': context,
            'question': question,
            'answer': str(num_occurrences),
            'expected_count': num_occurrences,
            'target_word': target_word,
        }
    
    def evaluate_response(self, response: str, expected: Any) -> bool:
        """Check if response contains correct count"""
        expected_str = str(expected)
        # Look for the number in response
        import re
        numbers = re.findall(r'\d+', response)
        return expected_str in numbers


class MultiNeedleTask(RULERTask):
    """
    Multi-needle task: Retrieve multiple pieces of information from context
    """
    
    def __init__(self, max_length: int = 256000, num_needles: int = 5):
        super().__init__("multi_needle", max_length)
        self.num_needles = num_needles
    
    def generate_sample(self, length: int) -> Dict[str, Any]:
        """Generate multi-needle task sample"""
        # Generate needles with key-value pairs
        needles = []
        for i in range(self.num_needles):
            key = f"Key{i+1}"
            value = f"Value{i+1}_{random.randint(1000, 9999)}"
            needles.append((key, value))
        
        # Filler text
        filler = "The grass is green. The sky is blue. The sun is yellow. " * 100
        
        # Insert needles at random positions
        context_parts = []
        filler_tokens = filler.split()
        tokens_per_section = length // (self.num_needles + 1)
        
        for i, (key, value) in enumerate(needles):
            # Add filler
            context_parts.extend(filler_tokens[:tokens_per_section])
            # Add needle
            context_parts.append(f"The {key} is {value}.")
        
        # Add final filler
        context_parts.extend(filler_tokens[:tokens_per_section])
        
        context = " ".join(context_parts)
        
        # Ask about a random needle
        query_idx = random.randint(0, self.num_needles - 1)
        query_key = needles[query_idx][0]
        query_value = needles[query_idx][1]
        
        question = f"What is the value of {query_key}?"
        
        return {
            'context': context,
            'question': question,
            'answer': query_value,
            'needles': needles,
            'query_key': query_key,
        }
    
    def evaluate_response(self, response: str, expected: Any) -> bool:
        """Check if response contains correct value"""
        return expected.lower() in response.lower()


class MultiHopTask(RULERTask):
    """
    Multi-hop reasoning task: Requires connecting multiple pieces of information
    """
    
    def __init__(self, max_length: int = 256000, num_hops: int = 3):
        super().__init__("multi_hop", max_length)
        self.num_hops = num_hops
    
    def generate_sample(self, length: int) -> Dict[str, Any]:
        """Generate multi-hop reasoning sample"""
        # Create a chain of relationships
        entities = [f"Entity{i}" for i in range(self.num_hops + 1)]
        relationships = []
        
        for i in range(self.num_hops):
            relationships.append(f"{entities[i]} is connected to {entities[i+1]}.")
        
        # Filler text
        filler = "Some random information about various topics. " * 100
        filler_tokens = filler.split()
        
        # Insert relationships at different positions
        context_parts = []
        tokens_per_section = length // (self.num_hops + 1)
        
        for rel in relationships:
            context_parts.extend(filler_tokens[:tokens_per_section])
            context_parts.append(rel)
        
        context_parts.extend(filler_tokens[:tokens_per_section])
        
        context = " ".join(context_parts)
        question = f"What is {entities[0]} ultimately connected to through the chain?"
        answer = entities[-1]
        
        return {
            'context': context,
            'question': question,
            'answer': answer,
            'entities': entities,
            'num_hops': self.num_hops,
        }
    
    def evaluate_response(self, response: str, expected: Any) -> bool:
        """Check if response contains final entity"""
        return expected.lower() in response.lower()


class AggregationTask(RULERTask):
    """
    Aggregation task: Aggregate information across the entire context
    """
    
    def __init__(self, max_length: int = 256000):
        super().__init__("aggregation", max_length)
    
    def generate_sample(self, length: int) -> Dict[str, Any]:
        """Generate aggregation task sample"""
        # Scatter numbers throughout context
        numbers = [random.randint(1, 10) for _ in range(10)]
        expected_sum = sum(numbers)
        
        # Filler text
        filler_words = ["word"] * 100
        
        # Build context with numbers scattered
        context_parts = []
        tokens_per_section = length // (len(numbers) + 1)
        
        for num in numbers:
            context_parts.extend(filler_words[:tokens_per_section])
            context_parts.append(f"Number: {num}.")
        
        context_parts.extend(filler_words[:tokens_per_section])
        
        context = " ".join(context_parts)
        question = "What is the sum of all numbers mentioned in the context?"
        
        return {
            'context': context,
            'question': question,
            'answer': str(expected_sum),
            'numbers': numbers,
            'expected_sum': expected_sum,
        }
    
    def evaluate_response(self, response: str, expected: Any) -> bool:
        """Check if response contains correct sum"""
        import re
        numbers = re.findall(r'\d+', response)
        return str(expected) in numbers


class RULEREvaluator:
    """
    Evaluator for RULER benchmark
    
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
        
        # Initialize tasks
        self.tasks = {
            'counting': CountingTask(),
            'multi_needle': MultiNeedleTask(),
            'multi_hop': MultiHopTask(),
            'aggregation': AggregationTask(),
        }
    
    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        task: RULERTask,
        max_new_tokens: int = 50,
    ) -> Dict[str, Any]:
        """Evaluate a single sample"""
        # Create prompt
        prompt = f"{sample['context']}\n\nQuestion: {sample['question']}\nAnswer:"
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            # Generate response
            if hasattr(self.model, 'generate'):
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=False,
                )
                response = self.tokenizer.decode(
                    output_ids[0, input_ids.shape[1]:],
                    skip_special_tokens=True
                )
            else:
                # Fallback
                outputs = self.model(input_ids)
                response = ""
        
        inference_time = time.time() - start_time
        
        # Evaluate
        is_correct = task.evaluate_response(response, sample['answer'])
        
        return {
            'response': response,
            'expected': sample['answer'],
            'is_correct': is_correct,
            'inference_time': inference_time,
            'context_length': len(input_ids[0]),
        }
    
    def evaluate_task(
        self,
        task_name: str,
        lengths: List[int],
        samples_per_length: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate a specific RULER task
        
        Args:
            task_name: Name of the task
            lengths: List of context lengths to test
            samples_per_length: Number of samples per length
        
        Returns:
            Dictionary with evaluation results
        """
        if task_name not in self.tasks:
            raise ValueError(f"Unknown task: {task_name}")
        
        task = self.tasks[task_name]
        logger.info(f"Evaluating RULER task: {task_name}")
        
        results = {
            'task': task_name,
            'lengths': lengths,
            'samples_per_length': samples_per_length,
            'results_by_length': {},
        }
        
        for length in lengths:
            logger.info(f"  Length: {length}")
            
            correct_count = 0
            length_results = []
            
            for i in range(samples_per_length):
                # Generate sample
                sample = task.generate_sample(length)
                
                try:
                    # Evaluate
                    result = self.evaluate_sample(sample, task)
                    length_results.append(result)
                    
                    if result['is_correct']:
                        correct_count += 1
                
                except Exception as e:
                    logger.error(f"Error evaluating sample: {e}")
                    continue
            
            accuracy = correct_count / samples_per_length if samples_per_length > 0 else 0
            
            results['results_by_length'][length] = {
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_count': samples_per_length,
                'samples': length_results,
            }
            
            logger.info(f"    Accuracy: {accuracy:.2%}")
        
        # Overall accuracy
        all_accuracies = [r['accuracy'] for r in results['results_by_length'].values()]
        results['overall_accuracy'] = float(np.mean(all_accuracies))
        
        return results
    
    def evaluate_all_tasks(
        self,
        lengths: List[int],
        samples_per_length: int = 10,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate all RULER tasks
        
        Args:
            lengths: List of context lengths to test
            samples_per_length: Number of samples per length
            output_dir: Optional directory to save results
        
        Returns:
            Dictionary with all results
        """
        logger.info(f"Starting RULER evaluation on {len(self.tasks)} tasks")
        
        all_results = {
            'tasks': {},
            'summary': {},
        }
        
        for task_name in self.tasks.keys():
            task_results = self.evaluate_task(task_name, lengths, samples_per_length)
            all_results['tasks'][task_name] = task_results
            all_results['summary'][task_name] = task_results['overall_accuracy']
        
        # Overall summary
        all_results['summary']['overall'] = float(np.mean(list(all_results['summary'].values())))
        
        logger.info(f"Overall RULER accuracy: {all_results['summary']['overall']:.2%}")
        
        # Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_file = output_path / 'ruler_results.json'
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
        
        return all_results


def run_ruler_benchmark(
    model: torch.nn.Module,
    tokenizer: Any,
    config: Optional[Any] = None,
    output_dir: str = './results/ruler',
) -> Dict[str, Any]:
    """
    Run complete RULER benchmark
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        config: Optional configuration object
        output_dir: Directory to save results
    
    Returns:
        Dictionary with benchmark results
    """
    evaluator = RULEREvaluator(model, tokenizer)
    
    # Get configuration
    if config and hasattr(config, 'evaluation'):
        max_length = config.evaluation.ruler_max_length
        lengths = [32000, 64000, 128000, max_length]
    else:
        lengths = [32000, 64000, 128000, 256000]
    
    # Run evaluation
    results = evaluator.evaluate_all_tasks(
        lengths=lengths,
        samples_per_length=10,
        output_dir=output_dir,
    )
    
    return results
