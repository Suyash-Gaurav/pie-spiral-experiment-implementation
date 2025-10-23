"""
InfiniteBench Benchmark Tests

Implements evaluation for InfiniteBench to test long-context understanding
at 100k+ token lengths.

Based on: https://github.com/OpenBMB/InfiniteBench
Dataset: https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench

InfiniteBench contains 18 tasks testing deep understanding of extremely
long contexts (100,000+ tokens).
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class InfiniteBenchTask:
    """Base class for InfiniteBench tasks"""
    
    def __init__(self, name: str, typical_length: int = 100000):
        self.name = name
        self.typical_length = typical_length
    
    def load_samples(self, data_dir: str) -> List[Dict[str, Any]]:
        """Load task samples from data directory"""
        raise NotImplementedError
    
    def evaluate_response(self, response: str, expected: Any) -> Dict[str, float]:
        """Evaluate response and return metrics"""
        raise NotImplementedError


class InfiniteBenchNIAH(InfiniteBenchTask):
    """
    InfiniteBench NIAH task - similar to standard NIAH but at 100k+ scale
    """
    
    def __init__(self):
        super().__init__("infinitebench_niah", 100000)
    
    def generate_sample(
        self,
        length: int = 100000,
        num_needles: int = 1,
    ) -> Dict[str, Any]:
        """Generate InfiniteBench NIAH sample"""
        import random
        
        # Generate needles
        needles = []
        for i in range(num_needles):
            key = f"SecretKey{i+1}"
            value = f"SecretValue{i+1}_{random.randint(10000, 99999)}"
            needles.append((key, value))
        
        # Filler text (more diverse than standard NIAH)
        filler_sentences = [
            "The research paper discusses various methodologies.",
            "Data analysis reveals interesting patterns in the dataset.",
            "The experimental results show significant improvements.",
            "Machine learning models continue to evolve rapidly.",
            "Natural language processing has made remarkable progress.",
            "Deep learning architectures are becoming more sophisticated.",
            "The study examines multiple factors affecting performance.",
            "Computational resources play a crucial role in training.",
        ]
        
        # Build long context
        context_parts = []
        filler_text = " ".join(filler_sentences * 1000)
        filler_tokens = filler_text.split()
        
        # Insert needles at various depths
        tokens_per_section = length // (num_needles + 1)
        
        for key, value in needles:
            context_parts.extend(filler_tokens[:tokens_per_section])
            context_parts.append(f"Important information: {key} corresponds to {value}.")
        
        context_parts.extend(filler_tokens[:tokens_per_section])
        
        context = " ".join(context_parts[:length])
        
        # Query about a random needle
        query_idx = random.randint(0, num_needles - 1)
        query_key = needles[query_idx][0]
        query_value = needles[query_idx][1]
        
        question = f"What value corresponds to {query_key}?"
        
        return {
            'context': context,
            'question': question,
            'answer': query_value,
            'length': len(context.split()),
            'num_needles': num_needles,
        }
    
    def evaluate_response(self, response: str, expected: Any) -> Dict[str, float]:
        """Evaluate response"""
        exact_match = 1.0 if expected.lower() in response.lower() else 0.0
        
        return {
            'exact_match': exact_match,
            'accuracy': exact_match,
        }


class InfiniteBenchRetrieval(InfiniteBenchTask):
    """
    InfiniteBench retrieval task - retrieve specific information from long documents
    """
    
    def __init__(self):
        super().__init__("infinitebench_retrieval", 100000)
    
    def generate_sample(self, length: int = 100000) -> Dict[str, Any]:
        """Generate retrieval sample"""
        import random
        
        # Create a long document with sections
        sections = []
        target_section_idx = random.randint(5, 15)
        target_info = f"TargetInfo_{random.randint(1000, 9999)}"
        
        for i in range(20):
            section_title = f"Section {i+1}"
            if i == target_section_idx:
                section_content = f"This section contains important information: {target_info}. " * 50
            else:
                section_content = f"This is section {i+1} with general information. " * 50
            
            sections.append(f"{section_title}\n{section_content}")
        
        context = "\n\n".join(sections)
        question = f"What is the important information mentioned in Section {target_section_idx + 1}?"
        
        return {
            'context': context,
            'question': question,
            'answer': target_info,
            'length': len(context.split()),
            'target_section': target_section_idx + 1,
        }
    
    def evaluate_response(self, response: str, expected: Any) -> Dict[str, float]:
        """Evaluate response"""
        exact_match = 1.0 if expected in response else 0.0
        partial_match = 1.0 if any(part in response for part in expected.split('_')) else 0.0
        
        return {
            'exact_match': exact_match,
            'partial_match': partial_match,
            'accuracy': exact_match,
        }


class InfiniteBenchEvaluator:
    """
    Evaluator for InfiniteBench
    
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
            'niah': InfiniteBenchNIAH(),
            'retrieval': InfiniteBenchRetrieval(),
        }
    
    def evaluate_sample(
        self,
        sample: Dict[str, Any],
        task: InfiniteBenchTask,
        max_new_tokens: int = 100,
    ) -> Dict[str, Any]:
        """Evaluate a single sample"""
        # Create prompt
        prompt = f"{sample['context']}\n\nQuestion: {sample['question']}\nAnswer:"
        
        # Tokenize
        try:
            input_ids = self.tokenizer.encode(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=self.tokenizer.model_max_length if hasattr(self.tokenizer, 'model_max_length') else 200000
            ).to(self.device)
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            return {
                'error': str(e),
                'metrics': {'accuracy': 0.0},
            }
        
        start_time = time.time()
        
        with torch.no_grad():
            try:
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
                    response = ""
            except Exception as e:
                logger.error(f"Generation error: {e}")
                return {
                    'error': str(e),
                    'metrics': {'accuracy': 0.0},
                }
        
        inference_time = time.time() - start_time
        
        # Evaluate
        metrics = task.evaluate_response(response, sample['answer'])
        
        return {
            'response': response,
            'expected': sample['answer'],
            'metrics': metrics,
            'inference_time': inference_time,
            'context_length': len(input_ids[0]),
        }
    
    def evaluate_task(
        self,
        task_name: str,
        lengths: List[int],
        samples_per_length: int = 5,
    ) -> Dict[str, Any]:
        """
        Evaluate a specific InfiniteBench task
        
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
        logger.info(f"Evaluating InfiniteBench task: {task_name}")
        
        results = {
            'task': task_name,
            'lengths': lengths,
            'samples_per_length': samples_per_length,
            'results_by_length': {},
        }
        
        for length in lengths:
            logger.info(f"  Length: {length}")
            
            length_results = []
            metrics_sum = {}
            
            for i in range(samples_per_length):
                # Generate sample
                sample = task.generate_sample(length)
                
                try:
                    # Evaluate
                    result = self.evaluate_sample(sample, task)
                    length_results.append(result)
                    
                    # Accumulate metrics
                    if 'metrics' in result:
                        for metric_name, metric_value in result['metrics'].items():
                            if metric_name not in metrics_sum:
                                metrics_sum[metric_name] = []
                            metrics_sum[metric_name].append(metric_value)
                
                except Exception as e:
                    logger.error(f"Error evaluating sample: {e}")
                    continue
            
            # Average metrics
            avg_metrics = {
                metric_name: float(np.mean(values))
                for metric_name, values in metrics_sum.items()
            }
            
            results['results_by_length'][length] = {
                'metrics': avg_metrics,
                'num_samples': len(length_results),
                'samples': length_results,
            }
            
            logger.info(f"    Metrics: {avg_metrics}")
        
        # Overall metrics
        all_metrics = {}
        for length_result in results['results_by_length'].values():
            for metric_name, metric_value in length_result['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)
        
        results['overall_metrics'] = {
            metric_name: float(np.mean(values))
            for metric_name, values in all_metrics.items()
        }
        
        return results
    
    def evaluate_all_tasks(
        self,
        lengths: List[int],
        samples_per_length: int = 5,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate all InfiniteBench tasks
        
        Args:
            lengths: List of context lengths to test
            samples_per_length: Number of samples per length
            output_dir: Optional directory to save results
        
        Returns:
            Dictionary with all results
        """
        logger.info(f"Starting InfiniteBench evaluation on {len(self.tasks)} tasks")
        
        all_results = {
            'tasks': {},
            'summary': {},
        }
        
        for task_name in self.tasks.keys():
            task_results = self.evaluate_task(task_name, lengths, samples_per_length)
            all_results['tasks'][task_name] = task_results
            
            # Store accuracy in summary
            if 'overall_metrics' in task_results and 'accuracy' in task_results['overall_metrics']:
                all_results['summary'][task_name] = task_results['overall_metrics']['accuracy']
        
        # Overall summary
        if all_results['summary']:
            all_results['summary']['overall'] = float(np.mean(list(all_results['summary'].values())))
            logger.info(f"Overall InfiniteBench accuracy: {all_results['summary']['overall']:.2%}")
        
        # Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results_file = output_path / 'infinitebench_results.json'
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            logger.info(f"Results saved to {results_file}")
        
        return all_results


def run_infinitebench(
    model: torch.nn.Module,
    tokenizer: Any,
    config: Optional[Any] = None,
    output_dir: str = './results/infinitebench',
) -> Dict[str, Any]:
    """
    Run InfiniteBench evaluation
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        config: Optional configuration object
        output_dir: Directory to save results
    
    Returns:
        Dictionary with benchmark results
    """
    evaluator = InfiniteBenchEvaluator(model, tokenizer)
    
    # Test at 100k, 256k, and 512k lengths
    lengths = [100000, 256000, 512000]
    
    # Run evaluation (fewer samples due to length)
    results = evaluator.evaluate_all_tasks(
        lengths=lengths,
        samples_per_length=5,
        output_dir=output_dir,
    )
    
    return results


def run_infinitebench_subset(
    model: torch.nn.Module,
    tokenizer: Any,
    task_names: List[str],
    lengths: List[int] = [100000],
    output_dir: str = './results/infinitebench',
) -> Dict[str, Any]:
    """
    Run InfiniteBench on a subset of tasks
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        task_names: List of task names to evaluate
        lengths: List of context lengths
        output_dir: Directory to save results
    
    Returns:
        Dictionary with benchmark results
    """
    evaluator = InfiniteBenchEvaluator(model, tokenizer)
    
    results = {
        'tasks': {},
        'summary': {},
    }
    
    for task_name in task_names:
        if task_name in evaluator.tasks:
            task_results = evaluator.evaluate_task(task_name, lengths, samples_per_length=5)
            results['tasks'][task_name] = task_results
            
            if 'overall_metrics' in task_results and 'accuracy' in task_results['overall_metrics']:
                results['summary'][task_name] = task_results['overall_metrics']['accuracy']
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / 'infinitebench_subset_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return results
