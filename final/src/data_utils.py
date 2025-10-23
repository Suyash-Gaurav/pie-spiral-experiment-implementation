"""
Data Utilities for π-Spiral Experiment

Provides data loading, preprocessing, and batch generation utilities for:
- NIAH (Needle in a Haystack)
- RULER benchmark
- InfiniteBench
- LongBench
- Custom long-context tasks
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple, Callable
import json
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class NIAHDataset(Dataset):
    """
    Needle in a Haystack (NIAH) Dataset
    
    Tests the model's ability to retrieve specific information (needle)
    from a long context (haystack).
    
    Args:
        num_samples: Number of samples to generate
        context_lengths: List of context lengths to test
        depths: List of depth positions (0.0 to 1.0) where needle is placed
        haystack_text: Text to use as haystack (repeated to fill context)
        needle_text: Text to use as needle
        question_template: Template for question about the needle
        tokenizer: Tokenizer for encoding text
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        num_samples: int,
        context_lengths: List[int],
        depths: List[float],
        haystack_text: str = "The grass is green. The sky is blue. The sun is yellow. ",
        needle_text: str = "The special magic number is 42.",
        question_template: str = "What is the special magic number mentioned in the context?",
        tokenizer: Optional[Any] = None,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.context_lengths = context_lengths
        self.depths = depths
        self.haystack_text = haystack_text
        self.needle_text = needle_text
        self.question_template = question_template
        self.tokenizer = tokenizer
        
        random.seed(seed)
        
        # Generate samples
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Dict[str, Any]]:
        """Generate NIAH samples"""
        samples = []
        
        for _ in range(self.num_samples):
            # Randomly select context length and depth
            context_length = random.choice(self.context_lengths)
            depth = random.choice(self.depths)
            
            # Create sample
            sample = self._create_sample(context_length, depth)
            samples.append(sample)
        
        return samples
    
    def _create_sample(self, context_length: int, depth: float) -> Dict[str, Any]:
        """
        Create a single NIAH sample
        
        Args:
            context_length: Target context length in tokens
            depth: Depth position (0.0 = start, 1.0 = end)
        
        Returns:
            Dictionary with context, question, answer, metadata
        """
        # Calculate needle position
        needle_position = int(context_length * depth)
        
        # Build context with needle at specified depth
        if self.tokenizer is not None:
            # Use tokenizer to control exact length
            haystack_tokens = self.tokenizer.encode(self.haystack_text, add_special_tokens=False)
            needle_tokens = self.tokenizer.encode(self.needle_text, add_special_tokens=False)
            
            # Repeat haystack to fill context
            tokens_before = []
            while len(tokens_before) < needle_position:
                tokens_before.extend(haystack_tokens)
            tokens_before = tokens_before[:needle_position]
            
            tokens_after = []
            remaining_length = context_length - needle_position - len(needle_tokens)
            while len(tokens_after) < remaining_length:
                tokens_after.extend(haystack_tokens)
            tokens_after = tokens_after[:remaining_length]
            
            # Combine
            full_tokens = tokens_before + needle_tokens + tokens_after
            context = self.tokenizer.decode(full_tokens)
        else:
            # Simple character-based approach
            chars_before = int(len(self.haystack_text) * (needle_position / context_length))
            chars_after = int(len(self.haystack_text) * ((context_length - needle_position) / context_length))
            
            context_before = (self.haystack_text * (chars_before // len(self.haystack_text) + 1))[:chars_before]
            context_after = (self.haystack_text * (chars_after // len(self.haystack_text) + 1))[:chars_after]
            
            context = context_before + self.needle_text + context_after
        
        return {
            'context': context,
            'question': self.question_template,
            'answer': self.needle_text,
            'context_length': context_length,
            'depth': depth,
            'needle_position': needle_position,
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class LongContextDataset(Dataset):
    """
    Generic long-context dataset for various benchmarks
    
    Supports loading data from JSON/JSONL files with flexible schema.
    
    Args:
        data_path: Path to data file
        max_length: Maximum sequence length
        tokenizer: Tokenizer for encoding
        task_type: Type of task ('qa', 'classification', 'generation')
    """
    
    def __init__(
        self,
        data_path: str,
        max_length: int = 100000,
        tokenizer: Optional[Any] = None,
        task_type: str = 'qa',
    ):
        self.data_path = Path(data_path)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.task_type = task_type
        
        # Load data
        self.samples = self._load_data()
        
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from file"""
        samples = []
        
        if not self.data_path.exists():
            logger.warning(f"Data file not found: {self.data_path}")
            return samples
        
        if self.data_path.suffix == '.json':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data
                else:
                    samples = [data]
        
        elif self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class DataCollator:
    """
    Collator for batching variable-length sequences
    
    Args:
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        padding: Padding strategy ('longest', 'max_length')
        return_tensors: Return type ('pt' for PyTorch)
    """
    
    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 100000,
        padding: str = 'longest',
        return_tensors: str = 'pt',
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.return_tensors = return_tensors
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples
        
        Args:
            batch: List of sample dictionaries
        
        Returns:
            Dictionary of batched tensors
        """
        # Extract texts
        contexts = [sample.get('context', '') for sample in batch]
        questions = [sample.get('question', '') for sample in batch]
        
        # Combine context and question
        inputs = [f"{ctx}\n\nQuestion: {q}\nAnswer:" for ctx, q in zip(contexts, questions)]
        
        # Tokenize
        encoded = self.tokenizer(
            inputs,
            padding=self.padding,
            max_length=self.max_length,
            truncation=True,
            return_tensors=self.return_tensors,
        )
        
        # Add labels if available
        if 'answer' in batch[0]:
            answers = [sample['answer'] for sample in batch]
            labels = self.tokenizer(
                answers,
                padding=self.padding,
                max_length=512,  # Answers are typically short
                truncation=True,
                return_tensors=self.return_tensors,
            )
            encoded['labels'] = labels['input_ids']
        
        # Add metadata
        for key in ['context_length', 'depth', 'task_type']:
            if key in batch[0]:
                encoded[key] = torch.tensor([sample[key] for sample in batch])
        
        return encoded


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    collate_fn: Optional[Callable] = None,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader with appropriate settings for long-context data
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size (typically 1 for long contexts)
        shuffle: Whether to shuffle data
        collate_fn: Custom collate function
        num_workers: Number of worker processes
        **kwargs: Additional DataLoader arguments
    
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        **kwargs
    )


def generate_synthetic_long_text(
    length: int,
    base_text: str = "This is a sample sentence. ",
    seed: Optional[int] = None,
) -> str:
    """
    Generate synthetic long text by repeating and varying base text
    
    Args:
        length: Target length in characters
        base_text: Base text to repeat
        seed: Random seed
    
    Returns:
        Generated text
    """
    if seed is not None:
        random.seed(seed)
    
    text = ""
    variations = [
        base_text,
        base_text.replace("sample", "example"),
        base_text.replace("This", "That"),
        base_text.replace("sentence", "statement"),
    ]
    
    while len(text) < length:
        text += random.choice(variations)
    
    return text[:length]


class StreamingDataset(Dataset):
    """
    Streaming dataset for very long sequences
    
    Loads data on-the-fly to avoid memory issues with extremely long contexts.
    
    Args:
        data_generator: Generator function that yields samples
        length: Number of samples (if known)
    """
    
    def __init__(
        self,
        data_generator: Callable,
        length: Optional[int] = None,
    ):
        self.data_generator = data_generator
        self.length = length
        self._cache = {}
    
    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        raise NotImplementedError("Length not specified for streaming dataset")
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx not in self._cache:
            # Generate sample on-the-fly
            self._cache[idx] = next(self.data_generator())
        return self._cache[idx]


def prepare_niah_sweep(
    lengths: List[int] = [32000, 64000, 128000, 256000, 512000, 1000000],
    depths: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    samples_per_config: int = 10,
    tokenizer: Optional[Any] = None,
) -> NIAHDataset:
    """
    Prepare NIAH sweep dataset as specified in experiment plan
    
    Args:
        lengths: List of context lengths to test
        depths: List of depth positions
        samples_per_config: Number of samples per (length, depth) configuration
        tokenizer: Tokenizer instance
    
    Returns:
        NIAHDataset instance
    """
    total_samples = len(lengths) * len(depths) * samples_per_config
    
    dataset = NIAHDataset(
        num_samples=total_samples,
        context_lengths=lengths,
        depths=depths,
        tokenizer=tokenizer,
    )
    
    logger.info(f"Prepared NIAH sweep with {total_samples} samples")
    logger.info(f"Lengths: {lengths}")
    logger.info(f"Depths: {depths}")
    
    return dataset


def load_benchmark_data(
    benchmark: str,
    data_dir: str = './data',
    split: str = 'test',
    **kwargs
) -> Dataset:
    """
    Load data from standard benchmarks
    
    Args:
        benchmark: Benchmark name ('niah', 'ruler', 'infinitebench', 'longbench')
        data_dir: Directory containing benchmark data
        split: Data split to load
        **kwargs: Additional arguments for dataset
    
    Returns:
        Dataset instance
    """
    data_path = Path(data_dir) / benchmark / f"{split}.jsonl"
    
    if benchmark == 'niah':
        return prepare_niah_sweep(**kwargs)
    else:
        return LongContextDataset(
            data_path=str(data_path),
            **kwargs
        )
