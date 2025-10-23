"""
Model Profiling Tools for π-Spiral Experiment

Implements profiling utilities for:
- Memory usage tracking (VRAM, RAM)
- Throughput measurement (tokens/sec)
- Model performance profiling
- Convergence diagnostics
- Perplexity tracking

Based on PyTorch profiler and Phase 8 requirements.
"""

import torch
import time
import psutil
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)


class ModelProfiler:
    """
    Profiler for model performance and resource usage
    
    Tracks:
    - GPU memory usage (VRAM)
    - CPU memory usage (RAM)
    - Throughput (tokens/second)
    - Latency per operation
    - Model parameter counts
    
    Args:
        model: PyTorch model to profile
        device: Device for profiling
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Profile storage
        self.profiles = []
        
        # Get model info
        self.model_info = self._get_model_info()
    
    def _get_model_info(self) -> Dict[str, Any]:
        """Get basic model information"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
        }
    
    def profile_memory(self) -> Dict[str, float]:
        """
        Profile current memory usage
        
        Returns:
            Dictionary with memory statistics
        """
        memory_stats = {}
        
        # GPU memory
        if torch.cuda.is_available() and self.device.startswith('cuda'):
            memory_stats['gpu_allocated_mb'] = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            memory_stats['gpu_reserved_mb'] = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            memory_stats['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            memory_stats['gpu_max_reserved_mb'] = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
        
        # CPU memory
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_stats['cpu_rss_mb'] = memory_info.rss / (1024 ** 2)
        memory_stats['cpu_vms_mb'] = memory_info.vms / (1024 ** 2)
        
        return memory_stats
    
    def reset_peak_memory_stats(self):
        """Reset peak memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def profile_forward_pass(
        self,
        input_data: Dict[str, torch.Tensor],
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> Dict[str, Any]:
        """
        Profile forward pass performance
        
        Args:
            input_data: Input data dictionary
            num_runs: Number of profiling runs
            warmup_runs: Number of warmup runs
        
        Returns:
            Dictionary with profiling results
        """
        self.model.eval()
        
        # Move input to device
        input_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in input_data.items()}
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(**input_data)
        
        # Synchronize
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Reset memory stats
        self.reset_peak_memory_stats()
        
        # Profile runs
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = self.model(**input_data)
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                latencies.append(end_time - start_time)
        
        # Get memory stats
        memory_stats = self.profile_memory()
        
        # Calculate throughput
        if 'input_ids' in input_data:
            num_tokens = input_data['input_ids'].numel()
            tokens_per_sec = num_tokens / np.mean(latencies)
        else:
            num_tokens = 0
            tokens_per_sec = 0
        
        results = {
            'latency_mean_ms': np.mean(latencies) * 1000,
            'latency_std_ms': np.std(latencies) * 1000,
            'latency_min_ms': np.min(latencies) * 1000,
            'latency_max_ms': np.max(latencies) * 1000,
            'throughput_tokens_per_sec': tokens_per_sec,
            'num_tokens': num_tokens,
            'memory': memory_stats,
        }
        
        return results
    
    def profile_length_scaling(
        self,
        lengths: List[int],
        batch_size: int = 1,
        vocab_size: int = 50000,
    ) -> Dict[str, Any]:
        """
        Profile how performance scales with sequence length
        
        Args:
            lengths: List of sequence lengths to test
            batch_size: Batch size for testing
            vocab_size: Vocabulary size for dummy inputs
        
        Returns:
            Dictionary with scaling results
        """
        logger.info(f"Profiling length scaling for lengths: {lengths}")
        
        results = {
            'lengths': lengths,
            'latency_ms': [],
            'throughput_tokens_per_sec': [],
            'memory_mb': [],
        }
        
        for length in lengths:
            logger.info(f"Profiling length {length}...")
            
            # Create dummy input
            input_ids = torch.randint(0, vocab_size, (batch_size, length))
            input_data = {'input_ids': input_ids}
            
            try:
                # Profile this length
                profile_result = self.profile_forward_pass(input_data, num_runs=5)
                
                results['latency_ms'].append(profile_result['latency_mean_ms'])
                results['throughput_tokens_per_sec'].append(profile_result['throughput_tokens_per_sec'])
                results['memory_mb'].append(profile_result['memory']['gpu_max_allocated_mb'] 
                                          if 'gpu_max_allocated_mb' in profile_result['memory'] 
                                          else 0)
                
            except RuntimeError as e:
                logger.warning(f"Failed to profile length {length}: {e}")
                results['latency_ms'].append(None)
                results['throughput_tokens_per_sec'].append(None)
                results['memory_mb'].append(None)
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def profile_with_pytorch_profiler(
        self,
        input_data: Dict[str, torch.Tensor],
        output_dir: str,
    ) -> str:
        """
        Profile using PyTorch profiler for detailed analysis
        
        Args:
            input_data: Input data dictionary
            output_dir: Directory to save profiler output
        
        Returns:
            Path to profiler output
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Move input to device
        input_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in input_data.items()}
        
        self.model.eval()
        
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            with torch.no_grad():
                _ = self.model(**input_data)
        
        # Save profiler output
        trace_file = output_path / 'profiler_trace.json'
        prof.export_chrome_trace(str(trace_file))
        
        # Save key averages
        key_averages_file = output_path / 'profiler_key_averages.txt'
        with open(key_averages_file, 'w') as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        
        logger.info(f"PyTorch profiler output saved to {output_path}")
        
        return str(trace_file)
    
    def save_profile(self, profile_data: Dict[str, Any], output_file: str):
        """Save profile data to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        logger.info(f"Profile saved to {output_file}")


class ConvergenceDiagnostics:
    """
    Diagnostics for training convergence
    
    Analyzes:
    - Loss convergence
    - Gradient statistics
    - Learning rate effects
    - Training stability
    """
    
    def __init__(self):
        self.loss_history = []
        self.gradient_norms = []
        self.learning_rates = []
    
    def update(
        self,
        loss: float,
        gradient_norm: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ):
        """Update diagnostics with new values"""
        self.loss_history.append(loss)
        
        if gradient_norm is not None:
            self.gradient_norms.append(gradient_norm)
        
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
    
    def analyze_convergence(self, window_size: int = 100) -> Dict[str, Any]:
        """
        Analyze convergence behavior
        
        Args:
            window_size: Window size for moving statistics
        
        Returns:
            Dictionary with convergence analysis
        """
        if len(self.loss_history) < window_size:
            window_size = len(self.loss_history)
        
        if window_size == 0:
            return {}
        
        recent_losses = self.loss_history[-window_size:]
        
        # Compute statistics
        loss_mean = np.mean(recent_losses)
        loss_std = np.std(recent_losses)
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        
        # Check for oscillations
        loss_diff = np.diff(recent_losses)
        sign_changes = np.sum(np.diff(np.sign(loss_diff)) != 0)
        oscillation_rate = sign_changes / len(loss_diff) if len(loss_diff) > 0 else 0
        
        analysis = {
            'loss_mean': float(loss_mean),
            'loss_std': float(loss_std),
            'loss_trend': float(loss_trend),
            'oscillation_rate': float(oscillation_rate),
            'is_converging': loss_trend < 0,
            'is_stable': loss_std < 0.1 * loss_mean,
        }
        
        # Gradient statistics
        if self.gradient_norms:
            recent_grads = self.gradient_norms[-window_size:]
            analysis['gradient_norm_mean'] = float(np.mean(recent_grads))
            analysis['gradient_norm_std'] = float(np.std(recent_grads))
            analysis['gradient_exploding'] = np.max(recent_grads) > 10 * np.mean(recent_grads)
            analysis['gradient_vanishing'] = np.mean(recent_grads) < 1e-6
        
        return analysis
    
    def detect_anomalies(self, threshold: float = 3.0) -> List[int]:
        """
        Detect anomalous loss values
        
        Args:
            threshold: Number of standard deviations for anomaly detection
        
        Returns:
            List of indices with anomalous losses
        """
        if len(self.loss_history) < 10:
            return []
        
        losses = np.array(self.loss_history)
        mean = np.mean(losses)
        std = np.std(losses)
        
        anomalies = []
        for i, loss in enumerate(losses):
            if abs(loss - mean) > threshold * std:
                anomalies.append(i)
        
        return anomalies


class PerplexityTracker:
    """
    Track perplexity over long sequences
    
    Useful for detecting periodic artifacts in positional encodings
    """
    
    def __init__(self):
        self.position_losses = []
    
    def compute_perplexity_by_position(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        device: str = 'cuda',
    ) -> Dict[str, Any]:
        """
        Compute per-token perplexity across sequence
        
        Args:
            model: Language model
            input_ids: Input token IDs [batch_size, seq_len]
            device: Device for computation
        
        Returns:
            Dictionary with per-position perplexity
        """
        model.eval()
        input_ids = input_ids.to(device)
        
        per_token_losses = []
        
        with torch.no_grad():
            # Get model outputs
            outputs = model(input_ids)
            
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Compute per-token loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Compute loss per token
            losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Reshape to [batch, seq_len-1]
            losses = losses.view(shift_labels.shape)
            
            # Average across batch
            per_token_losses = losses.mean(dim=0).cpu().numpy()
        
        # Compute perplexity
        perplexities = np.exp(per_token_losses)
        
        results = {
            'positions': list(range(len(per_token_losses))),
            'losses': per_token_losses.tolist(),
            'perplexities': perplexities.tolist(),
            'mean_perplexity': float(np.mean(perplexities)),
            'std_perplexity': float(np.std(perplexities)),
        }
        
        # Detect periodic patterns
        if len(perplexities) > 100:
            # Simple FFT to detect periodicity
            fft = np.fft.fft(perplexities)
            freqs = np.fft.fftfreq(len(perplexities))
            
            # Find dominant frequency
            positive_freqs = freqs[freqs > 0]
            positive_fft = np.abs(fft[freqs > 0])
            
            if len(positive_fft) > 0:
                dominant_freq_idx = np.argmax(positive_fft)
                dominant_freq = positive_freqs[dominant_freq_idx]
                
                results['dominant_frequency'] = float(dominant_freq)
                results['has_periodic_pattern'] = positive_fft[dominant_freq_idx] > 2 * np.mean(positive_fft)
        
        return results


def profile_model_comprehensive(
    model: torch.nn.Module,
    test_lengths: List[int],
    output_dir: str,
    device: str = 'cuda',
) -> Dict[str, Any]:
    """
    Comprehensive model profiling
    
    Args:
        model: Model to profile
        test_lengths: List of sequence lengths to test
        output_dir: Directory for output files
        device: Device for profiling
    
    Returns:
        Dictionary with comprehensive profiling results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting comprehensive model profiling...")
    
    profiler = ModelProfiler(model, device)
    
    # Get model info
    results = {
        'model_info': profiler.model_info,
    }
    
    # Profile length scaling
    scaling_results = profiler.profile_length_scaling(test_lengths)
    results['length_scaling'] = scaling_results
    
    # Save results
    results_file = output_path / 'profiling_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Profiling results saved to {results_file}")
    
    return results
