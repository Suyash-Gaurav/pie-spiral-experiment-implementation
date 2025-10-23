"""
Attention Pattern Visualization for π-Spiral Experiment

Implements tools for visualizing and analyzing attention patterns:
- Attention heatmaps
- Multi-head attention visualization
- Long-range attention analysis
- Attention pattern comparison across encodings

Based on BertViz approach and Phase 7 diagnostic requirements.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    Visualizer for attention patterns in transformer models
    
    Provides tools to:
    - Extract attention weights from model
    - Visualize attention heatmaps
    - Analyze long-range attention capabilities
    - Compare attention patterns across different encodings
    
    Args:
        model: Transformer model to analyze
        device: Device for computation
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Storage for attention weights
        self.attention_weights = []
        self._hooks = []
    
    def register_attention_hooks(self):
        """
        Register forward hooks to capture attention weights
        
        Hooks will store attention weights during forward pass
        """
        self.attention_weights = []
        self._hooks = []
        
        def attention_hook(module, input, output):
            """Hook to capture attention weights"""
            # Attention weights are typically returned as part of output
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]  # Usually second element
                if attn_weights is not None:
                    self.attention_weights.append(attn_weights.detach().cpu())
        
        # Register hooks on attention modules
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                if hasattr(module, 'forward'):
                    hook = module.register_forward_hook(attention_hook)
                    self._hooks.append(hook)
        
        logger.info(f"Registered {len(self._hooks)} attention hooks")
    
    def remove_attention_hooks(self):
        """Remove all registered hooks"""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self.attention_weights = []
    
    def extract_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Extract attention weights for given input
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
        
        Returns:
            List of attention weight tensors, one per layer
        """
        self.register_attention_hooks()
        
        with torch.no_grad():
            # Forward pass
            if attention_mask is not None:
                _ = self.model(input_ids.to(self.device), attention_mask=attention_mask.to(self.device))
            else:
                _ = self.model(input_ids.to(self.device))
        
        # Get captured attention weights
        attention_weights = self.attention_weights.copy()
        
        self.remove_attention_hooks()
        
        return attention_weights
    
    def compute_attention_statistics(
        self,
        attention_weights: List[torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Compute statistics about attention patterns
        
        Args:
            attention_weights: List of attention tensors [batch, heads, seq_len, seq_len]
        
        Returns:
            Dictionary with attention statistics
        """
        stats = {
            'num_layers': len(attention_weights),
            'layer_stats': [],
        }
        
        for layer_idx, attn in enumerate(attention_weights):
            # attn shape: [batch, num_heads, seq_len, seq_len]
            batch_size, num_heads, seq_len, _ = attn.shape
            
            # Average across batch and heads
            attn_avg = attn.mean(dim=(0, 1))  # [seq_len, seq_len]
            
            # Compute attention distance statistics
            attention_distances = []
            for i in range(seq_len):
                for j in range(seq_len):
                    if attn_avg[i, j] > 0.01:  # Threshold for significant attention
                        attention_distances.append(abs(i - j))
            
            layer_stat = {
                'layer': layer_idx,
                'num_heads': num_heads,
                'seq_len': seq_len,
                'mean_attention_distance': float(np.mean(attention_distances)) if attention_distances else 0,
                'max_attention_distance': float(np.max(attention_distances)) if attention_distances else 0,
                'attention_entropy': self._compute_attention_entropy(attn_avg),
            }
            
            stats['layer_stats'].append(layer_stat)
        
        return stats
    
    def _compute_attention_entropy(self, attention_matrix: torch.Tensor) -> float:
        """
        Compute entropy of attention distribution
        
        Higher entropy = more uniform attention
        Lower entropy = more focused attention
        """
        # Flatten and normalize
        attn_flat = attention_matrix.flatten()
        attn_probs = attn_flat / (attn_flat.sum() + 1e-10)
        
        # Remove zeros
        attn_probs = attn_probs[attn_probs > 0]
        
        # Compute entropy
        entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-10))
        
        return float(entropy.item())
    
    def analyze_long_range_attention(
        self,
        attention_weights: List[torch.Tensor],
        distance_thresholds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze long-range attention capabilities
        
        Measures how much attention is paid to distant tokens
        
        Args:
            attention_weights: List of attention tensors
            distance_thresholds: List of distance thresholds to analyze
        
        Returns:
            Dictionary with long-range attention analysis
        """
        if distance_thresholds is None:
            distance_thresholds = [10, 50, 100, 500, 1000, 5000]
        
        results = {
            'distance_thresholds': distance_thresholds,
            'layer_results': [],
        }
        
        for layer_idx, attn in enumerate(attention_weights):
            # Average across batch and heads
            attn_avg = attn.mean(dim=(0, 1)).numpy()  # [seq_len, seq_len]
            seq_len = attn_avg.shape[0]
            
            layer_result = {
                'layer': layer_idx,
                'seq_len': seq_len,
                'long_range_attention': {},
            }
            
            for threshold in distance_thresholds:
                if threshold >= seq_len:
                    continue
                
                # Compute attention to tokens beyond threshold distance
                long_range_attn = []
                for i in range(seq_len):
                    # Sum attention to tokens at distance > threshold
                    distant_attn = 0.0
                    for j in range(seq_len):
                        if abs(i - j) > threshold:
                            distant_attn += attn_avg[i, j]
                    long_range_attn.append(distant_attn)
                
                layer_result['long_range_attention'][threshold] = {
                    'mean': float(np.mean(long_range_attn)),
                    'max': float(np.max(long_range_attn)),
                    'min': float(np.min(long_range_attn)),
                }
            
            results['layer_results'].append(layer_result)
        
        return results
    
    def create_attention_heatmap_data(
        self,
        attention_weights: torch.Tensor,
        layer_idx: int = 0,
        head_idx: int = 0,
        max_length: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Prepare attention heatmap data for visualization
        
        Args:
            attention_weights: Attention tensor [batch, heads, seq_len, seq_len]
            layer_idx: Layer index (for multi-layer models)
            head_idx: Attention head index
            max_length: Maximum sequence length to visualize
        
        Returns:
            Dictionary with heatmap data
        """
        # Extract specific layer and head
        if len(attention_weights) > layer_idx:
            attn = attention_weights[layer_idx]
        else:
            attn = attention_weights[0]
        
        # Get specific head
        if attn.dim() == 4:  # [batch, heads, seq_len, seq_len]
            attn = attn[0, head_idx]  # [seq_len, seq_len]
        elif attn.dim() == 3:  # [heads, seq_len, seq_len]
            attn = attn[head_idx]
        
        # Limit length if specified
        if max_length is not None and attn.shape[0] > max_length:
            attn = attn[:max_length, :max_length]
        
        attn_np = attn.cpu().numpy()
        
        return {
            'attention_matrix': attn_np.tolist(),
            'shape': attn_np.shape,
            'layer': layer_idx,
            'head': head_idx,
            'max_value': float(attn_np.max()),
            'min_value': float(attn_np.min()),
            'mean_value': float(attn_np.mean()),
        }
    
    def compare_attention_patterns(
        self,
        attention_weights_1: List[torch.Tensor],
        attention_weights_2: List[torch.Tensor],
        encoding_name_1: str = 'encoding_1',
        encoding_name_2: str = 'encoding_2',
    ) -> Dict[str, Any]:
        """
        Compare attention patterns between two encoding schemes
        
        Args:
            attention_weights_1: Attention weights from first encoding
            attention_weights_2: Attention weights from second encoding
            encoding_name_1: Name of first encoding
            encoding_name_2: Name of second encoding
        
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing attention patterns: {encoding_name_1} vs {encoding_name_2}")
        
        # Compute statistics for both
        stats_1 = self.compute_attention_statistics(attention_weights_1)
        stats_2 = self.compute_attention_statistics(attention_weights_2)
        
        # Analyze long-range attention for both
        long_range_1 = self.analyze_long_range_attention(attention_weights_1)
        long_range_2 = self.analyze_long_range_attention(attention_weights_2)
        
        comparison = {
            'encoding_1': {
                'name': encoding_name_1,
                'statistics': stats_1,
                'long_range': long_range_1,
            },
            'encoding_2': {
                'name': encoding_name_2,
                'statistics': stats_2,
                'long_range': long_range_2,
            },
            'differences': {
                'mean_attention_distance': [],
                'attention_entropy': [],
            },
        }
        
        # Compute differences
        for layer_1, layer_2 in zip(stats_1['layer_stats'], stats_2['layer_stats']):
            comparison['differences']['mean_attention_distance'].append(
                layer_1['mean_attention_distance'] - layer_2['mean_attention_distance']
            )
            comparison['differences']['attention_entropy'].append(
                layer_1['attention_entropy'] - layer_2['attention_entropy']
            )
        
        return comparison
    
    def save_attention_data(
        self,
        attention_weights: List[torch.Tensor],
        output_dir: str,
        prefix: str = 'attention',
    ):
        """
        Save attention weights to files for later analysis
        
        Args:
            attention_weights: List of attention tensors
            output_dir: Directory to save files
            prefix: Prefix for filenames
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for layer_idx, attn in enumerate(attention_weights):
            filename = output_path / f'{prefix}_layer_{layer_idx}.pt'
            torch.save(attn, filename)
        
        logger.info(f"Saved {len(attention_weights)} attention weight files to {output_path}")
    
    def generate_attention_report(
        self,
        input_ids: torch.Tensor,
        output_dir: str,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive attention analysis report
        
        Args:
            input_ids: Input token IDs
            output_dir: Directory for output files
            attention_mask: Optional attention mask
        
        Returns:
            Dictionary with attention analysis results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating attention analysis report...")
        
        # Extract attention weights
        attention_weights = self.extract_attention_weights(input_ids, attention_mask)
        
        if not attention_weights:
            logger.warning("No attention weights captured. Model may not expose attention.")
            return {}
        
        # Compute statistics
        stats = self.compute_attention_statistics(attention_weights)
        
        # Analyze long-range attention
        long_range = self.analyze_long_range_attention(attention_weights)
        
        # Save attention weights
        self.save_attention_data(attention_weights, output_path)
        
        # Create report
        report = {
            'statistics': stats,
            'long_range_attention': long_range,
            'input_shape': list(input_ids.shape),
        }
        
        # Save report
        import json
        report_file = output_path / 'attention_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Attention report saved to {report_file}")
        
        return report


def analyze_attention_at_depth(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    needle_position: int,
    context_length: int,
) -> Dict[str, Any]:
    """
    Analyze attention to a specific position (needle) in context
    
    Useful for NIAH-style analysis of attention patterns
    
    Args:
        model: Transformer model
        input_ids: Input token IDs
        needle_position: Position of the needle token
        context_length: Total context length
    
    Returns:
        Dictionary with attention-to-needle analysis
    """
    visualizer = AttentionVisualizer(model)
    attention_weights = visualizer.extract_attention_weights(input_ids)
    
    if not attention_weights:
        return {}
    
    results = {
        'needle_position': needle_position,
        'context_length': context_length,
        'depth': needle_position / context_length,
        'layer_attention_to_needle': [],
    }
    
    for layer_idx, attn in enumerate(attention_weights):
        # Average across batch and heads
        attn_avg = attn.mean(dim=(0, 1)).numpy()  # [seq_len, seq_len]
        
        # Get attention to needle position from all positions
        attention_to_needle = attn_avg[:, needle_position]
        
        results['layer_attention_to_needle'].append({
            'layer': layer_idx,
            'mean_attention': float(np.mean(attention_to_needle)),
            'max_attention': float(np.max(attention_to_needle)),
            'positions_attending': int(np.sum(attention_to_needle > 0.01)),
        })
    
    return results
