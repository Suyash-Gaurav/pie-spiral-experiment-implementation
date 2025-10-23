"""
Visualization Tools for π-Spiral Experiment

Implements visualization utilities for:
- Attention heatmaps
- Encoding analysis plots
- Performance curves
- Comparison charts
- Diagnostic visualizations

Uses matplotlib and seaborn for high-quality plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class DiagnosticVisualizer:
    """
    Visualizer for diagnostic results
    
    Creates publication-quality plots for:
    - Encoding analysis
    - Attention patterns
    - Performance metrics
    - Comparison charts
    
    Args:
        output_dir: Directory to save plots
        dpi: Resolution for saved figures
    """
    
    def __init__(
        self,
        output_dir: str = './figures',
        dpi: int = 300,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
    
    def plot_attention_heatmap(
        self,
        attention_matrix: np.ndarray,
        title: str = 'Attention Heatmap',
        xlabel: str = 'Key Position',
        ylabel: str = 'Query Position',
        filename: Optional[str] = None,
        cmap: str = 'viridis',
        figsize: Tuple[int, int] = (10, 8),
    ):
        """
        Plot attention heatmap
        
        Args:
            attention_matrix: Attention weights [seq_len, seq_len]
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            filename: Output filename (None = auto-generate)
            cmap: Colormap name
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(attention_matrix, cmap=cmap, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        # Labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Save figure
        if filename is None:
            filename = 'attention_heatmap.png'
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved attention heatmap to {output_path}")
    
    def plot_positional_collision(
        self,
        collision_data: Dict[str, Any],
        encoding_name: str = 'Encoding',
        filename: Optional[str] = None,
    ):
        """
        Plot positional collision analysis
        
        Args:
            collision_data: Results from EncodingAnalyzer.analyze_positional_collisions
            encoding_name: Name of encoding scheme
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        offsets = collision_data['offsets']
        mean_sim = collision_data['mean_similarity']
        std_sim = collision_data['std_similarity']
        
        # Plot mean similarity with error bars
        ax.errorbar(offsets, mean_sim, yerr=std_sim, marker='o', capsize=5,
                   label='Mean ± Std', linewidth=2)
        
        # Plot max and min
        ax.plot(offsets, collision_data['max_similarity'], '--', 
               label='Max', alpha=0.7)
        ax.plot(offsets, collision_data['min_similarity'], '--',
               label='Min', alpha=0.7)
        
        ax.set_xlabel('Position Offset')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(f'Positional Collision Analysis - {encoding_name}')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if filename is None:
            filename = f'collision_{encoding_name.lower().replace(" ", "_")}.png'
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved collision plot to {output_path}")
    
    def plot_spectral_analysis(
        self,
        spectral_data: Dict[str, Any],
        encoding_name: str = 'Encoding',
        filename: Optional[str] = None,
        max_freq: float = 0.5,
    ):
        """
        Plot spectral analysis (FFT) results
        
        Args:
            spectral_data: Results from EncodingAnalyzer.spectral_analysis
            encoding_name: Name of encoding scheme
            filename: Output filename
            max_freq: Maximum frequency to plot
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Spectral magnitude for first few dimensions
        ax1 = axes[0]
        for dim_idx in range(min(4, len(spectral_data['frequencies']))):
            freqs = np.array(spectral_data['frequencies'][dim_idx])
            mags = np.array(spectral_data['magnitudes'][dim_idx])
            
            # Filter by max frequency
            mask = freqs <= max_freq
            freqs = freqs[mask]
            mags = mags[mask]
            
            ax1.plot(freqs, mags, label=f'Dim {dim_idx}', alpha=0.7)
        
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Magnitude')
        ax1.set_title(f'Spectral Analysis - {encoding_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Spectral entropy by dimension
        ax2 = axes[1]
        entropies = spectral_data['spectral_entropy']
        dims = list(range(len(entropies)))
        
        ax2.bar(dims, entropies, alpha=0.7)
        ax2.set_xlabel('Dimension')
        ax2.set_ylabel('Spectral Entropy')
        ax2.set_title('Spectral Entropy by Dimension (Higher = Less Periodic)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add mean line
        mean_entropy = np.mean(entropies)
        ax2.axhline(mean_entropy, color='r', linestyle='--', 
                   label=f'Mean: {mean_entropy:.2f}')
        ax2.legend()
        
        plt.tight_layout()
        
        if filename is None:
            filename = f'spectral_{encoding_name.lower().replace(" ", "_")}.png'
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved spectral plot to {output_path}")
    
    def plot_encoding_comparison(
        self,
        comparison_data: Dict[str, Dict[str, Any]],
        metric: str = 'mean_similarity',
        filename: Optional[str] = None,
    ):
        """
        Plot comparison of multiple encoding schemes
        
        Args:
            comparison_data: Dictionary mapping encoding names to their results
            metric: Metric to compare ('mean_similarity', 'spectral_entropy', etc.)
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for encoding_name, data in comparison_data.items():
            if 'collisions' in data and metric in data['collisions']:
                offsets = data['collisions']['offsets']
                values = data['collisions'][metric]
                ax.plot(offsets, values, marker='o', label=encoding_name, linewidth=2)
        
        ax.set_xlabel('Position Offset')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Encoding Comparison - {metric.replace("_", " ").title()}')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if filename is None:
            filename = f'comparison_{metric}.png'
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved comparison plot to {output_path}")
    
    def plot_niah_heatmap(
        self,
        results: Dict[str, Any],
        encoding_name: str = 'Encoding',
        filename: Optional[str] = None,
    ):
        """
        Plot NIAH (Needle in a Haystack) results as heatmap
        
        Args:
            results: NIAH results with lengths and depths
            encoding_name: Name of encoding scheme
            filename: Output filename
        """
        # Extract data
        lengths = results.get('lengths', [])
        depths = results.get('depths', [])
        accuracy = results.get('accuracy', [])
        
        # Create 2D array
        accuracy_matrix = np.array(accuracy).reshape(len(lengths), len(depths))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', 
                      vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(range(len(depths)))
        ax.set_yticks(range(len(lengths)))
        ax.set_xticklabels([f'{d:.1f}' for d in depths])
        ax.set_yticklabels([f'{l//1000}k' for l in lengths])
        
        # Labels
        ax.set_xlabel('Depth in Context')
        ax.set_ylabel('Context Length')
        ax.set_title(f'NIAH Accuracy Heatmap - {encoding_name}')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(lengths)):
            for j in range(len(depths)):
                text = ax.text(j, i, f'{accuracy_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        if filename is None:
            filename = f'niah_heatmap_{encoding_name.lower().replace(" ", "_")}.png'
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved NIAH heatmap to {output_path}")
    
    def plot_performance_curves(
        self,
        metrics_data: List[Dict[str, Any]],
        x_key: str = 'step',
        y_key: str = 'loss',
        title: Optional[str] = None,
        filename: Optional[str] = None,
    ):
        """
        Plot performance curves (loss, accuracy, etc.)
        
        Args:
            metrics_data: List of metric dictionaries
            x_key: Key for x-axis values
            y_key: Key for y-axis values
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_values = [m[x_key] for m in metrics_data if x_key in m and y_key in m]
        y_values = [m[y_key] for m in metrics_data if x_key in m and y_key in m]
        
        ax.plot(x_values, y_values, linewidth=2)
        
        ax.set_xlabel(x_key.replace('_', ' ').title())
        ax.set_ylabel(y_key.replace('_', ' ').title())
        
        if title is None:
            title = f'{y_key.replace("_", " ").title()} vs {x_key.replace("_", " ").title()}'
        ax.set_title(title)
        
        ax.grid(True, alpha=0.3)
        
        if filename is None:
            filename = f'{y_key}_vs_{x_key}.png'
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved performance curve to {output_path}")
    
    def plot_system_metrics(
        self,
        system_data: List[Dict[str, Any]],
        filename: Optional[str] = None,
    ):
        """
        Plot system metrics (VRAM, throughput, etc.)
        
        Args:
            system_data: List of system metric dictionaries
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        steps = [m['step'] for m in system_data]
        
        # Plot 1: Loss
        if 'loss' in system_data[0]:
            losses = [m['loss'] for m in system_data]
            axes[0, 0].plot(steps, losses, linewidth=2)
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: VRAM
        if 'vram_mb' in system_data[0]:
            vram = [m['vram_mb'] for m in system_data]
            axes[0, 1].plot(steps, vram, linewidth=2, color='orange')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('VRAM (MB)')
            axes[0, 1].set_title('Memory Usage')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Throughput
        if 'tokens_per_sec' in system_data[0]:
            throughput = [m['tokens_per_sec'] for m in system_data if m['tokens_per_sec'] > 0]
            steps_throughput = [m['step'] for m in system_data if m['tokens_per_sec'] > 0]
            axes[1, 0].plot(steps_throughput, throughput, linewidth=2, color='green')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Tokens/Second')
            axes[1, 0].set_title('Training Throughput')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate
        if 'learning_rate' in system_data[0]:
            lr = [m['learning_rate'] for m in system_data]
            axes[1, 1].plot(steps, lr, linewidth=2, color='red')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if filename is None:
            filename = 'system_metrics.png'
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved system metrics plot to {output_path}")
    
    def plot_radar_chart(
        self,
        data: Dict[str, List[float]],
        categories: List[str],
        title: str = 'Performance Comparison',
        filename: Optional[str] = None,
    ):
        """
        Plot radar chart for multi-dimensional comparison
        
        Args:
            data: Dictionary mapping labels to values
            categories: List of category names
            title: Plot title
            filename: Output filename
        """
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        for label, values in data.items():
            values = values + values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=label)
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title(title, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        if filename is None:
            filename = 'radar_chart.png'
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved radar chart to {output_path}")
    
    def plot_length_sweep(
        self,
        results: Dict[str, Dict[int, float]],
        metric_name: str = 'Accuracy',
        filename: Optional[str] = None,
    ):
        """
        Plot metric vs context length for multiple encodings
        
        Args:
            results: Dictionary mapping encoding names to {length: metric} dicts
            metric_name: Name of the metric
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for encoding_name, length_metrics in results.items():
            lengths = sorted(length_metrics.keys())
            metrics = [length_metrics[l] for l in lengths]
            
            ax.plot(lengths, metrics, marker='o', linewidth=2, label=encoding_name)
        
        ax.set_xlabel('Context Length')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs Context Length')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if filename is None:
            filename = f'length_sweep_{metric_name.lower().replace(" ", "_")}.png'
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved length sweep plot to {output_path}")


def create_diagnostic_plots(
    diagnostic_data: Dict[str, Any],
    output_dir: str,
) -> List[str]:
    """
    Create all diagnostic plots from diagnostic data
    
    Args:
        diagnostic_data: Dictionary with diagnostic results
        output_dir: Directory to save plots
    
    Returns:
        List of created plot filenames
    """
    visualizer = DiagnosticVisualizer(output_dir)
    created_files = []
    
    # Plot collision analysis if available
    if 'collision_analysis' in diagnostic_data:
        visualizer.plot_positional_collision(
            diagnostic_data['collision_analysis'],
            encoding_name='Encoding'
        )
        created_files.append('collision_encoding.png')
    
    # Plot spectral analysis if available
    if 'spectral_analysis' in diagnostic_data:
        visualizer.plot_spectral_analysis(
            diagnostic_data['spectral_analysis'],
            encoding_name='Encoding'
        )
        created_files.append('spectral_encoding.png')
    
    logger.info(f"Created {len(created_files)} diagnostic plots")
    
    return created_files
