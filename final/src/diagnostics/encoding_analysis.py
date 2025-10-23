"""
Encoding Analysis Tools for π-Spiral Experiment

Implements diagnostic tools for analyzing positional encoding behavior:
- Positional collision detection (cosine similarity analysis)
- Spectral analysis (FFT) to detect periodic patterns
- Distance preservation analysis
- Encoding visualization

Based on Phase 7 requirements from experiment plan.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EncodingAnalyzer:
    """
    Analyzer for positional encoding behavior
    
    Provides tools to:
    - Detect positional collisions via cosine similarity
    - Perform spectral analysis (FFT) to identify periodic patterns
    - Analyze distance preservation properties
    - Compare different encoding schemes
    
    Args:
        encoding_module: Positional encoding module to analyze
        max_length: Maximum sequence length to analyze
        device: Device for computation
    """
    
    def __init__(
        self,
        encoding_module: torch.nn.Module,
        max_length: int = 100000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.encoding_module = encoding_module
        self.max_length = max_length
        self.device = device
        
        # Move encoding module to device
        self.encoding_module.to(device)
        self.encoding_module.eval()
    
    def compute_positional_vectors(
        self,
        positions: Optional[torch.Tensor] = None,
        num_positions: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute positional encoding vectors
        
        Args:
            positions: Specific positions to compute (optional)
            num_positions: Number of positions to compute if positions not provided
        
        Returns:
            Tensor of shape [num_positions, d_model] with positional vectors
        """
        if positions is None:
            if num_positions is None:
                num_positions = min(self.max_length, 10000)
            positions = torch.arange(num_positions, device=self.device)
        
        with torch.no_grad():
            # Get positional encodings
            if hasattr(self.encoding_module, 'get_positional_encoding'):
                pos_encodings = self.encoding_module.get_positional_encoding(positions)
            elif hasattr(self.encoding_module, 'forward'):
                # Try calling forward with positions
                pos_encodings = self.encoding_module(positions)
            else:
                raise ValueError("Encoding module must have 'get_positional_encoding' or 'forward' method")
        
        return pos_encodings
    
    def analyze_positional_collisions(
        self,
        num_positions: int = 10000,
        sample_offsets: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze positional collisions using cosine similarity
        
        Computes cosine similarity between positional vectors at various offsets
        to detect if distant positions have similar encodings (collisions).
        
        Args:
            num_positions: Number of positions to analyze
            sample_offsets: List of offsets to test (e.g., [1000, 10000, 50000])
        
        Returns:
            Dictionary with collision analysis results
        """
        logger.info(f"Analyzing positional collisions for {num_positions} positions...")
        
        if sample_offsets is None:
            sample_offsets = [100, 1000, 5000, 10000, 20000, 50000]
        
        # Compute positional vectors
        positions = torch.arange(num_positions, device=self.device)
        pos_vectors = self.compute_positional_vectors(positions)
        
        # Normalize vectors for cosine similarity
        pos_vectors_norm = torch.nn.functional.normalize(pos_vectors, p=2, dim=-1)
        
        results = {
            'num_positions': num_positions,
            'offsets': [],
            'mean_similarity': [],
            'max_similarity': [],
            'min_similarity': [],
            'std_similarity': [],
        }
        
        for offset in sample_offsets:
            if offset >= num_positions:
                continue
            
            # Compute cosine similarity at this offset
            similarities = []
            for i in range(num_positions - offset):
                sim = torch.dot(pos_vectors_norm[i], pos_vectors_norm[i + offset]).item()
                similarities.append(sim)
            
            similarities = np.array(similarities)
            
            results['offsets'].append(offset)
            results['mean_similarity'].append(float(np.mean(similarities)))
            results['max_similarity'].append(float(np.max(similarities)))
            results['min_similarity'].append(float(np.min(similarities)))
            results['std_similarity'].append(float(np.std(similarities)))
            
            logger.info(f"Offset {offset}: mean_sim={np.mean(similarities):.4f}, "
                       f"max_sim={np.max(similarities):.4f}")
        
        return results
    
    def spectral_analysis(
        self,
        num_positions: int = 10000,
        dimension_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Perform FFT spectral analysis on positional encodings
        
        Analyzes the frequency spectrum of positional encoding sequences
        to detect periodic patterns. RoPE should show periodic spikes,
        while π-Spiral should show a flatter spectrum.
        
        Args:
            num_positions: Number of positions to analyze
            dimension_indices: Specific dimensions to analyze (None = all)
        
        Returns:
            Dictionary with spectral analysis results
        """
        logger.info(f"Performing spectral analysis for {num_positions} positions...")
        
        # Compute positional vectors
        positions = torch.arange(num_positions, device=self.device)
        pos_vectors = self.compute_positional_vectors(positions).cpu().numpy()
        
        # Select dimensions to analyze
        if dimension_indices is None:
            # Analyze first 8 dimensions as representative sample
            dimension_indices = list(range(min(8, pos_vectors.shape[1])))
        
        results = {
            'num_positions': num_positions,
            'dimensions_analyzed': dimension_indices,
            'frequencies': [],
            'magnitudes': [],
            'peak_frequencies': [],
            'spectral_entropy': [],
        }
        
        for dim_idx in dimension_indices:
            # Get sequence for this dimension
            sequence = pos_vectors[:, dim_idx]
            
            # Compute FFT
            fft_result = np.fft.fft(sequence)
            frequencies = np.fft.fftfreq(len(sequence))
            magnitudes = np.abs(fft_result)
            
            # Only keep positive frequencies
            positive_freq_mask = frequencies > 0
            frequencies = frequencies[positive_freq_mask]
            magnitudes = magnitudes[positive_freq_mask]
            
            # Find peak frequencies
            peak_indices = np.argsort(magnitudes)[-5:]  # Top 5 peaks
            peak_freqs = frequencies[peak_indices].tolist()
            
            # Compute spectral entropy (measure of periodicity)
            # Lower entropy = more periodic, higher entropy = more random
            magnitude_probs = magnitudes / np.sum(magnitudes)
            magnitude_probs = magnitude_probs[magnitude_probs > 0]  # Remove zeros
            spectral_entropy = -np.sum(magnitude_probs * np.log(magnitude_probs))
            
            results['frequencies'].append(frequencies.tolist())
            results['magnitudes'].append(magnitudes.tolist())
            results['peak_frequencies'].append(peak_freqs)
            results['spectral_entropy'].append(float(spectral_entropy))
        
        # Average spectral entropy across dimensions
        results['mean_spectral_entropy'] = float(np.mean(results['spectral_entropy']))
        
        logger.info(f"Mean spectral entropy: {results['mean_spectral_entropy']:.4f}")
        
        return results
    
    def analyze_distance_preservation(
        self,
        num_positions: int = 1000,
        distance_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze how well the encoding preserves relative distances
        
        Args:
            num_positions: Number of positions to sample
            distance_pairs: List of (offset1, offset2) pairs to compare
        
        Returns:
            Dictionary with distance preservation analysis
        """
        logger.info(f"Analyzing distance preservation...")
        
        if distance_pairs is None:
            distance_pairs = [(1, 2), (10, 20), (100, 200), (1000, 2000)]
        
        # Compute positional vectors
        positions = torch.arange(num_positions, device=self.device)
        pos_vectors = self.compute_positional_vectors(positions)
        
        results = {
            'distance_pairs': distance_pairs,
            'distance_ratios': [],
            'distance_consistency': [],
        }
        
        for offset1, offset2 in distance_pairs:
            if offset2 >= num_positions:
                continue
            
            # Compute distances for multiple starting positions
            ratios = []
            for start in range(0, num_positions - offset2, 100):
                # Distance between positions at offset1
                dist1 = torch.norm(pos_vectors[start + offset1] - pos_vectors[start]).item()
                
                # Distance between positions at offset2
                dist2 = torch.norm(pos_vectors[start + offset2] - pos_vectors[start]).item()
                
                if dist1 > 0:
                    ratio = dist2 / dist1
                    ratios.append(ratio)
            
            if ratios:
                results['distance_ratios'].append({
                    'pair': (offset1, offset2),
                    'mean_ratio': float(np.mean(ratios)),
                    'std_ratio': float(np.std(ratios)),
                })
                
                # Consistency: lower std = more consistent
                results['distance_consistency'].append(float(np.std(ratios)))
        
        results['mean_consistency'] = float(np.mean(results['distance_consistency'])) if results['distance_consistency'] else 0.0
        
        logger.info(f"Mean distance consistency (lower=better): {results['mean_consistency']:.4f}")
        
        return results
    
    def compare_encodings(
        self,
        other_analyzer: 'EncodingAnalyzer',
        num_positions: int = 10000,
    ) -> Dict[str, Any]:
        """
        Compare this encoding with another encoding scheme
        
        Args:
            other_analyzer: Another EncodingAnalyzer to compare with
            num_positions: Number of positions to compare
        
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing encodings...")
        
        # Get collision analysis for both
        collisions_self = self.analyze_positional_collisions(num_positions)
        collisions_other = other_analyzer.analyze_positional_collisions(num_positions)
        
        # Get spectral analysis for both
        spectral_self = self.spectral_analysis(num_positions)
        spectral_other = other_analyzer.spectral_analysis(num_positions)
        
        results = {
            'collision_comparison': {
                'self_mean_similarity': collisions_self['mean_similarity'],
                'other_mean_similarity': collisions_other['mean_similarity'],
                'improvement': [
                    (s - o) for s, o in zip(
                        collisions_self['mean_similarity'],
                        collisions_other['mean_similarity']
                    )
                ],
            },
            'spectral_comparison': {
                'self_entropy': spectral_self['mean_spectral_entropy'],
                'other_entropy': spectral_other['mean_spectral_entropy'],
                'entropy_difference': spectral_self['mean_spectral_entropy'] - spectral_other['mean_spectral_entropy'],
            },
        }
        
        logger.info(f"Spectral entropy difference: {results['spectral_comparison']['entropy_difference']:.4f}")
        
        return results
    
    def generate_diagnostic_report(
        self,
        output_dir: str,
        num_positions: int = 10000,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive diagnostic report
        
        Args:
            output_dir: Directory to save diagnostic files
            num_positions: Number of positions to analyze
        
        Returns:
            Dictionary with all diagnostic results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating comprehensive diagnostic report...")
        
        # Run all analyses
        collision_results = self.analyze_positional_collisions(num_positions)
        spectral_results = self.spectral_analysis(num_positions)
        distance_results = self.analyze_distance_preservation(min(num_positions, 1000))
        
        report = {
            'collision_analysis': collision_results,
            'spectral_analysis': spectral_results,
            'distance_preservation': distance_results,
        }
        
        # Save report
        import json
        report_file = output_path / 'encoding_diagnostic_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Diagnostic report saved to {report_file}")
        
        # Save detailed CSV files for plotting
        self._save_collision_csv(collision_results, output_path / 'positional_collisions.csv')
        self._save_spectral_csv(spectral_results, output_path / 'spectral_analysis.csv')
        
        return report
    
    def _save_collision_csv(self, results: Dict[str, Any], output_file: Path):
        """Save collision analysis to CSV"""
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['offset', 'mean_similarity', 'max_similarity', 'min_similarity', 'std_similarity'])
            
            for i, offset in enumerate(results['offsets']):
                writer.writerow([
                    offset,
                    results['mean_similarity'][i],
                    results['max_similarity'][i],
                    results['min_similarity'][i],
                    results['std_similarity'][i],
                ])
        
        logger.info(f"Collision data saved to {output_file}")
    
    def _save_spectral_csv(self, results: Dict[str, Any], output_file: Path):
        """Save spectral analysis to CSV"""
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dimension', 'frequency', 'magnitude'])
            
            for dim_idx, (freqs, mags) in enumerate(zip(results['frequencies'], results['magnitudes'])):
                for freq, mag in zip(freqs, mags):
                    writer.writerow([dim_idx, freq, mag])
        
        logger.info(f"Spectral data saved to {output_file}")


def compare_multiple_encodings(
    encodings: Dict[str, torch.nn.Module],
    num_positions: int = 10000,
    output_dir: str = './diagnostics',
) -> Dict[str, Any]:
    """
    Compare multiple encoding schemes
    
    Args:
        encodings: Dictionary mapping encoding names to modules
        num_positions: Number of positions to analyze
        output_dir: Directory for output files
    
    Returns:
        Dictionary with comparison results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Comparing {len(encodings)} encoding schemes...")
    
    # Create analyzers for each encoding
    analyzers = {
        name: EncodingAnalyzer(module, num_positions)
        for name, module in encodings.items()
    }
    
    # Run analyses
    results = {}
    for name, analyzer in analyzers.items():
        logger.info(f"Analyzing {name}...")
        results[name] = {
            'collisions': analyzer.analyze_positional_collisions(num_positions),
            'spectral': analyzer.spectral_analysis(num_positions),
            'distance': analyzer.analyze_distance_preservation(min(num_positions, 1000)),
        }
    
    # Save comparison report
    import json
    report_file = output_path / 'encoding_comparison.json'
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Comparison report saved to {report_file}")
    
    return results
