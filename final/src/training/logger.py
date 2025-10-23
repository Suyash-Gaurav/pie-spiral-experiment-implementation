"""
Logging Utilities for π-Spiral Experiment

Provides comprehensive logging infrastructure for:
- Training metrics tracking
- System resource monitoring
- Experiment metadata
- Results aggregation and analysis
"""

import logging
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import sys


class ExperimentLogger:
    """
    Comprehensive experiment logger
    
    Manages multiple log streams:
    - Console output
    - Training metrics (JSONL)
    - System metrics (CSV)
    - Experiment metadata (JSON)
    
    Args:
        output_dir: Directory for log files
        experiment_name: Name of the experiment
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    
    def __init__(
        self,
        output_dir: str,
        experiment_name: str = 'experiment',
        log_level: int = logging.INFO,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.start_time = datetime.now()
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        # Log files
        self.metrics_file = self.output_dir / 'metrics.jsonl'
        self.system_file = self.output_dir / 'system.csv'
        self.metadata_file = self.output_dir / 'metadata.json'
        
        # Initialize files
        self._init_files()
        
        # Metadata storage
        self.metadata = {
            'experiment_name': experiment_name,
            'start_time': self.start_time.isoformat(),
            'output_dir': str(self.output_dir),
        }
    
    def _setup_logger(self, log_level: int) -> logging.Logger:
        """Setup logger with console and file handlers"""
        logger = logging.getLogger(self.experiment_name)
        logger.setLevel(log_level)
        
        # Remove existing handlers
        logger.handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_dir / f'{self.experiment_name}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _init_files(self):
        """Initialize log files with headers"""
        # System metrics CSV
        if not self.system_file.exists():
            with open(self.system_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'step', 'epoch', 'vram_mb', 'tokens_per_sec',
                    'learning_rate', 'loss', 'timestamp'
                ])
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log training metrics to JSONL file
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        if step is not None:
            log_entry['step'] = step
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_system_metrics(
        self,
        step: int,
        epoch: int,
        vram_mb: float,
        tokens_per_sec: float,
        learning_rate: float,
        loss: float,
    ):
        """
        Log system metrics to CSV file
        
        Args:
            step: Training step
            epoch: Current epoch
            vram_mb: VRAM usage in MB
            tokens_per_sec: Throughput in tokens/second
            learning_rate: Current learning rate
            loss: Current loss value
        """
        with open(self.system_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, epoch, vram_mb, tokens_per_sec,
                learning_rate, loss, datetime.now().isoformat()
            ])
    
    def update_metadata(self, updates: Dict[str, Any]):
        """
        Update experiment metadata
        
        Args:
            updates: Dictionary of metadata updates
        """
        self.metadata.update(updates)
        self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def finalize(self, final_metrics: Optional[Dict[str, Any]] = None):
        """
        Finalize logging and save summary
        
        Args:
            final_metrics: Optional final metrics to include
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.metadata['end_time'] = end_time.isoformat()
        self.metadata['duration_seconds'] = duration
        
        if final_metrics:
            self.metadata['final_metrics'] = final_metrics
        
        self._save_metadata()
        
        self.logger.info(f"Experiment completed in {duration:.2f} seconds")
        self.logger.info(f"Logs saved to {self.output_dir}")
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)


class MetricsAggregator:
    """
    Aggregates and analyzes metrics from log files
    
    Useful for post-experiment analysis and reporting
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.metrics_file = self.log_dir / 'metrics.jsonl'
        self.system_file = self.log_dir / 'system.csv'
    
    def load_metrics(self) -> List[Dict[str, Any]]:
        """Load all metrics from JSONL file"""
        metrics = []
        
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    metrics.append(json.loads(line))
        
        return metrics
    
    def load_system_metrics(self) -> List[Dict[str, Any]]:
        """Load system metrics from CSV file"""
        metrics = []
        
        if self.system_file.exists():
            with open(self.system_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    row['step'] = int(row['step'])
                    row['epoch'] = int(row['epoch'])
                    row['vram_mb'] = float(row['vram_mb'])
                    row['tokens_per_sec'] = float(row['tokens_per_sec'])
                    row['learning_rate'] = float(row['learning_rate'])
                    row['loss'] = float(row['loss'])
                    metrics.append(row)
        
        return metrics
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics from metrics
        
        Returns:
            Dictionary with summary statistics
        """
        system_metrics = self.load_system_metrics()
        
        if not system_metrics:
            return {}
        
        import numpy as np
        
        losses = [m['loss'] for m in system_metrics]
        vram = [m['vram_mb'] for m in system_metrics]
        throughput = [m['tokens_per_sec'] for m in system_metrics if m['tokens_per_sec'] > 0]
        
        summary = {
            'total_steps': len(system_metrics),
            'loss': {
                'mean': np.mean(losses),
                'std': np.std(losses),
                'min': np.min(losses),
                'max': np.max(losses),
                'final': losses[-1] if losses else None,
            },
            'vram_mb': {
                'mean': np.mean(vram),
                'std': np.std(vram),
                'min': np.min(vram),
                'max': np.max(vram),
                'peak': np.max(vram),
            },
            'tokens_per_sec': {
                'mean': np.mean(throughput) if throughput else 0,
                'std': np.std(throughput) if throughput else 0,
                'min': np.min(throughput) if throughput else 0,
                'max': np.max(throughput) if throughput else 0,
            },
        }
        
        return summary
    
    def export_summary(self, output_file: str):
        """
        Export summary statistics to JSON file
        
        Args:
            output_file: Path to output JSON file
        """
        summary = self.get_summary_statistics()
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)


def setup_logging(
    output_dir: str,
    experiment_name: str = 'experiment',
    log_level: str = 'INFO',
) -> ExperimentLogger:
    """
    Convenience function to setup experiment logging
    
    Args:
        output_dir: Directory for log files
        experiment_name: Name of the experiment
        log_level: Logging level string
    
    Returns:
        ExperimentLogger instance
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    
    level = level_map.get(log_level.upper(), logging.INFO)
    
    return ExperimentLogger(
        output_dir=output_dir,
        experiment_name=experiment_name,
        log_level=level,
    )
