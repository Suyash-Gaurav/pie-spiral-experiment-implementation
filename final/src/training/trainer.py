"""
Training Infrastructure for π-Spiral Experiment

Implements comprehensive training loop with:
- Mixed precision training (AMP)
- Gradient accumulation
- Learning rate scheduling with warmup
- Checkpointing and early stopping
- Comprehensive logging and monitoring
- Support for distributed training

Based on PyTorch best practices (2024-2025) and HuggingFace Trainer design.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    _LRScheduler
)
from typing import Optional, Dict, Any, Callable, List, Tuple
from pathlib import Path
import logging
import time
import json
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Tracks training state for checkpointing and resumption"""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = float('inf')
    best_epoch: int = 0
    patience_counter: int = 0
    total_train_loss: float = 0.0
    num_train_samples: int = 0


class Trainer:
    """
    Comprehensive trainer for π-Spiral experiments
    
    Features:
    - Automatic mixed precision (AMP) training
    - Gradient accumulation for effective large batch sizes
    - Learning rate warmup and cosine annealing
    - Asynchronous checkpointing
    - Early stopping with patience
    - Comprehensive metrics logging
    - Memory and throughput monitoring
    
    Args:
        model: PyTorch model to train
        train_dataloader: DataLoader for training data
        eval_dataloader: Optional DataLoader for evaluation
        optimizer: Optimizer (defaults to AdamW)
        config: Training configuration from config.py
        device: Device to train on
        output_dir: Directory for checkpoints and logs
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        config: Optional[Any] = None,
        device: str = 'cuda',
        output_dir: str = './results',
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training configuration
        self.batch_size = config.training.batch_size if config else 1
        self.gradient_accumulation_steps = config.training.gradient_accumulation_steps if config else 1
        self.num_epochs = config.training.num_epochs if config else 3
        self.max_steps = config.training.max_steps if config else None
        self.max_grad_norm = config.training.max_grad_norm if config else 1.0
        
        # Mixed precision
        self.use_amp = (config.training.fp16 or config.training.bf16) if config else False
        self.scaler = GradScaler() if self.use_amp else None
        
        # Optimizer
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Logging configuration
        self.logging_steps = config.training.logging_steps if config else 10
        self.eval_steps = config.training.eval_steps if config else 100
        self.save_steps = config.training.save_steps if config else 500
        self.save_total_limit = config.training.save_total_limit if config else 3
        
        # Early stopping
        self.early_stopping_patience = getattr(config.training, 'early_stopping_patience', None) if config else None
        
        # Training state
        self.state = TrainingState()
        
        # Metrics storage
        self.metrics_file = self.output_dir / 'metrics.jsonl'
        self.system_metrics_file = self.output_dir / 'system.csv'
        
        # Initialize system metrics CSV
        self._init_system_metrics()
        
        # Move model to device
        self.model.to(self.device)
        
        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Mixed precision: {self.use_amp}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
    
    def _create_optimizer(self) -> Optimizer:
        """Create AdamW optimizer with weight decay"""
        if self.config:
            return AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
                eps=self.config.training.adam_epsilon,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            return AdamW(self.model.parameters(), lr=5e-5)
    
    def _create_scheduler(self) -> _LRScheduler:
        """
        Create learning rate scheduler with warmup and cosine annealing
        
        Implements:
        1. Linear warmup phase
        2. Cosine annealing decay
        """
        if not self.config:
            return None
        
        # Calculate total training steps
        steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
        if self.max_steps:
            total_steps = self.max_steps
        else:
            total_steps = steps_per_epoch * self.num_epochs
        
        # Warmup steps
        warmup_steps = self.config.training.warmup_steps
        if warmup_steps == 0 and self.config.training.warmup_ratio > 0:
            warmup_steps = int(total_steps * self.config.training.warmup_ratio)
        
        # Create warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Create cosine annealing scheduler
        cosine_steps = total_steps - warmup_steps
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=cosine_steps,
            eta_min=self.config.training.learning_rate * 0.1
        )
        
        # Combine schedulers
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        logger.info(f"Scheduler: warmup_steps={warmup_steps}, total_steps={total_steps}")
        return scheduler
    
    def _init_system_metrics(self):
        """Initialize system metrics CSV file"""
        with open(self.system_metrics_file, 'w') as f:
            f.write('step,epoch,vram_mb,tokens_per_sec,learning_rate,loss\n')
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop
        
        Returns:
            Dictionary with training statistics
        """
        logger.info("Starting training...")
        logger.info(f"Num epochs: {self.num_epochs}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logger.info(f"Total optimization steps per epoch: {len(self.train_dataloader) // self.gradient_accumulation_steps}")
        
        self.model.train()
        start_time = time.time()
        
        for epoch in range(self.state.epoch, self.num_epochs):
            self.state.epoch = epoch
            epoch_start_time = time.time()
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info(f"{'='*50}")
            
            epoch_metrics = self._train_epoch()
            
            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            logger.info(f"\nEpoch {epoch + 1} completed in {epoch_time:.2f}s")
            logger.info(f"Average loss: {epoch_metrics['loss']:.4f}")
            
            # Evaluation
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                logger.info(f"Evaluation metrics: {eval_metrics}")
                
                # Early stopping check
                if self.early_stopping_patience:
                    if self._check_early_stopping(eval_metrics.get('loss', float('inf'))):
                        logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                        break
            
            # Save checkpoint at end of epoch
            self.save_checkpoint(f'checkpoint-epoch-{epoch + 1}')
            
            # Check if max steps reached
            if self.max_steps and self.state.global_step >= self.max_steps:
                logger.info(f"Max steps {self.max_steps} reached")
                break
        
        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time:.2f}s")
        
        return {
            'total_time': total_time,
            'final_loss': self.state.total_train_loss / max(self.state.num_train_samples, 1),
            'total_steps': self.state.global_step,
            'best_metric': self.state.best_metric,
            'best_epoch': self.state.best_epoch,
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_loss = 0.0
        num_batches = 0
        tokens_processed = 0
        epoch_start_time = time.time()
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(self.train_dataloader):
            step_start_time = time.time()
            
            # Move batch to device
            batch = self._prepare_batch(batch)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    loss = self._compute_loss(batch)
            else:
                loss = self._compute_loss(batch)
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step with gradient accumulation
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.max_grad_norm > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.state.global_step += 1
                
                # Logging
                if self.state.global_step % self.logging_steps == 0:
                    step_time = time.time() - step_start_time
                    self._log_training_step(loss.item() * self.gradient_accumulation_steps, step_time, batch)
                
                # Evaluation
                if self.eval_dataloader and self.state.global_step % self.eval_steps == 0:
                    eval_metrics = self.evaluate()
                    self.model.train()  # Back to training mode
                
                # Checkpointing
                if self.state.global_step % self.save_steps == 0:
                    self.save_checkpoint(f'checkpoint-step-{self.state.global_step}')
                
                # Check max steps
                if self.max_steps and self.state.global_step >= self.max_steps:
                    break
            
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Track tokens processed
            if 'input_ids' in batch:
                tokens_processed += batch['input_ids'].numel()
        
        avg_loss = epoch_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start_time
        tokens_per_sec = tokens_processed / epoch_time if epoch_time > 0 else 0
        
        return {
            'loss': avg_loss,
            'tokens_per_sec': tokens_per_sec,
            'num_batches': num_batches,
        }
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute loss for a batch
        
        Override this method for custom loss computation
        """
        outputs = self.model(**batch)
        
        if isinstance(outputs, dict) and 'loss' in outputs:
            return outputs['loss']
        elif isinstance(outputs, tuple):
            return outputs[0]  # Assume first element is loss
        else:
            raise ValueError("Model output must contain 'loss' key or be a tuple with loss as first element")
    
    def _prepare_batch(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Move batch to device"""
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [v.to(self.device) if isinstance(v, torch.Tensor) else v for v in batch]
        else:
            return batch.to(self.device) if isinstance(batch, torch.Tensor) else batch
    
    def _log_training_step(self, loss: float, step_time: float, batch: Dict[str, torch.Tensor]):
        """Log training step metrics"""
        lr = self.optimizer.param_groups[0]['lr']
        
        # Memory usage
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            torch.cuda.reset_peak_memory_stats(self.device)
        else:
            vram_mb = 0
        
        # Tokens per second
        if 'input_ids' in batch:
            num_tokens = batch['input_ids'].numel()
            tokens_per_sec = num_tokens / step_time if step_time > 0 else 0
        else:
            tokens_per_sec = 0
        
        # Console logging
        logger.info(
            f"Step {self.state.global_step} | "
            f"Loss: {loss:.4f} | "
            f"LR: {lr:.2e} | "
            f"VRAM: {vram_mb:.0f}MB | "
            f"Tokens/s: {tokens_per_sec:.0f}"
        )
        
        # Write to metrics file
        metrics = {
            'step': self.state.global_step,
            'epoch': self.state.epoch,
            'loss': loss,
            'learning_rate': lr,
            'vram_mb': vram_mb,
            'tokens_per_sec': tokens_per_sec,
            'timestamp': time.time(),
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        # Write to system metrics CSV
        with open(self.system_metrics_file, 'a') as f:
            f.write(f"{self.state.global_step},{self.state.epoch},{vram_mb:.2f},{tokens_per_sec:.2f},{lr:.6e},{loss:.6f}\n")
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
        
        logger.info("Running evaluation...")
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._prepare_batch(batch)
                
                if self.use_amp:
                    with autocast():
                        loss = self._compute_loss(batch)
                else:
                    loss = self._compute_loss(batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        metrics = {
            'eval_loss': avg_loss,
            'eval_perplexity': np.exp(avg_loss),
        }
        
        logger.info(f"Evaluation - Loss: {avg_loss:.4f}, Perplexity: {metrics['eval_perplexity']:.2f}")
        
        return metrics
    
    def _check_early_stopping(self, current_metric: float) -> bool:
        """
        Check if early stopping should be triggered
        
        Args:
            current_metric: Current validation metric (lower is better)
        
        Returns:
            True if training should stop
        """
        if current_metric < self.state.best_metric:
            self.state.best_metric = current_metric
            self.state.best_epoch = self.state.epoch
            self.state.patience_counter = 0
            
            # Save best model
            self.save_checkpoint('checkpoint-best')
            logger.info(f"New best metric: {current_metric:.4f}")
            
            return False
        else:
            self.state.patience_counter += 1
            logger.info(f"No improvement. Patience: {self.state.patience_counter}/{self.early_stopping_patience}")
            
            if self.state.patience_counter >= self.early_stopping_patience:
                return True
            
            return False
    
    def save_checkpoint(self, checkpoint_name: str):
        """
        Save model checkpoint
        
        Args:
            checkpoint_name: Name for the checkpoint directory
        """
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_path = checkpoint_dir / 'model.pt'
        torch.save(self.model.state_dict(), model_path)
        
        # Save optimizer state
        optimizer_path = checkpoint_dir / 'optimizer.pt'
        torch.save(self.optimizer.state_dict(), optimizer_path)
        
        # Save scheduler state
        if self.scheduler:
            scheduler_path = checkpoint_dir / 'scheduler.pt'
            torch.save(self.scheduler.state_dict(), scheduler_path)
        
        # Save training state
        state_path = checkpoint_dir / 'training_state.json'
        with open(state_path, 'w') as f:
            json.dump(asdict(self.state), f, indent=2)
        
        # Save scaler state for AMP
        if self.scaler:
            scaler_path = checkpoint_dir / 'scaler.pt'
            torch.save(self.scaler.state_dict(), scaler_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
        
        # Manage checkpoint limit
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain save_total_limit"""
        if self.save_total_limit <= 0:
            return
        
        # Get all checkpoint directories (excluding 'best')
        checkpoints = [d for d in self.output_dir.iterdir() 
                      if d.is_dir() and d.name.startswith('checkpoint-') 
                      and 'best' not in d.name]
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest checkpoints
        while len(checkpoints) > self.save_total_limit:
            oldest = checkpoints.pop(0)
            logger.info(f"Removing old checkpoint: {oldest}")
            import shutil
            shutil.rmtree(oldest)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint to resume training
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model state
        model_path = checkpoint_dir / 'model.pt'
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info(f"Loaded model from {model_path}")
        
        # Load optimizer state
        optimizer_path = checkpoint_dir / 'optimizer.pt'
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            logger.info(f"Loaded optimizer from {optimizer_path}")
        
        # Load scheduler state
        scheduler_path = checkpoint_dir / 'scheduler.pt'
        if self.scheduler and scheduler_path.exists():
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
            logger.info(f"Loaded scheduler from {scheduler_path}")
        
        # Load training state
        state_path = checkpoint_dir / 'training_state.json'
        if state_path.exists():
            with open(state_path, 'r') as f:
                state_dict = json.load(f)
                self.state = TrainingState(**state_dict)
            logger.info(f"Loaded training state from {state_path}")
        
        # Load scaler state
        scaler_path = checkpoint_dir / 'scaler.pt'
        if self.scaler and scaler_path.exists():
            self.scaler.load_state_dict(torch.load(scaler_path, map_location=self.device))
            logger.info(f"Loaded scaler from {scaler_path}")
        
        logger.info(f"Checkpoint loaded from {checkpoint_dir}")
