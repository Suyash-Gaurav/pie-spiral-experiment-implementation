"""
Configuration Management for π-Spiral Experiment

Provides flexible configuration system for managing experiment parameters
including model settings, training hyperparameters, and evaluation configs.
"""

import yaml
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Literal
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionalEncodingConfig:
    """Configuration for positional encoding"""
    type: Literal['rope', 'rope_ntk', 'alibi', 'pi_spiral', 'hybrid'] = 'pi_spiral'
    irrational: Literal['pi', 'e', 'sqrt2', 'phi', 'prng'] = 'pi'
    hybrid_K: int = 16000
    transition_width: int = 1000
    max_seq_len: int = 100000
    rope_base: float = 10000.0


@dataclass
class AttractorConfig:
    """Configuration for attractor state"""
    use_attractor: bool = True
    alpha_policy: Literal['fixed', 'exp_c_over_N', 'learned'] = 'exp_c_over_N'
    alpha_value: float = 0.99
    c_value: Optional[float] = None  # Defaults to π
    inject_layers: Literal['all', 'last_N'] = 'last_N'
    N_inject: int = 4
    attractor_inject: Literal['cross_attn', 'residual', 'none'] = 'cross_attn'
    d_state: Optional[int] = None  # Defaults to d_model


@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    model_name: str = 'pi_spiral_transformer'
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: Optional[int] = None  # Defaults to 4 * d_model
    vocab_size: int = 50000
    dropout: float = 0.1
    activation: Literal['relu', 'gelu', 'swish'] = 'gelu'
    layer_norm_eps: float = 1e-5
    tie_weights: bool = True
    
    # Positional encoding config
    pos_encoding: PositionalEncodingConfig = field(default_factory=PositionalEncodingConfig)
    
    # Attractor config
    attractor: AttractorConfig = field(default_factory=AttractorConfig)


@dataclass
class PretrainedModelConfig:
    """Configuration for pre-trained model adaptation"""
    model_name_or_path: str = 'Qwen/Qwen2.5-1.5B'
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_flash_attn: bool = True
    torch_dtype: str = 'auto'  # 'auto', 'float16', 'bfloat16', 'float32'
    device_map: str = 'auto'
    trust_remote_code: bool = True
    
    # Adaptation settings
    inject_pi_spiral: bool = True
    pos_encoding: PositionalEncodingConfig = field(default_factory=PositionalEncodingConfig)
    attractor: AttractorConfig = field(default_factory=AttractorConfig)


@dataclass
class TrainingConfig:
    """Configuration for training"""
    # Basic settings
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    max_steps: Optional[int] = None
    
    # Optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_scheduler_type: str = 'cosine'
    warmup_steps: int = 100
    warmup_ratio: float = 0.1
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Logging and checkpointing
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Evaluation
    eval_strategy: str = 'steps'
    eval_accumulation_steps: Optional[int] = None
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


@dataclass
class DataConfig:
    """Configuration for data loading"""
    # Dataset settings
    dataset_name: str = 'niah'
    data_dir: str = './data'
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    
    # Preprocessing
    max_seq_length: int = 100000
    preprocessing_num_workers: int = 4
    
    # DataLoader settings
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    
    # NIAH specific
    niah_lengths: List[int] = field(default_factory=lambda: [32000, 64000, 128000, 256000, 512000, 1000000])
    niah_depths: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    niah_samples_per_config: int = 10


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    # Benchmarks to run
    benchmarks: List[str] = field(default_factory=lambda: ['niah', 'ruler', 'infinitebench'])
    
    # NIAH settings
    niah_lengths: List[int] = field(default_factory=lambda: [32000, 64000, 128000, 256000, 512000, 1000000])
    niah_depths: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    
    # RULER settings
    ruler_tasks: List[str] = field(default_factory=lambda: ['counting', 'multi_needle', 'multi_hop', 'aggregation'])
    ruler_max_length: int = 256000
    
    # Generation settings
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95
    do_sample: bool = True
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: ['accuracy', 'exact_match', 'f1', 'rouge'])


@dataclass
class SystemConfig:
    """Configuration for system resources"""
    # Device settings
    device: str = 'cuda'
    num_gpus: int = 1
    
    # Memory management
    window_size: Optional[int] = None  # Sliding window size
    gradient_checkpointing: bool = False
    cpu_offload: bool = False
    
    # Performance
    use_flash_attention: bool = True
    compile_model: bool = False  # PyTorch 2.0 compile
    
    # Monitoring
    log_memory: bool = True
    log_throughput: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    # Experiment metadata
    experiment_name: str = 'pi_spiral_experiment'
    run_name: Optional[str] = None
    output_dir: str = './results'
    logging_dir: str = './logs'
    
    # Component configs
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrained: Optional[PretrainedModelConfig] = None
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # Ablation settings
    ablation_mode: Optional[str] = None  # 'irrational', 'attractor', 'alpha_policy', 'hybrid_K'
    ablation_values: Optional[List[Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    def to_yaml(self, path: str):
        """Save config to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
        logger.info(f"Saved config to {path}")
    
    def to_json(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved config to {path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Load config from dictionary"""
        # Recursively convert nested dicts to dataclass instances
        if 'model' in config_dict and isinstance(config_dict['model'], dict):
            if 'pos_encoding' in config_dict['model']:
                config_dict['model']['pos_encoding'] = PositionalEncodingConfig(**config_dict['model']['pos_encoding'])
            if 'attractor' in config_dict['model']:
                config_dict['model']['attractor'] = AttractorConfig(**config_dict['model']['attractor'])
            config_dict['model'] = ModelConfig(**config_dict['model'])
        
        if 'pretrained' in config_dict and config_dict['pretrained'] is not None:
            if isinstance(config_dict['pretrained'], dict):
                if 'pos_encoding' in config_dict['pretrained']:
                    config_dict['pretrained']['pos_encoding'] = PositionalEncodingConfig(**config_dict['pretrained']['pos_encoding'])
                if 'attractor' in config_dict['pretrained']:
                    config_dict['pretrained']['attractor'] = AttractorConfig(**config_dict['pretrained']['attractor'])
                config_dict['pretrained'] = PretrainedModelConfig(**config_dict['pretrained'])
        
        if 'training' in config_dict and isinstance(config_dict['training'], dict):
            config_dict['training'] = TrainingConfig(**config_dict['training'])
        
        if 'data' in config_dict and isinstance(config_dict['data'], dict):
            config_dict['data'] = DataConfig(**config_dict['data'])
        
        if 'evaluation' in config_dict and isinstance(config_dict['evaluation'], dict):
            config_dict['evaluation'] = EvaluationConfig(**config_dict['evaluation'])
        
        if 'system' in config_dict and isinstance(config_dict['system'], dict):
            config_dict['system'] = SystemConfig(**config_dict['system'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'ExperimentConfig':
        """Load config from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        logger.info(f"Loaded config from {path}")
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> 'ExperimentConfig':
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        logger.info(f"Loaded config from {path}")
        return cls.from_dict(config_dict)


def create_default_configs() -> Dict[str, ExperimentConfig]:
    """
    Create default configurations for different experiment scenarios
    
    Returns:
        Dictionary of configuration presets
    """
    configs = {}
    
    # Qwen2.5-1.5B configuration
    configs['qwen_1.5b'] = ExperimentConfig(
        experiment_name='qwen_1.5b_pi_spiral',
        pretrained=PretrainedModelConfig(
            model_name_or_path='Qwen/Qwen2.5-1.5B',
            load_in_4bit=False,
            use_flash_attn=True,
        ),
        system=SystemConfig(
            window_size=8000,
            use_flash_attention=True,
        ),
    )
    
    # Llama-3-8B configuration
    configs['llama_8b'] = ExperimentConfig(
        experiment_name='llama_8b_pi_spiral',
        pretrained=PretrainedModelConfig(
            model_name_or_path='meta-llama/Meta-Llama-3-8B',
            load_in_4bit=True,
            use_flash_attn=True,
        ),
        system=SystemConfig(
            window_size=4000,
            use_flash_attention=True,
            gradient_checkpointing=True,
        ),
    )
    
    # Llama-3-34B configuration (with CPU offload)
    configs['llama_34b'] = ExperimentConfig(
        experiment_name='llama_34b_pi_spiral',
        pretrained=PretrainedModelConfig(
            model_name_or_path='meta-llama/Meta-Llama-3-34B',
            load_in_4bit=True,
            use_flash_attn=True,
        ),
        system=SystemConfig(
            window_size=2000,
            use_flash_attention=True,
            gradient_checkpointing=True,
            cpu_offload=True,
        ),
        training=TrainingConfig(
            batch_size=1,
            gradient_accumulation_steps=16,
        ),
    )
    
    # Hybrid encoding configuration
    configs['hybrid'] = ExperimentConfig(
        experiment_name='hybrid_encoding',
        model=ModelConfig(
            pos_encoding=PositionalEncodingConfig(
                type='hybrid',
                hybrid_K=16000,
            ),
        ),
    )
    
    # Ablation study configuration
    configs['ablation_irrational'] = ExperimentConfig(
        experiment_name='ablation_irrational',
        ablation_mode='irrational',
        ablation_values=['pi', 'e', 'sqrt2', 'phi', 'prng'],
    )
    
    return configs


def get_config_for_phase(phase: int) -> ExperimentConfig:
    """
    Get configuration for specific experiment phase
    
    Args:
        phase: Phase number (0-10) from experiment plan
    
    Returns:
        ExperimentConfig for the phase
    """
    if phase == 0:
        # Phase 0: Setup
        return ExperimentConfig(experiment_name='phase0_setup')
    
    elif phase == 1:
        # Phase 1: Module development
        return ExperimentConfig(
            experiment_name='phase1_module_dev',
            model=ModelConfig(d_model=256, num_layers=2),  # Small for testing
        )
    
    elif phase == 2:
        # Phase 2: Sanity runs on Qwen2.5-1.5B
        return create_default_configs()['qwen_1.5b']
    
    elif phase == 3:
        # Phase 3: Core benchmarks on Qwen2.5-1.5B
        config = create_default_configs()['qwen_1.5b']
        config.experiment_name = 'phase3_core_benchmarks'
        config.evaluation.benchmarks = ['niah', 'ruler', 'infinitebench']
        return config
    
    elif phase == 4:
        # Phase 4: Llama-3-8B
        return create_default_configs()['llama_8b']
    
    elif phase == 5:
        # Phase 5: Llama-3-34B demo
        return create_default_configs()['llama_34b']
    
    elif phase == 6:
        # Phase 6: Ablations
        return create_default_configs()['ablation_irrational']
    
    else:
        return ExperimentConfig(experiment_name=f'phase{phase}')
