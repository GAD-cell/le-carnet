import torch
from dataclasses import dataclass


@dataclass
class TrainConfig:
    dataset_name: str = "MaxLSB/LeCarnet"
    tokenizer_name: str = "lightonai/pagnol-small"
    output_dir: str = "LeCarnet-3M/"
    load_checkpoint_path: str = "LeCarnet-3M/checkpoints/checkpoint-epoch-0.pt"
    load_checkpoint: bool = False
    mixed_precision: bool = True
    cache_dir: str = "cache/"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    eval_steps: int = 50000
    gradient_accumulation_steps: int = 2
    train_batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 5e-4
    num_warmup_steps: int = 100
    num_epochs: int = 1
    block_size: int = 512
    num_workers: int = 4


@dataclass
class CustomConfig:
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    hidden_act: str = "silu"
    block_size: int = 512
    max_position_embeddings: int = 2048
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True


@dataclass
class ModelConfig_3M:
    hidden_size: int = 64
    intermediate_size: int = 256
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    hidden_act: str = "silu"
    block_size: int = 512
    max_position_embeddings: int = 2048
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True


@dataclass
class ModelConfig_8M:
    hidden_size: int = 128
    intermediate_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    hidden_act: str = "silu"
    block_size: int = 512
    max_position_embeddings: int = 2048
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True


@dataclass
class ModelConfig_21M:
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_hidden_layers: int = 8
    num_attention_heads: int = 16
    hidden_act: str = "silu"
    block_size: int = 512
    max_position_embeddings: int = 2048
    initializer_range: float = 0.041666666666666664
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True
