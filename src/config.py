from dataclasses import dataclass, field


@dataclass
class DataConfig:
    data_dir:    str   = "./data"
    batch_size:  int   = 128
    val_split:   float = 0.1
    augment:     bool  = True
    num_workers: int   = 2


@dataclass
class TrainConfig:
    epochs:         int   = 30
    learning_rate:  float = 1e-3
    weight_decay:   float = 5e-4
    max_lr:         float = 1e-2
    use_amp:        bool  = True
    checkpoint_dir: str   = "./checkpoints"


@dataclass
class ExperimentConfig:
    seed:       int         = 42
    model_name: str         = "SimpleCNN"
    data:       DataConfig  = field(default_factory=DataConfig)
    train:      TrainConfig = field(default_factory=TrainConfig)
