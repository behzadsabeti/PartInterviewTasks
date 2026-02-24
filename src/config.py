"""
config.py

Centralised place for all hyper-parameters and paths.
Instead of scattering magic numbers across scripts and the notebook,
everything lives here so changing one value propagates everywhere.

Will contain:
  - DataConfig  : batch size, val split fraction, data directory, num_workers
  - TrainConfig : learning rate, epochs, weight decay, scheduler choice,
                  checkpoint directory, AMP (mixed precision) flag
  - ExperimentConfig : top-level config that groups the above + seed + model name
"""
