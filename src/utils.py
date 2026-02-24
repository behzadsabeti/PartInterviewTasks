"""
utils.py

Small helper functions that do not belong to a specific module.
Imported freely by both src/ scripts and the notebook.

Functions to implement:

  set_seed(seed)
    - Seeds Python random, NumPy, and PyTorch (CPU + CUDA)
    - Sets cudnn.deterministic=True so results are reproducible across runs

  get_device()
    - Returns the best available torch.device: CUDA → MPS → CPU
    - Prints a short summary (GPU name, VRAM) when a GPU is found

  plot_history(history, title, save_path)
    - Receives the history dict returned by trainer.train()
    - Draws loss and accuracy curves side-by-side with matplotlib
    - Optionally saves the figure to disk instead of displaying it

  compare_histories(histories, metric, save_path)
    - Receives {model_name: history} for all trained models
    - Overlays one chosen metric (e.g. val_acc) for easy comparison
"""

# TODO: imports


# TODO: def set_seed(seed: int = 42) -> None


# TODO: def get_device(verbose: bool = True) -> torch.device


# TODO: def plot_history(history, title=None, save_path=None) -> None


# TODO: def compare_histories(histories, metric="val_acc", save_path=None) -> None
