"""
trainer.py

All training logic lives here so the notebook only has one function call per
experiment. Nothing in this file is architecture-specific.

Functions to implement:

  train_one_epoch(model, loader, criterion, optimizer, device, scaler)
    - Sets model to train mode
    - Iterates over batches: forward → loss → backward → optimizer step
    - Supports mixed-precision (AMP) via an optional GradScaler
    - Returns (avg_loss, accuracy%) for the epoch

  evaluate(model, loader, criterion, device)
    - Sets model to eval mode, runs inference under torch.no_grad()
    - Returns (avg_loss, accuracy%) — used for val and test sets

  train(model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, epochs, checkpoint_dir, model_name, use_amp)
    - Outer loop: calls train_one_epoch + evaluate each epoch
    - Prints a one-line progress summary per epoch
    - Saves the best checkpoint (by val accuracy) to checkpoint_dir/
    - Returns a history dict: {train_loss, train_acc, val_loss, val_acc}
      (one value per epoch — fed into utils.plot_history)

  load_checkpoint(model, path, device)
    - Loads a .pt file saved by train() back into the model
    - Returns the metadata dict (epoch, val_acc, …)
"""

# TODO: imports


# TODO: def train_one_epoch(...) -> tuple[float, float]


# TODO: def evaluate(...) -> tuple[float, float]


# TODO: def train(...) -> dict


# TODO: def load_checkpoint(...) -> dict
