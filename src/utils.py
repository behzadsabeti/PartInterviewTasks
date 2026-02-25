import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device(verbose=True):
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    if verbose:
        print(f"Device: {device}", end="")
        if device.type == "cuda":
            p = torch.cuda.get_device_properties(0)
            print(f"  ({torch.cuda.get_device_name(0)}, {p.total_memory/1e9:.1f} GB)", end="")
        print()
    return device


def plot_history(history, title=None, save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"],   label="val")
    ax1.set(xlabel="Epoch", ylabel="Loss",     title="Loss");     ax1.legend(); ax1.grid(alpha=.3)
    ax2.plot(epochs, history["train_acc"],  label="train")
    ax2.plot(epochs, history["val_acc"],    label="val")
    ax2.set(xlabel="Epoch", ylabel="Acc (%)", title="Accuracy"); ax2.legend(); ax2.grid(alpha=.3)
    if title: fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150) if save_path else plt.show()


def compare_histories(histories, metric="val_acc", save_path=None):
    plt.figure(figsize=(8, 4))
    for name, h in histories.items():
        plt.plot(range(1, len(h[metric]) + 1), h[metric], label=name)
    plt.xlabel("Epoch"); plt.ylabel(metric); plt.title("Comparison")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(save_path, dpi=150) if save_path else plt.show()
