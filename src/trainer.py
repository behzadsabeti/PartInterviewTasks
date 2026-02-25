import os, time
import torch
import torch.nn as nn


def _run_epoch(model, loader, criterion, optimizer, device, scaler, training):
    model.train() if training else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                out  = model(x)
                loss = criterion(out, y)
            if training:
                if scaler:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            n          += x.size(0)
    return total_loss / n, 100.0 * correct / n


def evaluate(model, loader, criterion, device):
    return _run_epoch(model, loader, criterion, None, device, None, training=False)


def train(model, train_loader, val_loader, criterion, optimizer, scheduler,
          device, epochs, checkpoint_dir, model_name, use_amp=True):
    os.makedirs(checkpoint_dir, exist_ok=True)
    scaler  = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best    = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = _run_epoch(model, train_loader, criterion, optimizer, device, scaler, True)
        vl, va = evaluate(model, val_loader, criterion, device)
        if scheduler: scheduler.step()
        for k, v in zip(history, [tl, ta, vl, va]): history[k].append(v)
        print(f"[{epoch:03d}/{epochs}] "
              f"loss {tl:.3f}/{vl:.3f}  acc {ta:.1f}%/{va:.1f}%  "
              f"lr {optimizer.param_groups[0]['lr']:.5f}  ({time.time()-t0:.1f}s)")
        if va > best:
            best = va
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_acc": va},
                       os.path.join(checkpoint_dir, f"{model_name}_best.pt"))

    print(f"Best val acc: {best:.2f}%")
    return history


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded '{path}'  (epoch {ckpt['epoch']}, val_acc {ckpt['val_acc']:.2f}%)") 
    return ckpt
