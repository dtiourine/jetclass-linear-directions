from pathlib import Path

import torch


def save_training_checkpoint(model, optimizer, epoch, val_acc, val_loss, filepath, scheduler=None):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "val_loss": val_loss,
    }
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()  
    torch.save(state, filepath)
    print(f"✓ Saved checkpoint: {filepath}")


def load_training_checkpoint(model, optimizer, filepath, device, scheduler=None):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    val_acc = checkpoint["val_acc"]
    print(f"✓ Loaded checkpoint from epoch {epoch} (Val Acc: {val_acc:.4f})")
    return epoch, val_acc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
