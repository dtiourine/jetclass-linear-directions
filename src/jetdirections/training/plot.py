import csv
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from jetdirections.model.transformer import JetTagTransformer
from jetdirections.data.dataset import JetClassDataset
from jetdirections.training.engine import prefetch
from jetdirections import config

RESULTS_CSV = config.PROJ_ROOT / "plots" / "val_curve.csv"
PLOT_PNG = config.PROJ_ROOT / "plots" / "val_curve.png"

# Evaluate checkpoints at this step spacing (checkpoints are saved every 5k).
# Set to None to evaluate ALL of them. The CSV is additive, so a coarse run now
# can be densified later just by lowering this and re-running (cached steps skip).
STRIDE_STEPS = 20000


def checkpoint_step(path: Path) -> int | None:
    """checkpoint_5K.pt -> 5000"""
    match = re.fullmatch(r"checkpoint_(\d+)K\.pt", path.name)
    return int(match.group(1)) * 1000 if match else None


def find_checkpoints() -> list[Path]:
    paths = [p for p in config.CHECKPOINT_DIR.glob("checkpoint_*K.pt") if checkpoint_step(p) is not None]
    return sorted(paths, key=checkpoint_step)


def load_results() -> dict[int, tuple[float, float]]:
    if not RESULTS_CSV.exists():
        return {}
    with open(RESULTS_CSV) as f:
        return {int(r["step"]): (float(r["val_loss"]), float(r["val_acc"])) for r in csv.DictReader(f)}


def save_results(results: dict[int, tuple[float, float]]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "val_loss", "val_acc"])
        for step in sorted(results):
            val_loss, val_acc = results[step]
            writer.writerow([step, f"{val_loss:.6f}", f"{val_acc:.6f}"])


def plot(results: dict[int, tuple[float, float]]) -> None:
    steps = sorted(results)
    accs = [results[s][1] for s in steps]
    losses = [results[s][0] for s in steps]

    fig, ax_acc = plt.subplots(figsize=(8, 5))
    ax_acc.plot(steps, accs, "o-", color="tab:blue", label="Val accuracy")
    ax_acc.set_xlabel("Training step")
    ax_acc.set_ylabel("Validation accuracy", color="tab:blue")
    ax_acc.tick_params(axis="y", labelcolor="tab:blue")
    ax_acc.grid(True, alpha=0.3)

    ax_loss = ax_acc.twinx()
    ax_loss.plot(steps, losses, "s--", color="tab:red", label="Val loss")
    ax_loss.set_ylabel("Validation loss", color="tab:red")
    ax_loss.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle("Validation convergence (full val set)")
    fig.tight_layout()
    PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PNG, dpi=150)
    print(f"✓ Saved plot: {PLOT_PNG}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    checkpoints = find_checkpoints()
    if not checkpoints:
        print(f"No checkpoints found in {config.CHECKPOINT_DIR}")
        return
    results = load_results()  # already-evaluated checkpoints are skipped
    todo = [p for p in checkpoints if checkpoint_step(p) not in results]
    if STRIDE_STEPS:
        # Keep checkpoints at the stride spacing, plus the first and last so the
        # curve spans the whole run.
        endpoints = {checkpoint_step(todo[0]), checkpoint_step(todo[-1])}
        todo = [p for p in todo
                if checkpoint_step(p) % STRIDE_STEPS == 0 or checkpoint_step(p) in endpoints]
    print(f"Found {len(checkpoints)} checkpoints; {len(todo)} to evaluate "
          f"(stride={STRIDE_STEPS})")
    if not todo:
        plot(results)
        return

    val_dataset = JetClassDataset(config.VAL_DATA_DIR, max_particles=config.MAX_PARTICLES, shuffle_files=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    steps = [checkpoint_step(p) for p in todo]

    models = []
    for p in todo:
        state = torch.load(p, map_location="cpu", weights_only=False)
        m = JetTagTransformer(
            input_dim=config.INPUT_DIM, embed_dim=config.MODEL_DIM,
            num_heads=config.NUM_HEADS, num_layers=config.NUM_LAYERS,
            num_classes=config.NUM_CLASSES, dropout=config.DROPOUT,
        ).to(device)
        m.load_state_dict(state["model_state_dict"])
        m.eval()
        models.append(m)
    print(f"Loaded {len(models)} models onto {device}")

    loss_sums = torch.zeros(len(models), device=device)
    correct = torch.zeros(len(models), device=device)
    n_total = 0
    files_done = 0

    # Resume support: partial sums are checkpointed after each val file
    partial = config.PROJ_ROOT / "plots" / "val_curve_partial.npz"
    if partial.exists():
        z = np.load(partial)
        if list(z["steps"]) == steps:  # same checkpoint set -> safe to resume
            loss_sums = torch.tensor(z["loss_sums"], device=device)
            correct = torch.tensor(z["correct"], device=device)
            n_total = int(z["n_total"])
            files_done = int(z["files_done"])
            print(f"Resuming after {files_done} files, {n_total:,} jets already counted")

    val_dataset.files = val_dataset.files[files_done:]  # remaining files only
    n_files = files_done + len(val_dataset.files)
    bs = 1024  # large eval batch for throughput (metrics are batch-invariant)

    with torch.no_grad():
        for x, mask, labels in prefetch(val_dataset):  # one background-loaded file at a time
            for i in range(0, x.size(0), bs):
                xb = x[i:i + bs].to(device)
                mb = mask[i:i + bs].to(device)
                yb = labels[i:i + bs].to(device)
                for k, m in enumerate(models):  # every checkpoint sees this batch once
                    # bf16 autocast ~halves compute; loss/argmax stay fp32-stable.
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = m(xb, mb)
                    loss_sums[k] += criterion(logits.float(), yb) * yb.size(0)
                    correct[k] += (logits.argmax(1) == yb).sum()
                n_total += yb.size(0)
            files_done += 1
            np.savez(partial, steps=np.array(steps),
                     loss_sums=loss_sums.cpu().numpy(), correct=correct.cpu().numpy(),
                     n_total=n_total, files_done=files_done)
            print(f"file {files_done}/{n_files}: {n_total:,} jets counted", flush=True)

    val_loss = (loss_sums / n_total).cpu().numpy()
    val_acc = (correct / n_total).cpu().numpy()
    for k, step in enumerate(steps):
        results[step] = (float(val_loss[k]), float(val_acc[k]))
        print(f"step {step:>7}: val_loss {val_loss[k]:.4f} | val_acc {val_acc[k]:.4f}")
    save_results(results)
    partial.unlink(missing_ok=True)
    plot(results)


if __name__ == "__main__":
    main()
