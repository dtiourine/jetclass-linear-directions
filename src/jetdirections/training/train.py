import random

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from jetdirections.model.transformer import JetTagTransformer
from jetdirections.data.dataset import JetClassDataset
from jetdirections.data.dataloader import JetClassDataLoader
from jetdirections.data.sampling import load_stratified_batch
from jetdirections.training.engine import evaluate
from jetdirections.training.utils import save_training_checkpoint, count_parameters
from jetdirections import config


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    random.seed(config.SEED)
    torch.manual_seed(config.SEED)  

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    config.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    config.LOG_DIR.mkdir(exist_ok=True, parents=True)
    writer = SummaryWriter(log_dir=config.LOG_DIR)

    print("\nInitializing datasets...")
    train_dataset = JetClassDataset(config.TRAIN_DATA_DIR, max_particles=config.MAX_PARTICLES, shuffle_files=True)
    val_dataset = JetClassDataset(config.VAL_DATA_DIR, max_particles=config.MAX_PARTICLES, shuffle_files=False)

    train_loader = JetClassDataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        buffer_size=config.BUFFER_SIZE,
        drop_last=True,  # keeps actual batch count <= __len__ estimate so OneCycleLR never overruns
    )
    # Validate on a stratified subset of the val set during training
    # for a cheap, class-balanced signal to watch live.
    val_x, val_mask, val_labels = load_stratified_batch(val_dataset, per_class=5000, seed=config.SEED)
    val_batches = [
        (val_x[i:i + config.BATCH_SIZE], val_mask[i:i + config.BATCH_SIZE], val_labels[i:i + config.BATCH_SIZE])
        for i in range(0, val_x.size(0), config.BATCH_SIZE)
    ]
    print(f"Validation subset: {val_x.size(0):,} jets (stratified, balanced across 10 classes)")

    model = JetTagTransformer(
        input_dim=config.INPUT_DIM,
        embed_dim=config.MODEL_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT,
    ).to(device)
    print(f"Model has {count_parameters(model):,} trainable parameters\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.NUM_EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=config.WARMUP_PCT,
        anneal_strategy="cos",
        div_factor=config.LR_DIV_FACTOR,
        final_div_factor=config.LR_FINAL_DIV_FACTOR,
    )

    compiled_model = torch.compile(model)

    best_val_acc = 0.0
    global_step = 0
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print("-" * 60)

        compiled_model.train()
        running_loss, correct, total = 0.0, 0, 0
        for x, mask, labels in tqdm(train_loader, total=steps_per_epoch, desc="Training"):
            x = x.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = compiled_model(x, mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(compiled_model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            global_step += 1

            loss_val = loss.item()
            batch_correct = (logits.argmax(dim=1) == labels).sum().item()
            running_loss += loss_val * labels.size(0)
            correct += batch_correct
            total += labels.size(0)

            if global_step % config.LOG_EVERY_STEPS == 0:
                writer.add_scalar("Loss/train", loss_val, global_step)
                writer.add_scalar("Accuracy/train", batch_correct / labels.size(0), global_step)
                writer.add_scalar("LR", scheduler.get_last_lr()[0], global_step)

            if global_step % config.EVAL_EVERY_STEPS == 0:
                v_loss, v_acc = evaluate(compiled_model, val_batches, criterion, device)
                compiled_model.train()
                writer.add_scalar("Loss/val", v_loss, global_step)
                writer.add_scalar("Accuracy/val", v_acc, global_step)

            if global_step % config.CHECKPOINT_EVERY_STEPS == 0:
                ckpt_path = config.CHECKPOINT_DIR / f"checkpoint_{global_step // 1000}K.pt"
                save_training_checkpoint(model, optimizer, epoch, best_val_acc, float("nan"),
                                         ckpt_path, scheduler=scheduler)

        train_loss = running_loss / total
        train_acc = correct / total
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_loss, val_acc = evaluate(compiled_model, val_batches, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr:.6f}")

        writer.add_scalar("Loss/val", val_loss, global_step)
        writer.add_scalar("Accuracy/val", val_acc, global_step)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_training_checkpoint(model, optimizer, epoch + 1, val_acc, val_loss, config.CHECKPOINT_DIR / "jetclass_transformer.pt", scheduler=scheduler)

    writer.close()
    print(f"\n{'='*60}")
    print(f"Training complete! Best Val Acc: {best_val_acc:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
