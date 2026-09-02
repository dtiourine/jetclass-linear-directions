import torch
import argparse
from pathlib import Path

from jetdirections.model.transformer import JetTagTransformer
from jetdirections.data.dataset import JetClassDataset
from jetdirections.data.dataloader import JetClassDataLoader
from jetdirections.training.engine import predict
from jetdirections.training.metrics import compute_metrics, format_metrics
from jetdirections.config import (
    MODEL_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    DROPOUT,
    BATCH_SIZE,
    MAX_PARTICLES,
    TEST_DATA_DIR,
)


def load_checkpoint(checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_acc' in checkpoint:
        print(f"  Validation Accuracy: {checkpoint['val_acc']:.4f}")
    if 'val_loss' in checkpoint:
        print(f"  Validation Loss: {checkpoint['val_loss']:.4f}")

    return checkpoint


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    checkpoint = load_checkpoint(args.checkpoint, device)

    print("\nInitializing model...")
    model = JetTagTransformer(
        input_dim=17,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=10,
        dropout=args.dropout,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully")

    print(f"\nLoading test dataset from {args.test_data_dir}...")
    test_dataset = JetClassDataset(
        args.test_data_dir,
        max_particles=args.max_particles,
        shuffle_files=False,
    )

    test_loader = JetClassDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        buffer_size=args.buffer_size,
    )

    print(f"Test dataset loaded with {len(test_dataset.files)} files")

    print("\n" + "="*60)
    print("Starting evaluation on test set...")
    print("="*60 + "\n")

    probs, labels = predict(model, test_loader, device)
    metrics = compute_metrics(probs, labels)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(format_metrics(metrics))
    print("="*60 + "\n")


def cli():
    parser = argparse.ArgumentParser(
        description="Evaluate the Jet Tagging Transformer on test data"
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/jetclass_transformer.pt"),
        help="Path to model checkpoint"
    )

    # Data
    parser.add_argument("--test_data_dir", type=str, default=TEST_DATA_DIR)
    parser.add_argument("--max_particles", type=int, default=MAX_PARTICLES)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--buffer_size", type=int, default=20, help="Number of ROOT files to buffer")

    # Model (should match training configuration)
    parser.add_argument("--embed_dim", type=int, default=MODEL_DIM)
    parser.add_argument("--num_heads", type=int, default=NUM_HEADS)
    parser.add_argument("--num_layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
