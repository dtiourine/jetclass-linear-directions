import itertools
import queue
import threading

import torch
from tqdm import tqdm


def prefetch(iterable, size=2):
    """Run `iterable` on a background thread, buffering up to `size` items so the
    next item is produced while the consumer works on the current one."""
    q = queue.Queue(maxsize=size)
    done = object()

    def worker():
        try:
            for item in iterable:
                q.put(item)
        finally:
            q.put(done)

    threading.Thread(target=worker, daemon=True).start()
    yield from iter(q.get, done)


@torch.no_grad()
def predict(model, dataloader, device):
    """
    Run the model over a full dataset and collect predictions.

    Returns:
        probs:  (N, num_classes) softmax probabilities as a numpy array
        labels: (N,) integer class labels as a numpy array

    Used by the test-set evaluation to compute AUC / background rejection 
    """
    model.eval()
    all_probs = []
    all_labels = []

    for x, mask, labels in tqdm(dataloader, desc="Predicting"):
        x = x.to(device)
        mask = mask.to(device)

        logits = model(x, mask)
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.cpu())
        all_labels.append(labels)

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    return probs, labels


def evaluate(model, dataloader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        batches = dataloader if max_batches is None else itertools.islice(dataloader, max_batches)
        for x, mask, labels in tqdm(batches, total=max_batches, desc="Evaluating"):
            x = x.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            logits = model(x, mask)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += batch_size

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy
