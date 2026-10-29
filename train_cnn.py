"""Fine-tune a pretrained ResNet-18 on a folder-per-class plant dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def split_indices(targets: list[int], seed: int) -> tuple[list[int], list[int], list[int]]:
    """Make deterministic, per-class 60/20/20 train/validation/test splits."""
    rng = random.Random(seed)
    grouped: dict[int, list[int]] = {}
    for index, label in enumerate(targets):
        grouped.setdefault(label, []).append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for label, indices in grouped.items():
        if len(indices) < 3:
            raise ValueError(f"Class index {label} has fewer than 3 images; need at least 3 to split.")
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * 0.20))
        n_val = max(1, round(len(indices) * 0.20))
        while n_test + n_val >= len(indices):
            if n_test >= n_val:
                n_test -= 1
            else:
                n_val -= 1
        test_indices.extend(indices[:n_test])
        val_indices.extend(indices[n_test:n_test + n_val])
        train_indices.extend(indices[n_test + n_val:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)
    return train_indices, val_indices, test_indices


def make_transforms(image_size: int):
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, use_amp=False):
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    predictions: list[int] = []
    labels_all: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits = model(images)
                loss = loss_fn(logits, labels)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        loss_total += loss.item() * labels.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())

    mean_loss = loss_total / max(1, len(labels_all))
    accuracy = sum(p == y for p, y in zip(predictions, labels_all)) / max(1, len(labels_all))
    macro_f1 = f1_score(labels_all, predictions, average="macro", zero_division=0)
    return mean_loss, accuracy, macro_f1, labels_all, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing one subfolder per plant class.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cnn"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Initialize ResNet-18 randomly instead of downloading ImageNet weights.")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    class_folders = [p for p in data_dir.iterdir() if p.is_dir()]
    if len(class_folders) < 2:
        raise ValueError("Expected at least two class subfolders inside --data-dir.")
    for folder in class_folders:
        image_count = sum(
            1 for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if image_count == 0:
            raise ValueError(f"No supported image files found in class folder: {folder.name}")

    train_transform, eval_transform = make_transforms(args.image_size)
    train_base = datasets.ImageFolder(data_dir, transform=train_transform)
    eval_base = datasets.ImageFolder(data_dir, transform=eval_transform)
    if train_base.classes != eval_base.classes:
        raise RuntimeError("Training and evaluation class labels do not match.")

    train_indices, val_indices, test_indices = split_indices(train_base.targets, args.seed)
    split_counts = {
        "train": Counter(train_base.targets[i] for i in train_indices),
        "validation": Counter(train_base.targets[i] for i in val_indices),
        "test": Counter(train_base.targets[i] for i in test_indices),
    }
    split_summary = {
        class_name: {
            split_name: int(counts[class_index])
            for split_name, counts in split_counts.items()
        }
        for class_index, class_name in enumerate(train_base.classes)
    }
    train_set = Subset(train_base, train_indices)
    val_set = Subset(eval_base, val_indices)
    test_set = Subset(eval_base, test_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 2,
        "pin_memory": use_amp,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)
    test_loader = DataLoader(test_set, shuffle=False, **loader_options)

    weights = None if args.no_pretrained else models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(train_base.classes))
    model = model.to(device)

    train_counts = Counter(train_base.targets[i] for i in train_indices)
    class_weights = torch.tensor(
        [len(train_indices) / (len(train_base.classes) * train_counts[i])
         for i in range(len(train_base.classes))],
        dtype=torch.float32,
        device=device,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_resnet18.pt"
    history = []
    best_val_f1 = -1.0
    patience = 5
    stale_epochs = 0

    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Classes: {train_base.classes}")
    print(f"Images: train={len(train_set)}, validation={len(val_set)}, test={len(test_set)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1, _, _ = run_epoch(
            model, train_loader, loss_fn, device, optimizer, scaler, use_amp
        )
        val_loss, val_acc, val_f1, _, _ = run_epoch(
            model, val_loader, loss_fn, device, use_amp=use_amp
        )
        scheduler.step(val_f1)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.3f}, F1 {train_f1:.3f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.3f}, F1 {val_f1:.3f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": train_base.classes,
                "image_size": args.image_size,
                "seed": args.seed,
                "split": "60/20/20",
                "epoch": epoch,
                "val_macro_f1": val_f1,
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_f1, y_true, y_pred = run_epoch(
        model, test_loader, loss_fn, device, use_amp=use_amp
    )
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(train_base.classes))),
        target_names=train_base.classes,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(train_base.classes))))
    with (args.output_dir / "test_confusion_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true/predicted", *train_base.classes])
        for class_name, row in zip(train_base.classes, matrix):
            writer.writerow([class_name, *row.tolist()])
    (args.output_dir / "test_report.json").write_text(
        json.dumps({
            "split": "60/20/20",
            "seed": args.seed,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
            "class_report": report,
        }, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "split_summary.json").write_text(
        json.dumps({
            "split": "60/20/20",
            "seed": args.seed,
            "total_images": len(train_base),
            "train_images": len(train_indices),
            "validation_images": len(val_indices),
            "test_images": len(test_indices),
            "per_class": split_summary,
        }, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    print(f"Test accuracy: {test_acc:.3f} | test macro F1: {test_f1:.3f}")
    print(f"Saved best checkpoint and reports under: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
