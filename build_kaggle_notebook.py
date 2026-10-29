"""Build a Kaggle notebook with the trainer source embedded inline."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAINER_PATH = ROOT / "train_cnn.py"
NOTEBOOK_PATH = ROOT / "kaggle_train_all_in_one.ipynb"


def as_lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": as_lines(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": as_lines(text),
    }


trainer_source = TRAINER_PATH.read_text(encoding="utf-8")
if not trainer_source.endswith("\n"):
    trainer_source += "\n"

cells = [
    markdown_cell(
        """# Plant classifier — complete Kaggle notebook

This single notebook contains the Kaggle setup, the complete `train_cnn.py` source, the 60/20/20 training run, evaluation tables, charts, and ZIP creation. The trainer source below is embedded from the repository's `train_cnn.py`.

Before running, attach the `plant_classifier` image dataset to this notebook, enable a GPU accelerator, and enable Internet so TorchVision can fetch pretrained ResNet-18 weights if needed. Run the notebook from top to bottom once using **Save Version → Save & Run All** so the outputs are saved with the notebook version.
"""
    ),
    code_cell(
        """from pathlib import Path
import torch

DATA_DIR = Path("/kaggle/input/datasets/adityashrinet/plant-classifier/images/images")
OUTPUT_DIR = Path("/kaggle/working/cnn_outputs_60_20_20")

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("Enable a GPU accelerator in Kaggle Settings before training.")

if not DATA_DIR.is_dir():
    raise FileNotFoundError(f"Image folder not found: {DATA_DIR}")
class_folders = sorted(path.name for path in DATA_DIR.iterdir() if path.is_dir())
print("Plant classes:", class_folders)
if len(class_folders) != 10:
    raise ValueError(f"Expected 10 class folders, found {len(class_folders)}")
"""
    ),
    markdown_cell(
        """## Complete trainer source

The next cell writes the entire standalone trainer into the Kaggle working directory. Keeping it in this notebook makes the training implementation visible in this one file.
"""
    ),
    code_cell("%%writefile /kaggle/working/train_cnn.py\n" + trainer_source),
    code_cell(
        """import subprocess
import sys

subprocess.run([
    sys.executable,
    "/kaggle/working/train_cnn.py",
    "--data-dir", str(DATA_DIR),
    "--output-dir", str(OUTPUT_DIR),
    "--epochs", "15",
    "--batch-size", "32",
], check=True)
"""
    ),
    markdown_cell(
        """## Final scores, reports, and charts

This reads the actual files created by the training cell. It displays overall and per-class scores, split counts, training curves, and the confusion matrix.
"""
    ),
    code_cell(
        """import json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

with (OUTPUT_DIR / "test_report.json").open() as file:
    report = json.load(file)
with (OUTPUT_DIR / "split_summary.json").open() as file:
    split = json.load(file)
history = pd.read_json(OUTPUT_DIR / "training_history.json")
matrix = pd.read_csv(OUTPUT_DIR / "test_confusion_matrix.csv").set_index("true/predicted")

print(f"Split: {split['split']} | seed: {split['seed']}")
print(f"Images: train={split['train_images']}, validation={split['validation_images']}, test={split['test_images']}")
print(f"Test accuracy: {report['test_accuracy']:.1%}")
print(f"Test macro F1: {report['test_macro_f1']:.1%}")
print(f"Test loss: {report['test_loss']:.4f}")

print("\\nPer-class test scores:")
for plant, scores in report["class_report"].items():
    if isinstance(scores, dict) and "f1-score" in scores and plant not in {"macro avg", "weighted avg"}:
        print(f"{plant}: precision={scores['precision']:.1%}, recall={scores['recall']:.1%}, F1={scores['f1-score']:.1%}, images={int(scores['support'])}")

print("\\nPer-class split counts:")
display(pd.DataFrame(split["per_class"]).T)
print("\\nTest confusion matrix (rows=true, columns=predicted):")
display(matrix)

fig, axes = plt.subplots(2, 2, figsize=(17, 13))
axes[0, 0].plot(history["epoch"], history["train_loss"], marker="o", label="Train")
axes[0, 0].plot(history["epoch"], history["val_loss"], marker="o", label="Validation")
axes[0, 0].set(title="Loss by epoch", xlabel="Epoch", ylabel="Loss")
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.25)

axes[0, 1].plot(history["epoch"], history["train_accuracy"] * 100, marker="o", label="Train")
axes[0, 1].plot(history["epoch"], history["val_accuracy"] * 100, marker="o", label="Validation")
axes[0, 1].set(title="Accuracy by epoch", xlabel="Epoch", ylabel="Accuracy (%)")
axes[0, 1].legend(); axes[0, 1].grid(alpha=0.25)

axes[1, 0].plot(history["epoch"], history["train_macro_f1"] * 100, marker="o", label="Train")
axes[1, 0].plot(history["epoch"], history["val_macro_f1"] * 100, marker="o", label="Validation")
axes[1, 0].set(title="Macro F1 by epoch", xlabel="Epoch", ylabel="Macro F1 (%)")
axes[1, 0].legend(); axes[1, 0].grid(alpha=0.25)

image = axes[1, 1].imshow(matrix.values, cmap="Blues")
axes[1, 1].set_xticks(range(len(matrix.columns)), matrix.columns, rotation=60, ha="right")
axes[1, 1].set_yticks(range(len(matrix.index)), matrix.index)
axes[1, 1].set(title="Test confusion matrix", xlabel="Predicted", ylabel="Actual")
fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)
for row in range(matrix.shape[0]):
    for col in range(matrix.shape[1]):
        axes[1, 1].text(col, row, str(matrix.iat[row, col]), ha="center", va="center")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "results_dashboard.png", dpi=160, bbox_inches="tight")
plt.show()
"""
    ),
    markdown_cell("## Package outputs\n\nThe final cell creates one ZIP containing the best checkpoint and all reports/charts in `/kaggle/working`. Download that ZIP from this notebook version's Output tab."),
    code_cell(
        """import shutil

ZIP_PATH = shutil.make_archive(
    "/kaggle/working/cnn_outputs_60_20_20",
    "zip",
    root_dir=str(OUTPUT_DIR),
)
print("Created:", ZIP_PATH)
print("Size (MB):", round(Path(ZIP_PATH).stat().st_size / (1024 * 1024), 1))
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kaggle": {"accelerator": "gpu", "dataSources": [], "isInternetEnabled": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Created {NOTEBOOK_PATH}")
