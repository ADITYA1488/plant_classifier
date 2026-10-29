"""Print saved CNN scores and plot the training/evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Folder with the recorded results or a training output folder.",
    )
    parser.add_argument(
        "--save-only",
        action="store_true",
        help="Save the dashboard without opening a plot window.",
    )
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()

    report_path = results_dir / "test_report.json"
    summary_path = results_dir / "cnn_60_20_20_summary.json"
    history_path = results_dir / "training_history.json"
    split_path = results_dir / "split_summary.json"
    if not history_path.is_file():
        history_path = results_dir / "cnn_60_20_20_training_history.json"
    matrix_path = results_dir / "test_confusion_matrix.csv"

    if not report_path.is_file() and not summary_path.is_file():
        raise FileNotFoundError(
            f"No test_report.json or cnn_60_20_20_summary.json found in {results_dir}"
        )

    report = load_json(report_path) if report_path.is_file() else None
    summary = load_json(summary_path) if summary_path.is_file() else None
    split_summary = load_json(split_path) if split_path.is_file() else None
    if not isinstance(summary, dict):
        summary = {}
    if report is not None and not isinstance(report, dict):
        raise ValueError(f"Unexpected report format: {report_path}")
    history = pd.DataFrame(load_json(history_path)) if history_path.is_file() else pd.DataFrame()

    accuracy = report["test_accuracy"] if report else summary["test_accuracy"]
    macro_f1 = report["test_macro_f1"] if report else summary["test_macro_f1"]
    print(f"Results folder: {results_dir}")
    print(f"Split: {report.get('split', '60/20/20') if report else summary.get('split', 'not recorded')}")
    print(f"Test accuracy: {accuracy:.1%}")
    print(f"Test macro F1: {macro_f1:.1%}")
    if report:
        print(f"Test loss: {report['test_loss']:.4f}")
    if summary:
        print(
            "Images: "
            f"train={summary['train_images']}, "
            f"validation={summary['validation_images']}, "
            f"test={summary['test_images']}"
        )
    elif split_summary:
        print(
            "Images: "
            f"train={split_summary['train_images']}, "
            f"validation={split_summary['validation_images']}, "
            f"test={split_summary['test_images']}"
        )

    class_report = report.get("class_report", {}) if report else {}
    class_names = [
        name for name, scores in class_report.items()
        if isinstance(scores, dict) and "f1-score" in scores
        and name not in {"macro avg", "weighted avg"}
    ]
    if class_names:
        print("\nPer-class test metrics:")
        for name in class_names:
            scores = class_report[name]
            print(
                f"{name}: precision={scores['precision']:.1%}, "
                f"recall={scores['recall']:.1%}, "
                f"F1={scores['f1-score']:.1%}, "
                f"images={int(scores['support'])}"
            )

    fig, axes = plt.subplots(2, 2, figsize=(17, 13))
    if not history.empty:
        epochs = history["epoch"]
        axes[0, 0].plot(epochs, history["train_loss"], marker="o", label="Train")
        axes[0, 0].plot(epochs, history["val_loss"], marker="o", label="Validation")
        axes[0, 0].set(title="Loss by epoch", xlabel="Epoch", ylabel="Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.25)

        for prefix, label in (("train", "Train"), ("val", "Validation")):
            axes[0, 1].plot(
                epochs, history[f"{prefix}_accuracy"] * 100, marker="o", label=label
            )
        axes[0, 1].set(title="Accuracy by epoch", xlabel="Epoch", ylabel="Accuracy (%)")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.25)

        for prefix, label in (("train", "Train"), ("val", "Validation")):
            axes[1, 0].plot(
                epochs, history[f"{prefix}_macro_f1"] * 100, marker="o", label=label
            )
        axes[1, 0].set(title="Macro F1 by epoch", xlabel="Epoch", ylabel="Macro F1 (%)")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.25)
    else:
        axes[0, 0].text(0.5, 0.5, "Training history not available", ha="center", va="center")
        axes[0, 1].axis("off")
        axes[1, 0].axis("off")

    if matrix_path.is_file():
        matrix = pd.read_csv(matrix_path).set_index("true/predicted")
        image = axes[1, 1].imshow(matrix.values, cmap="Blues")
        axes[1, 1].set_xticks(
            range(len(matrix.columns)), matrix.columns, rotation=60, ha="right"
        )
        axes[1, 1].set_yticks(range(len(matrix.index)), matrix.index)
        axes[1, 1].set(title="Test confusion matrix", xlabel="Predicted", ylabel="Actual")
        fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.04)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                axes[1, 1].text(
                    col, row, str(matrix.iat[row, col]), ha="center", va="center"
                )
    elif class_names:
        f1_scores = [class_report[name]["f1-score"] * 100 for name in class_names]
        axes[1, 1].barh(class_names, f1_scores, color="seagreen")
        axes[1, 1].set(title="Test F1 by plant", xlabel="F1 (%)", xlim=(0, 100))
        axes[1, 1].grid(axis="x", alpha=0.25)
    else:
        metric_names = ["Test accuracy", "Test macro F1"]
        metric_values = [accuracy * 100, macro_f1 * 100]
        axes[1, 1].bar(metric_names, metric_values, color=["steelblue", "seagreen"])
        axes[1, 1].set(title="Recorded test scores", ylabel="Score (%)", ylim=(0, 100))
        axes[1, 1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    plot_path = results_dir / "results_dashboard.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"\nSaved visual dashboard: {plot_path}")
    if args.save_only:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
