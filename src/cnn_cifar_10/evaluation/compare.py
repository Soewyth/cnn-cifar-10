import json
from pathlib import Path


def load_all_metrics(models_dir: Path) -> list[dict]:
    """Load all metrics from the models directory.
    Args:
        models_dir (Path): The directory containing the model with metrics.json files.
    Returns:
        list[dict]: A list of dictionaries containing the metrics for each model.
    """
    all_metrics = []

    for metrics_file in sorted(models_dir.glob("metrics_*.json")):
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        metrics["source_file"] = str(metrics_file.name)
        all_metrics.append(metrics)

    return all_metrics


def generate_comparison_report(metric_list: list[dict], save_path: Path) -> None:
    """Generate a comparison report from a list of metrics dictionaries.
    Args:
        metric_list (list[dict]): A list of dictionaries containing the metrics for each model.
        save_path (Path): The path to save the comparison report.
    Returns:
       None
    """
    if not metric_list:
        print("No metrics to compare.")
        return

    rows = []
    for metrics in metric_list:
        config = metrics.get("config", {})
        training_config = config.get("training", {})
        model_config = config.get("model", {})
        results = metrics.get("results", {})

        run_id = metrics.get("datetime", "unknown_run_id")
        source_file = metrics.get("source_file", "unknown_source_file")
        # Handle missing keys with .get() and provide default values
        rows.append(
            {
                "run_id": run_id,
                "source_file": source_file,
                "architecture": model_config.get("architecture", "N/A"),
                "pretrained": model_config.get("pretrained", "N/A"),
                "freeze_backbone": model_config.get("freeze_backbone", "N/A"),
                "input_size": model_config.get("input_size", "N/A"),
                "batch_size": training_config.get("batch_size", "N/A"),
                "learning_rate": training_config.get("learning_rate", "N/A"),
                "optimizer": training_config.get("optimizer", "N/A"),
                "best_accuracy": results.get("best_accuracy", None),
                "epoch_runs": results.get("epoch_runs", None),
                "average_train_loss": results.get("average_train_loss", None),
                "average_test_loss": results.get("average_test_loss", None),
                "average_test_accuracies": results.get("average_test_accuracies", None),
                "training_time_seconds": results.get("training_time_seconds", None),
            }
        )
    # Sort rows by best accuracy in descending order (handle None values)
    rows = sorted(
        rows,
        key=lambda row: (
            row["best_accuracy"] if row["best_accuracy"] is not None else -1
        ),
        reverse=True,
    )

    best_run = rows[0]

    lines = []
    lines.append("# Runs comparison")
    lines.append("")
    lines.append(
        "This report compares all training metrics found in `outputs/models/`."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Best run: `{best_run['run_id']}` with "
        f"**{best_run['best_accuracy']:.4f}** best accuracy."
        if isinstance(best_run["best_accuracy"], (int, float))
        else f"- Best run: `{best_run['run_id']}`."
    )
    lines.append(
        f"- Architecture: `{best_run['architecture']}`"
        f", freeze_backbone=`{best_run['freeze_backbone']}`"
        f", input_size=`{best_run['input_size']}`"
        f", batch_size=`{best_run['batch_size']}`"
        f", lr=`{best_run['learning_rate']}`."
    )
    lines.append("")

    lines.append("## Comparison table")
    lines.append("")
    lines.append(
        "| Rank | Run ID | Architecture | Pretrained | Freeze backbone | Input size | Batch size | LR | Optimizer | Best acc | Avg train loss | Avg test loss | Avg test acc | Time (s) | Epochs |"
    )
    lines.append(
        "|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"
    )

    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| "
            f"{rank} | "
            f"`{row['run_id']}` | "
            f"{row['architecture']} | "
            f"{row['pretrained']} | "
            f"{row['freeze_backbone']} | "
            f"{row['input_size']} | "
            f"{row['batch_size']} | "
            f"{row['learning_rate']} | "
            f"{row['optimizer']} | "
            f"{float(row['best_accuracy'])} | "
            f"{float(row['average_train_loss']):.4f} | "
            f"{float(row['average_test_loss']):.4f} | "
            f"{float(row['average_test_accuracies']):.4f} | "
            f"{float(row['training_time_seconds']):.2f} | "
            f"{row['epoch_runs']} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `best_accuracy` is the best evaluation accuracy reached during training."
    )
    lines.append(
        "- Average losses and accuracies are computed across the epochs actually run."
    )
    lines.append("- Runs are sorted by descending best accuracy.")
    lines.append("")

    save_path.write_text("\n".join(lines), encoding="utf-8")
