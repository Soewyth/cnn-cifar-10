from cnn_cifar_10.config.paths import get_paths, get_root_dir
from cnn_cifar_10.evaluation.compare import load_all_metrics, generate_comparison_report


def main():
    # Load config and get paths
    root_dir = get_root_dir()
    paths = get_paths(root_dir=root_dir)
    path_model = paths["models"]
    path_report = paths["reports"]
    # Load results from different runs
    all_metrics = load_all_metrics(models_dir=path_model)
    # Generate comparison report
    generate_comparison_report(
        all_metrics, save_path=path_report / "comparison_report.md"
    )

    print(" All metrics load from :", path_model)
    print(" Comparison report saved to :", path_report / "comparison_report.md")


if __name__ == "__main__":
    main()
