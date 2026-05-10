from cnn_cifar_10.config.paths import get_paths, get_root_dir
from cnn_cifar_10.evaluation.compare import load_all_metrics, generate_comparison_report
from cnn_cifar_10.io.run_id import get_run_id


def main():
    # Load config and get paths
    root_dir = get_root_dir()
    paths = get_paths(root_dir=root_dir)
    path_model = paths["models"]
    path_report = paths["reports"]

    # get current run id
    run_id = get_run_id()
    # Load results from different runs
    all_metrics = load_all_metrics(models_dir=path_model)
    # Generate comparison report
    generate_comparison_report(
        all_metrics, save_path=path_report / f"comparison_report_{run_id}.md"
    )

    print(" All metrics load from :", path_model)
    print(
        " Comparison report saved to :", path_report / f"comparison_report_{run_id}.md"
    )


if __name__ == "__main__":
    main()
