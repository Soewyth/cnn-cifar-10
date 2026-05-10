from cnn_cifar_10.config.paths import get_paths


def test_paths_create_and_exist(tmp_path):
    paths = get_paths(root_dir=tmp_path)

    for path_name, path_value in paths.items():
        print(f"Testing path: {path_name} -> {path_value}")
        assert path_value.exists(), f"Path '{path_name}' does not exist: {path_value}"
