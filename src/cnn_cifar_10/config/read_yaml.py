import yaml


def read_config_yaml(config_path: str) -> dict:
    """Read the configuration file in yaml format
    and return a dictionary of the configuration parameters
    Args:
        config_path (str): The path to the configuration file
    Returns:
        dict: The configuration parameters"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
