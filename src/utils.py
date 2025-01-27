import yaml
from typing import Any, Dict
import argparse

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Loads configuration data from a specified YAML file.

    Args:
        config_file (str): The file path to the YAML configuration.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.
    """
    with open(config_file, 'r', encoding='utf-8') as file:
        config: Dict[str, Any] = yaml.safe_load(file)
    return config

def program_parameters_reader():
    parser = argparse.ArgumentParser(description="Run simulation.")
    parser.add_argument(
        "--config",
        type=str,
        default="SOqC config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/snn_v2",
        help="Optional output directory name.",
    )
    
    args = parser.parse_args()
    return args