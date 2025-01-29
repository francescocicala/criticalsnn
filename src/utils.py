import yaml
from typing import Any, Dict
import argparse
import os
import pandas as pd

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

def program_parameters_reader(file_name:str):
    parser = argparse.ArgumentParser(description="Run simulation.")
    parser.add_argument(
        "--config",
        type=str,
        default=file_name,
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

def save_results_to_csv(results, filename):
    """
    Saves results into sub folder SOqC (created if doesn't exist) in filename
    """

    folder_name = "SOqC"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    data = []
    
    for itereation, sim_params, observed_critical_synaptic_mean_weight, theoretical_critical_synaptic_mean_weight, error in results:
        data.append({
            "iteration": itereation,
            "num_neurons": sim_params.num_neurons,
            "membrane_threshold": sim_params.membrane_threshold,
            "currents_period": sim_params.currents_period,
            "external_current": sim_params.external_current,
            "number_of_external_current": sim_params.number_of_external_current,
            "leak_coefficient": sim_params.leak_coefficient,
            "simulation_duration": sim_params.simulation_duration,
            "refractory_period": sim_params.refractory_period,
            "guessed_critical_weight": sim_params.guessed_critical_weight,
            "observed_critical_synaptic_mean_weight": observed_critical_synaptic_mean_weight,
            "theoretical_critical_synaptic_mean_weight": theoretical_critical_synaptic_mean_weight,
            "error": error
        })                
    
    df = pd.DataFrame(data)
    
    file_path = os.path.join(folder_name, filename)
    
    df.to_csv(file_path, index=False)
    print(f"CSV saved as {file_path}.")

