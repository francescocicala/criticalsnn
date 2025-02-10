"""Simulate a Spiking Neural Network (SNN) to observe errors on a parameter grid."""

import logging
import random
import numpy as np
import os

from src.soqc import SNN, SoqcParams
from soqc_plot import analyze_csv_and_plot
from src.utils import load_config

# Configure logging to include INFO level and above
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def save_results_to_csv(results, output_directory, filename):
    """
    Saves results into sub folder SOqC (created if doesn't exist) in filename
    """

    folder_name = output_directory
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

def main():
    """Run simulation for a range of w_mean values, save the results, and generate plots."""

    ITERATIONS = 4
    
    parser = argparse.ArgumentParser(description="Run simulation.")
    parser.add_argument(
        "--config",
        type=str,
        default="soqc_config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/run_soqc",
        help="Optional output directory name.",
    )
    
    args = parser.parse_args()

    logging.info("Loading configuration from %s", args.config)

    config = load_config(args.config)

    # Set seed for reproducibility
    # np.random.seed(config["seed"])
    # random.seed(config["seed"])

    external_current_values = np.linspace(
        config["external_current_start"],
        config["external_current_end"],
        config["number_of_current_samples"],
    )
    membrane_threshold_values = np.linspace(
        config["membrane_threshold_start"],
        config["membrane_threshold_end"],
        config["number_of_threshold_samples"],
    )

    results = []
    for iteration in range(ITERATIONS):
        print("Iteration: ", iteration)
        for external_current in external_current_values:
            for membrane_threshold in membrane_threshold_values:
                config["simulation_params"]["external_current"] = external_current
                config["simulation_params"]["membrane_threshold"] = membrane_threshold
                simulation_params = SoqcParams(**config["simulation_params"])

                logging.info(f"Running simulation with parameters: {simulation_params}")

                network = SNN(simulation_params)
                observed_critical_synaptic_mean_weight = network.simulate()
                theoretical_critical_synaptic_mean_weight = network.theoretical_critical_weight()

                result = (
                    iteration,
                    simulation_params,
                    observed_critical_synaptic_mean_weight,
                    theoretical_critical_synaptic_mean_weight,
                    np.abs(
                        observed_critical_synaptic_mean_weight
                        - theoretical_critical_synaptic_mean_weight
                    )
                    / theoretical_critical_synaptic_mean_weight,
                )
                results.append(result)
                print("Error: ", result[-1])
                logging.info("Simulation completed")

    save_results_to_csv(results, args.output,"soqc_results.csv")
    analyze_csv_and_plot("soqc_results_10_iteration.csv")


if __name__ == "__main__":
    main()
