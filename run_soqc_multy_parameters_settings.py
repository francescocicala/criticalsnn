"""Simulate a Spiking Neural Network (SNN) to observe errors on a parameter grid."""

import logging
import random
import numpy as np

from src.models_for_soqc import SNN, SoqcParams
from src.utils import load_config, program_parameters_reader, save_results_to_csv

# Configure logging to include INFO level and above
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """Run simulation for a range of w_mean values, save the results, and generate plots."""

    ITERATIONS = 2

    args = program_parameters_reader("soqc_multi_parameters.yaml")

    logging.info("Loading configuration from %s", args.config)

    config = load_config(args.config)

    # Set seed for reproducibility
    np.random.seed(config["seed"])
    random.seed(config["seed"])

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

    save_results_to_csv(results, "soqc_results.csv")


if __name__ == "__main__":
    main()
