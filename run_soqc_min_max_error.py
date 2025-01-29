"""Simulate a Spiking Neural Network (SNN) to observe the minimum and maximum error ratios observed during the second half of the simulation."""

import logging
import random
import numpy as np

from src.models_for_SOqC import SNN, SoqcParams
from src.utils import load_config, program_parameters_reader

import numpy as np
import matplotlib.pyplot as plt
import os

# Configure logging to include INFO level and above
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    """Run simulation for a range of w_mean values, save the results and generate plots."""

    args = program_parameters_reader("soqc_config_v2.yaml")

    logging.info("Loading configuration from %s", args.config)
    config = load_config(args.config)
    simulation_params = SoqcParams(**config["simulation_params"])

    # Set seed for reproducibility
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    logging.info(f"Running simulation with parameters: {simulation_params}")

    network = SNN(simulation_params, True)
    theoretical_critical_weight = network.theoretical_critical_weight()
    weights_history = network.simulate()
    max_error = weights_history[0]
    min_error = weights_history[0]
    
    cnt = 0
    for weight in weights_history:
        cnt += 1
        error = np.abs(weight - theoretical_critical_weight) / theoretical_critical_weight
        logging.info(f"error ratio: {error} at step {cnt}")
        if error > max_error:
            max_error = error
        if error < min_error:
            min_error = error

    logging.info(f"theoretical critical synaptic mean weight: {theoretical_critical_weight}")
    logging.info(f"max error: {max_error}")
    logging.info(f"min error: {min_error}")


    logging.info("Simulation completed")


if __name__ == "__main__":
    main()
