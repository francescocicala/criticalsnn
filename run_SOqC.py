"""Simulate a Spiking Neural Network (SNN) to reproduce Self Organized quasi-Criticality (SOqC)."""

import logging
import random
import numpy as np

from src.models_for_SOqC import SNN, SimulationParams
from src.utils import load_config, program_parameters_reader

# Configure logging to include INFO level and above
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    """Run simulation for a range of w_mean values, save the results and generate plots."""

    args = program_parameters_reader()

    logging.info("Loading configuration from %s", args.config)
    config = load_config(args.config)
    simulation_params = SimulationParams(**config["simulation_params"])
    simulation_params.validate_conditions()

    # Set seed for reproducibility
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    logging.info(f"Running simulation with parameters: {simulation_params}")

    network = SNN(simulation_params)
    print("estimated critical synaptic mean weight: ", network.simulate())
    print("theoretical critical synaptic mean weight: ", network.theoretical_critical_weight())

    logging.info("Simulation completed")


if __name__ == "__main__":
    main()
