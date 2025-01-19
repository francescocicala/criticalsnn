import os
import time
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from dotenv import load_dotenv
import comet_ml
import logging

from src.models import SNN, SNNParameters
from src.utils import load_config
from src.snn_plots import plot_isi_results, plot_lzw_median

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def lzw_complexity_from_matrix(matrix: np.ndarray) -> int:
    """
    Calculate the Lempel-Ziv-Welch (LZW) complexity of a vector created by
    concatenating the columns of a 2D matrix.
    Args:
        matrix (np.ndarray): A 2D NumPy array representing spike data
                             (rows typically time, columns neurons).

    Returns:
        int: The LZW complexity of the concatenated sequence.
    """
    def lzw(seq: str) -> int:
        """
        Calculate the LZW complexity of a binary (string) sequence.

        Args:
            seq (str): The sequence string (e.g., '101001...').

        Returns:
            int: The size of the generated dictionary, representing
                 the LZW complexity.
        """
        dictionary = {}
        w = ""
        for c in seq:
            wc = w + c
            if wc not in dictionary:
                dictionary[wc] = len(dictionary)
                w = c
            else:
                w = wc
        return len(dictionary)

    # Transpose and then flatten to read column by column
    vector = matrix.T.flatten()
    # Convert the vector into a string
    vector_str = "".join(map(str, vector))

    # Calculate the LZW complexity
    complexity = lzw(vector_str)
    return complexity

def run_simulation(params: SNNParameters) -> Dict[str, Any]:
    snn = SNN(params)
    snn.simulate()
    mean_isi = snn.get_mean_isi()
    complexity = lzw_complexity_from_matrix(snn.spike_matrix)
    return {
        "w_mean": params.w_mean,
        "mean_isi": mean_isi,
        "total_spikes": snn.get_total_spikes(),
        "lzw": complexity,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulation.")
    parser.add_argument(
        "--config",
        type=str,
        default="snn_config.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/snn",
        help="Optional output directory name."
    )
    parser.add_argument(
        "--cometml",
        action="store_true",
        default=False,
        help="Enable CometML integration."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="If enabled, doesn't create a folder with results."
    )
    
    args = parser.parse_args()

    if args.dry_run:
        logging.info("Dry run enabled. No output will be saved.")

    logging.info("Loading configuration from %s", args.config)
    config = load_config(args.config)
    params = SNNParameters(**config['snn_params'])

    if args.cometml:
        load_dotenv()
        comet_experiment = comet_ml.Experiment(
            api_key=os.environ.get("COMETML_API_KEY"),
            project_name=os.environ.get("COMETML_PROJECT"),
            workspace=os.environ.get("COMETML_WORKSPACE")
        )
        comet_experiment.log_parameters(config)
        logging.info("CometML integration enabled")

    # Set seed for reproducibility
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    logging.info("Running simulations over a range of w_mean values")
    w_means = np.arange(
        config["w_means_range_min"],
        config["w_means_range_max"],
        config["w_means_range_step"]
    )

    results: List[Dict[str, Any]] = []
    for w_mean in w_means:
        params.w_mean = float(w_mean)
        for _ in range(config["experiment_repetitions"]):
            logging.info("Running simulation with w_mean = %f", w_mean)
            results.append(run_simulation(params))

    df_results = pd.DataFrame(results)
    
    if not args.dry_run:
        # Create output directory with a timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(args.output, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Saving results to %s", output_dir)

        # Save results
        df_results.to_csv(os.path.join(output_dir, "simulation_results.csv"), index=False)

    # Plot ISI results
    logging.info("Plotting ISI results")
    plot_isi_results(
        df_results,
        params.num_neurons,
        params.theta,
        params.tau,
        params.external_current,
        params.t_ref
    )
    if not args.dry_run:
        # Save ISI plot
        plt.savefig(os.path.join(output_dir, "simulation_isi_plot.png"), dpi=300)

    # Plot LZW complexity results
    logging.info("Plotting LZW complexity results")
    plot_lzw_median(
        df_results,
        params.theta,
        params.tau,
        params.external_current,
        params.t_ref,
        params.num_neurons
    )
    if not args.dry_run:
        # Save LZW complexity plot
        plt.savefig(os.path.join(output_dir, "simulation_lzw_plot.png"), dpi=300)

    if args.cometml:
        comet_experiment.log_figure("isi_vs_w_mean", figure=plt)
        comet_experiment.log_figure("lzw_vs_w_mean", figure=plt)
        comet_experiment.end()
        logging.info("CometML figures logged")

    logging.info("Simulation completed")

if __name__ == "__main__":
    main()
