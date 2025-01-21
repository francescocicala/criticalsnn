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
from tqdm import tqdm
import yaml

from src.models import SNN, SNNParameters
from src.utils import load_config
from src.snn_three_plots import plot_all_results

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
    return {
        "w_mean": params.w_mean,
        "mean_isi": snn.get_mean_isi(),
        "total_spikes": snn.get_total_spikes(),
        "lzw_complexity": lzw_complexity_from_matrix(snn.spike_matrix),
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
    for w_mean in tqdm(w_means, desc="w_mean values"):
        params.w_mean = float(w_mean)
        for _ in range(config["experiment_repetitions"]):
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
        
        # Store the SNNParameters as a YAML file
        with open(os.path.join(output_dir, "snn_parameters.yaml"), "w") as f:
            yaml.dump(config, f)
        
        logging.info("Plotting all results")
        plot_all_results(df_results, params)
        plt.savefig(os.path.join(output_dir, "simulation_all_plot.png"), dpi=300)
        logging.info("All results plot saved to %s", os.path.join(output_dir, "simulation_all_plot.png"))

    if args.cometml:
        comet_experiment.log_figure("all_results", figure=plt)
        logging.info("All results plot logged to CometML")
        comet_experiment.end()

    logging.info("Simulation completed")

if __name__ == "__main__":
    main()
