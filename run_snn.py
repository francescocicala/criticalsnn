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

from src.models import SNN, SNNParameters
from src.utils import load_config
from src.snn_plots import plot_isi_results

def lzw_complexity_from_matrix(matrix: np.ndarray) -> int:
    """
    Calculate the Lempel-Ziv-Welch (LZW) complexity of a vector created by
    concatenating the columns of a 2D matrix.

    Steps:
        1. Transpose the matrix to read it column by column.
        2. Flatten the transposed matrix into a 1D vector.
        3. Convert the numeric vector into a string.
        4. Compute the LZW complexity of the string.

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
    """
    Run a simulation for a Spiking Neural Network and compute relevant metrics.

    Args:
        params (SNNParameters): An instance containing the SNN settings (e.g., num_neurons, threshold).

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'w_mean': The mean synaptic weight used in this simulation.
            - 'mean_isi': The mean inter-spike interval of the network.
            - 'total_spikes': Total spikes observed in the simulation.
            - 'lzw': The LZW complexity of the spike matrix.
    """
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
    """
    Main function to parse arguments, run the simulation(s), and save outputs.
    
    Steps:
        1. Parse command-line arguments.
        2. Load configuration from YAML.
        3. Set RNG seeds for reproducibility.
        4. Create an output directory with timestamp.
        5. Loop over a range of w_mean values and run multiple simulations.
        6. Save results and plot ISI results.
    """
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
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)
    params = SNNParameters(**config['snn_params'])

    if args.cometml:
        load_dotenv()
        comet_experiment = comet_ml.Experiment(
                    api_key=os.environ.get("COMETML_API_KEY"),
                    project_name=os.environ.get("COMETML_PROJECT"),
                    workspace=os.environ.get("COMETML_WORKSPACE")
                )

    # Set seed for reproducibility
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    # Create output directory with a timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Run simulations over a range of w_mean values
    w_means = np.arange(
        config["w_means_range_min"],
        config["w_means_range_max"],
        config["w_means_range_step"]
    )

    results: List[Dict[str, Any]] = []
    for w_mean in w_means:
        params.w_mean = float(w_mean)
        for _ in range(config["experiment_repetitions"]):
            results.append(run_simulation(params))

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, "simulation_results.csv"), index=False)

    # Plot ISI results and save the figure
    plot_isi_results(
        df_results,
        params.num_neurons,
        params.theta,
        params.tau,
        params.external_current,
        params.t_ref
    )
    plt.savefig(os.path.join(output_dir, "simulation_isi_plot.png"), dpi=300)

    if args.cometml:
        comet_experiment.log_figure("isi_vs_w_mean", figure=plt)
        comet_experiment.end()

if __name__ == "__main__":
    main()
