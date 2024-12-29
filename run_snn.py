import numpy as np
import random
import pandas as pd
import os
import argparse
import time
import matplotlib.pyplot as plt
from src.models import SNN, SNNParameters
from src.utils import load_config
from src.snn_plots import plot_isi_results

def lzw_complexity_from_matrix(matrix):
    """
    Calculate the Lempel-Ziv (LZW) complexity of a vector created by concatenating the columns of a matrix.
    """
    def lzw(seq):
        """Calculate the LZW complexity of a binary sequence."""
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

    # Transform the matrix into a vector by concatenating its columns
    vector = matrix.T.flatten()  # Transpose and then flatten to read column by column
    vector_str = "".join(map(str, vector))  # Convert the vector into a string

    # Calculate the LZW complexity of the concatenated vector
    complexity = lzw(vector_str)
    return complexity

def run_simulation(params):
    """Runs the ISI simulation."""
    snn = SNN(params)
    snn.simulate()
    mean_isi = snn.get_mean_isi()
    lzw = lzw_complexity_from_matrix(snn.spike_matrix)
    return {"w_mean": params.w_mean, "mean_isi": mean_isi, "total_spikes": snn.get_total_spikes(), "lzw": lzw}

def main():
    parser = argparse.ArgumentParser(description="Run simulation.")
    parser.add_argument("--config", type=str, default="snn_config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--output", type=str, default="results/snn", help="Optional output directory name.")
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)
    params = SNNParameters(**config['snn_params'])

    # Set seed for reproducibility.
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    # Create output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    w_means = np.arange(config["w_means_range_min"],
                        config["w_means_range_max"],
                        config["w_means_range_step"])
    results = []
    for w_mean in w_means:
        params.w_mean = float(w_mean)
        for _ in range(config["experiment_repetitions"]):
            results.append(run_simulation(params))
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir, "simulation_results.csv"), index=False)
    plot_isi_results(df_results, params.num_neurons, params.theta, params.tau, params.external_current, params.t_ref)
    plt.savefig(os.path.join(output_dir, "simulation_isi_plot.png"), dpi=300)

if __name__ == "__main__":
    main()