import os
import numpy as np
import argparse
import time
import pandas as pd

from src.utils import load_config
from src.soqc import SolverParams, solve_w, compute_w_theoretical_crit

def main():
    parser = argparse.ArgumentParser(description="SOqC.")
    parser.add_argument("--config", type=str, default="soqc_config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--output", type=str, default="results/soqc", help="Optional output directory name.")
    args = parser.parse_args()

    # Load configuration from YAML
    config = load_config(args.config)

    # Pull out parameters from config
    tau = config['tau']
    threshold = config['threshold']
    refractory_time = config['refractory_time']
    w0 = config['w0']
    w1 = config['w1']
    alpha_value = config['alpha_value']

    i_small_min = config['i_small_range']['min']
    i_small_max = config['i_small_range']['max']
    i_small_n_steps = config['i_small_range']['n_steps']

    i_large_min = config['i_large_range']['min']
    i_large_max = config['i_large_range']['max']
    i_large_n_steps = config['i_large_range']['n_steps']

    n_min = config['n_range']['min']
    n_max = config['n_range']['max']
    n_step = config['n_range']['step']

    # Create the arrays for small and large currents
    i_small_values = np.linspace(i_small_min, i_small_max, i_small_n_steps)
    i_large_values = np.linspace(i_large_min, i_large_max, i_large_n_steps)
    # Combine them into one, ensuring small currents come first
    i_all_values = np.concatenate([i_small_values, i_large_values])

    min_error_percent = 1e9
    max_error_percent = 0.0
    results = []

    # Single loop over the combined i_all_values
    for external_current in i_all_values:
        # We still want a different print format for small vs. large
        if external_current < 1:
            print(f"Solving for external_current = {external_current:.2f} ...")
        else:
            print(f"Solving for external_current = {external_current:.0f} ...")

        # The same loop over num_neurons
        for num_neurons in range(n_min, n_max, n_step):
            params = SolverParams(
                w0=w0,
                w1=w1,
                alpha_value=alpha_value,
                tau=tau,
                num_neurons=num_neurons,
                threshold=threshold,
                external_current=external_current,
                refractory_time=refractory_time
            )

            w_solution, _ = solve_w(params)

            w_theoretical_crit = compute_w_theoretical_crit(
                num_neurons, threshold, external_current, refractory_time, tau
            )

            relative_error_percent = abs(w_solution - w_theoretical_crit) / abs(w_theoretical_crit) * 100.0
            results.append({
                'N': num_neurons,
                'I': external_current,
                'relative_error_percent': float(relative_error_percent)
            })

            if relative_error_percent < min_error_percent:
                min_error_percent = relative_error_percent
            if relative_error_percent > max_error_percent:
                max_error_percent = relative_error_percent

    # Print summary
    print(f"\nMinimum error (%): {min_error_percent}")
    print(f"Maximum error (%): {max_error_percent}\n")

    # Create output directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "soqc_results.csv"), index=False)

if __name__ == "__main__":
    main()
