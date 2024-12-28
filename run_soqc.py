"""
Script to solve a nonlinear equation for W across a range of parameters,
compare with a theoretical critical value, record relative errors,
and save results to a CSV file. The parameters are loaded from a YAML
configuration file using `load_config`.
"""

import os
import numpy as np
from mpmath import mp, mpf
import argparse
import time
import pandas as pd
import random

# Import our config loader
from src.utils import load_config


def solve_w(w0, w1, alpha_value, tau, num_neurons, threshold, external_current, refractory_time):
    """
    Solve the nonlinear equation for W using mpmath's findroot (Newton-Raphson).

    Parameters
    ----------
    w0 : float
        Initial W value.
    w1 : float
        Maximum W value.
    alpha_value : float
        Exponential decay parameter.
    tau : float
        Characteristic time constant.
    num_neurons : int
        Number of neurons (N).
    threshold : float
        Threshold value (theta).
    external_current : float
        External current (I).
    refractory_time : float
        Refractory time (tau_ref).

    Returns
    -------
    tuple
        (solution, max_estimated_error)
        solution : mp.mpf
            The solution for W obtained from the numeric solver.
        max_estimated_error : mp.mpf
            An estimate of the maximum error in the solution.
    """

    # Convert parameters to multiprecision
    w0_mp = mpf(w0)
    w1_mp = mpf(w1)
    alpha_mp = mpf(alpha_value)
    tau_mp = mpf(tau)
    n_mp = mpf(num_neurons)
    theta_mp = mpf(threshold)
    i_mp = mpf(external_current)
    tau_ref_mp = mpf(refractory_time)

    # Define the equation for W
    def equation_w_mpmath(w_value):
        """
        The function to be solved: w0 + (w1 - w0)*exp(-alpha*Delta) - w = 0
        where Delta depends on w_value.
        """
        delta = (tau_mp * n_mp / (2 * i_mp)) * (
            (theta_mp - w_value * (n_mp - 1)) +
            mp.sqrt((theta_mp - w_value * (n_mp - 1))**2 +
                    (4 * i_mp * w_value * (n_mp - 1) * tau_ref_mp) / (n_mp * tau_mp))
        )
        return w0_mp + (w1_mp - w0_mp) * mp.e**(-alpha_mp * delta) - w_value

    # Initial guess
    initial_guess = (
        theta_mp / (n_mp - 1)
        - 2 * i_mp * tau_ref_mp / (tau_mp * n_mp * (n_mp - 1))
    )

    # Solve using Newton's method
    solution = mp.findroot(equation_w_mpmath, initial_guess, solver="newton", tol=mp.mpf('1e-30'))

    # Calculate the residual
    residual = abs(equation_w_mpmath(solution))

    # Estimate derivative near the solution
    epsilon = mp.mpf('1e-10')
    derivative_val = abs(
        (equation_w_mpmath(solution + epsilon) - equation_w_mpmath(solution)) / epsilon
    )

    # Protect against a near-zero derivative
    min_derivative_threshold = mp.mpf('1e-10')
    effective_derivative = max(derivative_val, min_derivative_threshold)

    # Estimate maximum error
    max_estimated_error = residual / effective_derivative

    # Optional diagnostic checks
    if residual > mp.mpf('2e-10'):
        print("Warning: High residual detected:", residual)

    if derivative_val < mp.mpf('0.9'):
        print("Warning: Low derivative detected:", derivative_val)

    return solution, max_estimated_error


def main():
    """
    Main function to:
      1. Load config values from 'soqc_config.yaml'.
      2. Loop over two ranges of external_current (I) and a range of N.
      3. Solve W numerically and compare with the theoretical critical value.
      4. Track min/max relative errors.
      5. Save results to a CSV file.
    """

    parser = argparse.ArgumentParser(description="SOqC.")
    parser.add_argument("--config", type=str, default="soqc_config.yaml", help="Path to the YAML configuration file.")
    parser.add_argument("--output", type=str, default="results/soqc", help="Optional output directory name.")
    args = parser.parse_args()

    # Load configuration from YAML
    config = load_config(args.config)

    # Set seed for reproducibility.
    np.random.seed(config["seed"])
    random.seed(config["seed"])

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

    # Prepare for storing results
    min_error_percent = 1e9
    max_error_percent = 0.0
    results = []

    # --------------------------------------
    # First loop: external_current from 0.01 to 0.99
    # via i_small_range in config (1..99 => 0.01..0.99)
    # --------------------------------------
    for external_current in np.linspace(i_small_min, i_small_max, i_small_n_steps):
        print(f"Solving for external_current = {external_current:.2f} ...")

        for num_neurons in range(n_min, n_max, n_step):
            w_solution, _ = solve_w(w0, w1, alpha_value, tau,
                                    num_neurons, threshold,
                                    external_current, refractory_time)

            # Theoretical critical value for W
            w_theoretical_crit = (
                threshold / (num_neurons - 1)
                - 2 * external_current * refractory_time
                / (tau * num_neurons * (num_neurons - 1))
            )

            # Relative error (%)
            relative_error_percent = (
                abs(w_solution - w_theoretical_crit) 
                / abs(w_theoretical_crit)
                * 100.0
            )

            results.append({
                'N': num_neurons,
                'I': external_current,
                'relative_error_percent': float(relative_error_percent)
            })

            # Track min/max across all parameter sets
            if relative_error_percent < min_error_percent:
                min_error_percent = relative_error_percent
            if relative_error_percent > max_error_percent:
                max_error_percent = relative_error_percent

    # --------------------------------------
    # Second loop: external_current as an integer from 1 to 99
    # via i_large_range in config
    # --------------------------------------
    for external_current in np.linspace(i_large_min, i_large_max, i_large_n_steps):
        print(f"Solving for external_current = {external_current:.0f} ...")

        for num_neurons in range(n_min, n_max, n_step):
            w_solution, _ = solve_w(w0, w1, alpha_value, tau,
                                    num_neurons, threshold,
                                    external_current, refractory_time)

            # Theoretical critical value for W
            w_theoretical_crit = (
                threshold / (num_neurons - 1)
                - 2 * external_current * refractory_time
                / (tau * num_neurons * (num_neurons - 1))
            )

            relative_error_percent = (
                abs(w_solution - w_theoretical_crit)
                / abs(w_theoretical_crit)
                * 100.0
            )

            results.append({
                'N': num_neurons,
                'I': external_current,
                'relative_error_percent': float(relative_error_percent)
            })

            # Track min/max errors
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

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "soqc_results.csv"), index=False)

if __name__ == "__main__":
    main()
