import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from src.models import SNNParameters

def init_matplotlib_style():
    """
    Initialize the matplotlib style/settings.
    """
    mpl.rcParams.update({
        "text.usetex": False,               # Disable LaTeX
        "font.family": "DeJavu Serif",             # Use serif font
        "font.serif": ["Times New Roman"],  # A LaTeX-like font
        "axes.labelsize": 26,               # Font size for axis labels
        "axes.titlesize": 26,               # Font size for titles
        "xtick.labelsize": 26,              # Font size for x-axis ticks
        "ytick.labelsize": 26,              # Font size for y-axis ticks
        "legend.fontsize": 18,              # Font size for legend
    })

def plot_all_results(simulation_df: pd.DataFrame, snn_parameters: SNNParameters):
    init_matplotlib_style()
    
    theta = snn_parameters.theta
    tau = snn_parameters.tau
    external_current = snn_parameters.external_current
    tau_ref = snn_parameters.t_ref
    leak = snn_parameters.leak
    num_neurons = snn_parameters.num_neurons

    def w_leak_free(delta):
        return (theta * delta - (external_current * delta**2) / (tau * num_neurons)) / ((num_neurons - 1) * (delta - tau_ref))

    def w_leak(delta):
        return (
            ((leak * theta * delta**2) / (1 - np.exp(-leak * delta)) - (external_current * delta**2) / (tau * num_neurons))
            / ((num_neurons - 1) * (delta - tau_ref))
        )

    # TODO: Parameterize the range of delta.
    delta = np.logspace(0.001, 3.2, 500)
    w_leak_free_values = w_leak_free(delta)
    w_leak_values = w_leak(delta)

    horizontal_line = theta / (num_neurons - 1) - 2 * external_current * tau_ref / (tau * (num_neurons - 1) * num_neurons)

    w_crit_spike = theta / (num_neurons - 1) - (2 * external_current * tau_ref) / (tau * num_neurons * (num_neurons - 1))

    w_mean_unique_values = simulation_df["w_mean"].unique()

    grouped_spike = simulation_df.groupby("w_mean")
    nspike_median = grouped_spike["total_spikes"].median().values
    nspike_q1 = grouped_spike["total_spikes"].quantile(0.25).values
    nspike_q3 = grouped_spike["total_spikes"].quantile(0.75).values

    w_crit_lzw = theta / (num_neurons - 1) - (2 * external_current * tau_ref) / (tau * num_neurons * (num_neurons - 1))

    grouped_lzw = simulation_df.groupby('w_mean')
    median_values_complexity = grouped_lzw['lzw_complexity'].median().values
    q1_values_complexity = grouped_lzw['lzw_complexity'].quantile(0.25).values
    q3_values_complexity = grouped_lzw['lzw_complexity'].quantile(0.75).values

    max_complexity = median_values_complexity.max()
    median_complexity = median_values_complexity / max_complexity
    q1_complexity = q1_values_complexity / max_complexity
    q3_complexity = q3_values_complexity / max_complexity

    fig, axs = plt.subplots(2, 2, figsize=(24, 16), constrained_layout=True)

    # Top-left plot
    axs[0, 0].plot(delta, w_leak_free_values, label=r'$\langle W \rangle(\langle \delta\rangle)_{\text{leak-free}}$', color='purple', linewidth=5)
    axs[0, 0].axhline(horizontal_line, color='purple', linestyle='--', linewidth=3, label=r"$\langle W \rangle_{\text{critical}}$")
    axs[0, 0].plot(delta, w_leak_values, label=r'$\langle W \rangle(\langle \delta\rangle)_{\text{leak}}$ with $\alpha_1$', color='green', linewidth=5)
    if leak > 0:
        delta_alpha = 0.2 / leak if leak != 0 else np.inf
        axs[0, 0].axvline(delta_alpha, color='green', linestyle=':', linewidth=3, label=r'$\alpha_1\langle \delta\rangle=0.2$')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_yscale('log')
    axs[0, 0].set_xlabel(r'$\langle \delta \rangle$')
    axs[0, 0].set_ylabel(r'$\langle W \rangle$')
    axs[0, 0].legend(loc="lower left")

    # Top-right plot
    axs[0, 1].plot(w_mean_unique_values, nspike_median, color='black', linewidth=5, label=r"Num. Spikes (synthetic SNN)")
    axs[0, 1].fill_between(w_mean_unique_values, nspike_q1, nspike_q3, color='black', alpha=0.2)
    axs[0, 1].axvline(x=w_crit_spike, color='purple', linestyle='--', linewidth=5, label=r"$\langle W \rangle_{\text{critical}}$")
    axs[0, 1].set_xlim(0.011, 0.049)
    axs[0, 1].set_xlabel(r'$\langle W \rangle$')
    axs[0, 1].set_ylabel(r'Number of Spikes')

    # Configure y-axis to use scientific notation
    axs[0, 1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axs[0, 1].yaxis.get_major_formatter().set_scientific(True)
    axs[0, 1].yaxis.get_major_formatter().set_powerlimits((-1, 1))

    axs[0, 1].legend(loc="upper left")

    # Bottom-left plot
    axs[1, 0].plot(w_mean_unique_values, median_complexity, color='black', linewidth=5,label=r"LZW complexity (synthetic SNN)")
    axs[1, 0].fill_between(w_mean_unique_values, q1_complexity, q3_complexity, color='black', alpha=0.2)
    axs[1, 0].axvline(x=w_crit_lzw, color='purple', linestyle='--', linewidth=5, label=r"$\langle W \rangle_{\text{critical}}$")
    axs[1, 0].set_xlim(0.011, 0.049)
    axs[1, 0].set_xlabel(r'$\langle W \rangle$')
    axs[1, 0].set_ylabel(r'Normalized LZW Complexity')
    axs[1, 0].legend(loc="upper left")

    # Bottom-right plot
    axs[1, 1].axis('off')  # Disattiva l'asse per lo spazio vuoto
