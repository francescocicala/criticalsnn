import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from src.models_v2 import SimulationParams


def init_matplotlib_style():
    """
    Initialize the matplotlib style/settings.
    """
    mpl.rcParams.update(
        {
            "text.usetex": False,  # Disable LaTeX
            "font.family": "DeJavu Serif",  # Use serif font
            "font.serif": ["Times New Roman"],  # A LaTeX-like font
            "axes.labelsize": 26,  # Font size for axis labels
            "axes.titlesize": 26,  # Font size for titles
            "xtick.labelsize": 26,  # Font size for x-axis ticks
            "ytick.labelsize": 26,  # Font size for y-axis ticks
            "legend.fontsize": 18,  # Font size for legend
        }
    )


def plot_all_results(simulation_df: pd.DataFrame, simulation_params: SimulationParams):
    """Generate three plots: (1) Leak-free vs leak <W>(<ISI>) comparison, (2) number of spikes, and (3) LZW complexity."""

    # Raise error if any of "w_mean", "total_spikes", or "lzw_complexity" columns are missing
    if not all(
        col in simulation_df.columns
        for col in ["w_mean", "total_spikes", "lzw_complexity"]
    ):
        raise ValueError(
            "Missing columns in simulation_df: 'w_mean', 'total_spikes', 'lzw_complexity'"
        )
    init_matplotlib_style()

    membrane_threshold = simulation_params.membrane_threshold
    currents_period = simulation_params.currents_period
    external_current = simulation_params.external_current
    refractory_period = simulation_params.refractory_period
    leak_coefficient = simulation_params.leak_coefficient
    num_neurons = simulation_params.num_neurons

    def w_leak_free(delta):
        return (
            membrane_threshold * delta
            - (external_current * delta**2) / (currents_period * num_neurons)
        ) / ((num_neurons - 1) * (delta - refractory_period))

    def w_leak(delta):
        return (
            (leak_coefficient * membrane_threshold * delta**2)
            / (1 - np.exp(-leak_coefficient * delta))
            - (external_current * delta**2) / (currents_period * num_neurons)
        ) / ((num_neurons - 1) * (delta - refractory_period))

    # TODO: Parameterize the range of delta.
    delta = np.logspace(0.001, 3.2, 500)
    w_leak_free_values = w_leak_free(delta)
    w_leak_values = w_leak(delta)

    horizontal_line = membrane_threshold / (
        num_neurons - 1
    ) - 2 * external_current * refractory_period / (
        currents_period * (num_neurons - 1) * num_neurons
    )

    w_crit_spike = membrane_threshold / (num_neurons - 1) - (
        2 * external_current * refractory_period
    ) / (currents_period * num_neurons * (num_neurons - 1))

    w_mean_unique_values = simulation_df["w_mean"].unique()

    fig, axs = plt.subplots(2, 2, figsize=(24, 16), constrained_layout=True)

    # Top-left plot
    axs[0, 0].plot(
        delta,
        w_leak_free_values,
        label=r"$\langle W \rangle(\langle \delta\rangle)_{\text{leak-free}}$",
        color="purple",
        linewidth=5,
    )
    axs[0, 0].axhline(
        horizontal_line,
        color="purple",
        linestyle="--",
        linewidth=3,
        label=r"$\langle W \rangle_{\text{critical}}$",
    )
    axs[0, 0].plot(
        delta,
        w_leak_values,
        label=r"$\langle W \rangle(\langle \delta\rangle)_{\text{leak}}$",
        color="green",
        linewidth=5,
    )
    if leak_coefficient > 0:
        delta_alpha = 0.2 / leak_coefficient if leak_coefficient != 0 else np.inf
        axs[0, 0].axvline(
            delta_alpha,
            color="green",
            linestyle=":",
            linewidth=3,
            label=r"$\alpha\langle \delta\rangle=0.2$",
        )
    axs[0, 0].set_xscale("log")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_xlabel(r"$\langle \delta \rangle$")
    axs[0, 0].set_ylabel(r"$\langle W \rangle$")
    axs[0, 0].legend(loc="lower left")

    # Top-right plot
    grouped_spike = simulation_df.groupby("w_mean")
    nspike_median = grouped_spike["total_spikes"].median().values
    nspike_q1 = grouped_spike["total_spikes"].quantile(0.25).values
    nspike_q3 = grouped_spike["total_spikes"].quantile(0.75).values

    axs[0, 1].plot(
        w_mean_unique_values,
        nspike_median,
        color="black",
        linewidth=5,
        label=r"Num. Spikes (synthetic SNN)",
    )
    axs[0, 1].fill_between(
        w_mean_unique_values, nspike_q1, nspike_q3, color="black", alpha=0.2
    )
    axs[0, 1].axvline(
        x=w_crit_spike,
        color="purple",
        linestyle="--",
        linewidth=5,
        label=r"$\langle W \rangle_{\text{critical}}$",
    )
    axs[0, 1].set_xlim(0.011, 0.049)
    axs[0, 1].set_xlabel(r"$\langle W \rangle$")
    axs[0, 1].set_ylabel(r"Number of Spikes")

    # Configure y-axis to use scientific notation
    axs[0, 1].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axs[0, 1].yaxis.get_major_formatter().set_scientific(True)
    axs[0, 1].yaxis.get_major_formatter().set_powerlimits((-1, 1))

    axs[0, 1].legend(loc="upper left")

    # Bottom-left plot
    w_crit_lzw = membrane_threshold / (num_neurons - 1) - (
        2 * external_current * refractory_period
    ) / (currents_period * num_neurons * (num_neurons - 1))

    grouped_lzw = simulation_df.groupby("w_mean")
    median_complexity = grouped_lzw["lzw_complexity"].median().values
    median_complexity_max = median_complexity.max()
    median_complexity /= median_complexity_max
    q1_complexity = (
        grouped_lzw["lzw_complexity"].quantile(0.25).values / median_complexity_max
    )
    q3_complexity = (
        grouped_lzw["lzw_complexity"].quantile(0.75).values / median_complexity_max
    )

    axs[1, 0].plot(
        w_mean_unique_values,
        median_complexity,
        color="black",
        linewidth=5,
        label=r"LZW complexity (synthetic SNN)",
    )
    axs[1, 0].fill_between(
        w_mean_unique_values, q1_complexity, q3_complexity, color="black", alpha=0.2
    )
    axs[1, 0].axvline(
        x=w_crit_lzw,
        color="purple",
        linestyle="--",
        linewidth=5,
        label=r"$\langle W \rangle_{\text{critical}}$",
    )
    axs[1, 0].set_xlim(0.011, 0.049)
    axs[1, 0].set_xlabel(r"$\langle W \rangle$")
    axs[1, 0].set_ylabel(r"Normalized LZW Complexity")
    axs[1, 0].legend(loc="upper left")

    # Bottom-right plot
    axs[1, 1].axis("off")  # Remove empty axis.
