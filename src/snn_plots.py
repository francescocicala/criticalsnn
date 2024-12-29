import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt


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


def calculate_theoretical_isi(w_mean, N, theta, tau, I, t_ref):
    """
    Calculate the theoretical ISI value for each mean W in w_mean.
    
    Parameters
    ----------
    w_mean : array-like
        The mean W values over which to compute the theoretical ISI.
    N : int
        Number of neurons.
    theta : float
        Firing threshold.
    tau : float
        Time constant.
    I : float
        External current.
    t_ref : float
        Refractory period.
    
    Returns
    -------
    numpy.ndarray
        The theoretical ISI values corresponding to each w_mean.
    """
    term1 = theta - w_mean * (N - 1)
    term2 = np.sqrt(
        term1**2 + (4 * I * w_mean * (N - 1) * t_ref) / (tau * N)
    )
    return (tau * N) / (2 * I) * (term1 + term2)


def compute_w_critical(N, theta, tau, I, t_ref):
    """
    Compute the critical point of W (w_critical).
    
    Parameters
    ----------
    N : int
        Number of neurons.
    theta : float
        Firing threshold.
    tau : float
        Time constant.
    I : float
        External current.
    t_ref : float
        Refractory period.

    Returns
    -------
    float
        The critical value of W (w_critical).
    """
    return (theta / (N - 1)) - (2 * I * t_ref) / (tau * N * (N - 1))


def plot_isi_results(
    df_results,
    N=1000,
    theta=7,
    tau=10,
    I=0.5,
    t_ref=2,
    x_min=0.0061,
    x_max=0.1
):
    """
    Reads a dataframe containing ISI data, computes mean ISI, 
    plots it against the theoretical ISI, and draws the critical 
    W line in log-log scale.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing ISI data (with columns 'w_mean' and 'mean_isi').
    N : int, optional
        Number of neurons (default = 1000).
    theta : float, optional
        Firing threshold (default = 7).
    tau : float, optional
        Time constant (default = 10).
    I : float, optional
        External current (default = 0.5).
    t_ref : float, optional
        Refractory period (default = 2).
    x_min : float, optional
        Minimum x value for the plot (default = 0.0061).
    x_max : float, optional
        Maximum x value for the plot (default = 0.1).
    """

    # Initialize matplotlib style
    init_matplotlib_style()

    # Compute the critical point
    w_critical = compute_w_critical(N, theta, tau, I, t_ref)

    # Calculate mean values from the data (group by w_mean)
    mean_values = df_results.groupby('w_mean').mean()['mean_isi']

    w_means = mean_values.index.values

    theoretical_isi = calculate_theoretical_isi(w_means, N, theta, tau, I, t_ref)

    # Create the plot with a logarithmic scale on both axes
    plt.figure(figsize=(10, 6))
    plt.plot(
        w_means,
        mean_values,
        label=r'$\langle \Delta \rangle (\langle W \rangle)_{\text{leak-free }}$ (Single iter. simulation)',
        color='black',
        linewidth=4
    )
    plt.plot(
        w_means,
        theoretical_isi,
        label=r'$\langle \Delta \rangle (\langle W \rangle)_{\text{leak-free }}$ (Eq. 8)',
        linestyle='-',
        color='purple',
        linewidth=4
    )

    # Add a vertical line at the critical point
    plt.axvline(
        x=w_critical,
        color='purple',
        linestyle='--',
        label=r'Theoretical $\langle W \rangle_{\text{critical, leak-free}}$'
    )

    # Use log scales
    plt.yscale('log')

    # Set x-axis range
    plt.xlim(left=x_min, right=x_max)

    # Axis labels
    plt.xlabel(r'$\langle W \rangle$', fontsize=26, labelpad=10)
    plt.ylabel(r'$\langle \Delta \rangle  (\langle W \rangle)_{\text{leak-free }}$', fontsize=26, labelpad=10)

    # Legend
    plt.legend(fontsize=16)

    # Grid (use dashed lines and show on both major/minor ticks)
    plt.grid(True, which="both", linestyle='--')

    # Tight layout for cleaner spacing
    plt.tight_layout()
