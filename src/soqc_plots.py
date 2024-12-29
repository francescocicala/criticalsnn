import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

def get_relative_error_stats(data: pd.DataFrame) -> None:
    """
    Print the minimum and maximum values of the 'relative_error_percent' column.

    Args:
        data (pd.DataFrame): A DataFrame containing at least the column 'relative_error_percent'.
    """
    min_error = data['relative_error_percent'].min()
    max_error = data['relative_error_percent'].max()
    print(f"Minimum relative_error_percent: {min_error}")
    print(f"Maximum relative_error_percent: {max_error}")

def plot_relative_error_3d(
    data: pd.DataFrame,
    tau: float,
    figure_size: tuple = (12, 8),
) -> None:
    """
    Create and optionally display a 3D scatter plot to visualize the relative_error_percent 
    versus N (number of neurons) and I/tau (normalized current).

    Args:
        data (pd.DataFrame): A DataFrame containing columns:
            - 'N' (number of neurons)
            - 'I' (input current)
            - 'relative_error_percent' (percentage or fraction).
        tau (float): A timescale parameter used to normalize the current I.
        figure_size (tuple): The size of the resulting Matplotlib figure.
    """
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of the data in 3D
    sc = ax.scatter(
        data['N'],
        data['I'] / tau,
        data['relative_error_percent'],
        c=data['relative_error_percent'],
        cmap='viridis',
        alpha=0.8
    )

    # Label and style the axes
    ax.set_xlabel('N (Number of Neurons)', fontsize=12)
    ax.set_ylabel('I/tau (Normalized Current)', fontsize=12)
    ax.set_zlabel('relative_error_percent (%)', fontsize=12)
    ax.set_title('3D Plot of Model Accuracy', fontsize=14)

    # Create a colorbar to match the scatter plot
    cb = plt.colorbar(sc, pad=0.1)
    cb.set_label('relative_error_percent (%)', fontsize=12)

    plt.tight_layout()
