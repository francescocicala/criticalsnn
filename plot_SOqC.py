import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def analyze_csv_and_plot(filename):
    """Prints the maximum error and plots a 3D chart from the data in the CSV file."""
    
    output_folder = "SOqC"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_path = os.path.join(output_folder, filename)

    data = pd.read_csv(file_path)

    max_error = data["error"].max()
    print(f"Max error: {max_error}")

    membrane_threshold_values = data["membrane_threshold"]
    external_current_values = data["external_current"]

    X_grid, Y_grid = np.meshgrid(np.unique(membrane_threshold_values), np.unique(external_current_values))
    error_values = np.zeros_like(X_grid)

    for i, membrane_threshold in enumerate(np.unique(membrane_threshold_values)):
        for j, external_current in enumerate(np.unique(external_current_values)):
            error_values[j, i] = data[
                (data["membrane_threshold"] == membrane_threshold) & (data["external_current"] == external_current)
            ]["error"].values[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X_grid, Y_grid, error_values, cmap="viridis", edgecolor="none")

    ax.set_xlabel("Membrane Threshold")
    ax.set_ylabel("External Current")
    ax.set_zlabel("Error")

    output_plot_path = os.path.join(output_folder, "error_3d_surface_plot.png")
    fig.savefig(output_plot_path, dpi=300)
    print(f"Graph saved as {output_plot_path}")


analyze_csv_and_plot("SOqC results.csv")
