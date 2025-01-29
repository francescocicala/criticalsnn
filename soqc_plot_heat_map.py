import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_csv_and_plot(filename):
    """Prints the maximum error and plots a heatmap from the data in the CSV file."""

    output_folder = "SOqC"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_path = os.path.join(output_folder, filename)

    data = pd.read_csv(file_path)

    max_error = data["error"].max()
    print(f"Max error: {max_error}")

    # Group by all columns except 'error' and calculate the mean of each group
    grouped_data = data.groupby(
        ["membrane_threshold", "external_current"], as_index=False
    ).mean()

    membrane_threshold_values = grouped_data["membrane_threshold"]
    external_current_values = grouped_data["external_current"]
    error_values = grouped_data["error"]

    # Create a meshgrid for plotting
    X_grid, Y_grid = np.meshgrid(np.unique(membrane_threshold_values), np.unique(external_current_values))
    error_values_grid = np.zeros_like(X_grid)

    # Fill the error values grid based on the combination of membrane_threshold and external_current
    for i, membrane_threshold in enumerate(np.unique(membrane_threshold_values)):
        for j, external_current in enumerate(np.unique(external_current_values)):
            error_values_grid[j, i] = grouped_data[
                (grouped_data["membrane_threshold"] == membrane_threshold) & 
                (grouped_data["external_current"] == external_current)
            ]["error"].values[0]

    # Plotting the heatmap
    fig, ax = plt.subplots()
    cax = ax.imshow(
        error_values_grid, cmap="viridis", aspect="auto", origin="lower",
        extent=[
            np.min(membrane_threshold_values), np.max(membrane_threshold_values),
            np.min(external_current_values), np.max(external_current_values)
        ]
    )

    ax.set_xlabel("Membrane Threshold")
    ax.set_ylabel("External Current")
    fig.colorbar(cax, ax=ax, label="Error")

    output_plot_path = os.path.join(output_folder, "error_heatmap.png")
    fig.savefig(output_plot_path, dpi=300)
    print(f"Heatmap saved as {output_plot_path}")


analyze_csv_and_plot("soqc_results.csv")
