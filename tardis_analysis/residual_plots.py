# tardis_analysis/residual_plots.py

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from .config import MATPLOTLIB_COLORS, PLOTLY_COLORS

def plot_residuals_matplotlib(all_data, spectrum_keys, output_dir, reference_index=0):
    """Plot fractional residuals for all commits relative to a reference commit using Matplotlib."""
    if not all_data:
        print("No data to plot.")
        return

    # Set up 2x2 subplot grid to match existing layout
    fig, axes = plt.subplots(2, 2, figsize=(20, 15), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, key in enumerate(spectrum_keys):
        ax = axes[idx]
        # Check if reference commit has the key
        if reference_index >= len(all_data) or key not in all_data[reference_index]:
            print(f"Reference commit does not have data for {key}. Skipping.")
            continue

        ref_wavelength = all_data[reference_index][key]['wavelength']
        ref_luminosity = all_data[reference_index][key]['luminosity']

        # Plot residuals for each commit
        for commit_idx, data in enumerate(all_data):
            if key in data:
                wavelength = data[key]['wavelength']
                luminosity = data[key]['luminosity']
                # Ensure wavelengths match the reference
                if not np.array_equal(wavelength, ref_wavelength):
                    print(f"Wavelengths differ for commit {commit_idx} and reference for {key}. Skipping.")
                    continue
                # Calculate fractional residuals, handling division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    residuals = np.where(ref_luminosity != 0, (luminosity - ref_luminosity) / ref_luminosity, 0)
                color = MATPLOTLIB_COLORS[commit_idx % len(MATPLOTLIB_COLORS)]
                ax.plot(wavelength, residuals, label=f'Commit {commit_idx + 1}', color=color)

        # Add reference line at y=0
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_title(f'Fractional Residuals for {key}')
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Fractional Residual')
        ax.legend()
        ax.grid(True)

    plt.suptitle('Fractional Residuals Across All Commits', fontsize=16)
    plt.tight_layout()
    file_name = os.path.join(output_dir, "residuals_all_commits.pdf")
    try:
        plt.savefig(file_name)
        print(f"Saved Matplotlib residual plot as {file_name}")
    except Exception as e:
        print(f"Failed to save {file_name}: {e}")
    plt.close()

def plot_residuals_plotly(all_data, spectrum_keys, output_dir, reference_index=0):
    """Plot fractional residuals for all commits relative to a reference commit using Plotly."""
    if not all_data:
        print("No data to plot.")
        return

    # Set up 2x2 subplot grid
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f'Fractional Residuals for {key}' for key in spectrum_keys])

    for idx, key in enumerate(spectrum_keys):
        row = idx // 2 + 1
        col = idx % 2 + 1
        # Check if reference commit has the key
        if reference_index >= len(all_data) or key not in all_data[reference_index]:
            print(f"Reference commit does not have data for {key}. Skipping.")
            continue

        ref_wavelength = all_data[reference_index][key]['wavelength']
        ref_luminosity = all_data[reference_index][key]['luminosity']

        # Plot residuals for each commit
        for commit_idx, data in enumerate(all_data):
            if key in data:
                wavelength = data[key]['wavelength']
                luminosity = data[key]['luminosity']
                if not np.array_equal(wavelength, ref_wavelength):
                    print(f"Wavelengths differ for commit {commit_idx} and reference for {key}. Skipping.")
                    continue
                with np.errstate(divide='ignore', invalid='ignore'):
                    residuals = np.where(ref_luminosity != 0, (luminosity - ref_luminosity) / ref_luminosity, 0)
                color = PLOTLY_COLORS[commit_idx % len(PLOTLY_COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=wavelength,
                        y=residuals,
                        mode='lines',
                        name=f'Commit {commit_idx + 1}',
                        legendgroup=f'Commit {commit_idx + 1}',
                        showlegend=(idx == 0),  # Show legend only in first subplot
                        line=dict(color=color)
                    ),
                    row=row,
                    col=col
                )
        # Add reference line at y=0
        fig.add_hline(y=0, line=dict(color='black', dash='dash', width=0.8), row=row, col=col)

    fig.update_layout(
        title='Fractional Residuals Across All Commits',
        height=800,
        width=1200,
        showlegend=True
    )

    file_name = os.path.join(output_dir, "residuals_all_commits.html")
    try:
        fig.write_html(file_name)
        print(f"Saved Plotly residual plot as {file_name}")
    except Exception as e:
        print(f"Failed to save {file_name}: {e}")