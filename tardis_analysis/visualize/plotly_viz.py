# tardis_analysis/visualize/plotly_viz.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from ..config import PLOTLY_COLORS

def plot_all_commits_plotly(all_data, spectrum_keys, output_dir):
    """Plot spectrum data for all commits using Plotly."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f'Luminosity for {key}' for key in spectrum_keys])

    for idx, key in enumerate(spectrum_keys):
        row = idx // 2 + 1
        col = idx % 2 + 1
        for commit_idx, data in enumerate(all_data, 1):
            if key in data:
                wavelength = data[key]['wavelength']
                luminosity = data[key]['luminosity']
                color = PLOTLY_COLORS[(commit_idx - 1) % len(PLOTLY_COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=wavelength,
                        y=luminosity,
                        mode='lines',
                        name=f'Commit {commit_idx}',
                        legendgroup=f'Commit {commit_idx}',
                        showlegend=(idx == 0),
                        line=dict(color=color)
                    ),
                    row=row,
                    col=col
                )

    fig.update_layout(
        title='Comparison of Spectrum Solvers Across All Commits',
        height=800,
        width=1200,
        showlegend=True,
    )

    file_name = os.path.join(output_dir, "all_commits_spectrum_comparison.html")
    try:
        fig.write_html(file_name)
        print(f"Saved Plotly plot as {file_name}")
    except Exception as e:
        print(f"Failed to save {file_name}: {e}")