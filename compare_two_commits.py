import subprocess
import os
from git import Repo
import h5py
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

n = 3  
tardis_repo_path = "/home/riddhi/workspace/tardis-main/tardis" 
regression_data_repo_path = "/home/riddhi/workspace/tardis-main/tardis-regression-data"  
branch = "master"  
target_file = "tardis/spectrum/tests/test_spectrum_solver/test_spectrum_solver/TestSpectrumSolver.h5"  
target_file_path = os.path.join(regression_data_repo_path, target_file)

tardis_repo = Repo(tardis_repo_path)
regression_repo = Repo(regression_data_repo_path)

original_head = regression_repo.head.commit.hexsha
print(f"Original HEAD of regression data repo: {original_head}")

commits = list(tardis_repo.iter_commits(branch, max_count=n))
commits.reverse()  

processed_commits = []  
regression_commits = []  
commit_data = []  

spectrum_keys = [
    'spectrum_integrated',
    'spectrum_real_packets',
    'spectrum_real_packets_reabsorbed',
    'spectrum_virtual_packets'
]

def load_h5_data(file_path, spectrum_keys):
    h5_data = {}
    print(f"\nInspecting HDF5 file: {file_path}")
    try:
        with h5py.File(file_path, 'r') as f:
            print("Top-level keys:", list(f.keys()))
            if 'simulation' not in f:
                print("Error: 'simulation' group not found in the HDF5 file.")
                return h5_data
            if 'spectrum_solver' not in f['simulation']:
                print("Error: 'spectrum_solver' group not found under 'simulation'.")
                return h5_data
            for key in spectrum_keys:
                group_path = f'simulation/spectrum_solver/{key}'
                print(f"Checking {group_path}")
                if group_path in f:
                    group = f[group_path]
                    if not isinstance(group, h5py.Group):
                        print(f"Warning: {group_path} is not a group, skipping.")
                        continue
                    wavelength_path = f'{group_path}/wavelength/values'
                    luminosity_path = f'{group_path}/luminosity/values'
                    print(f"  {wavelength_path}: {'exists' if wavelength_path in f else 'missing'}")
                    print(f"  {luminosity_path}: {'exists' if luminosity_path in f else 'missing'}")
                    if wavelength_path in f and luminosity_path in f:
                        if isinstance(f[wavelength_path], h5py.Dataset) and isinstance(f[luminosity_path], h5py.Dataset):
                            wavelength = f[wavelength_path][()]
                            luminosity = f[luminosity_path][()]
                            h5_data[key] = {'wavelength': wavelength, 'luminosity': luminosity}
                            print(f"Loaded data for {key}: wavelength shape={wavelength.shape}, luminosity shape={luminosity.shape}")
                        else:
                            print(f"Error: {wavelength_path} or {luminosity_path} is not a dataset.")
                    else:
                        print(f"Warning: Missing 'wavelength/values' or 'luminosity/values' in {group_path}")
                else:
                    print(f"Warning: {group_path} not found in {file_path}")
            if not h5_data:
                print("No data loaded for any spectrum keys.")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return h5_data

def plot_spectrum_comparison_matplotlib(data1, data2, spectrum_keys, commit_i, commit_j, output_dir):
    has_data = any(key in data1 and key in data2 for key in spectrum_keys)
    if not has_data:
        print(f"No data to plot for commits {commit_i} and {commit_j}.")
        return

    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 3, 1], hspace=0.1, wspace=0.3)

    for idx, key in enumerate(spectrum_keys):
        if key not in data1 or key not in data2:
            continue

        row = (idx // 2) * 2
        col = idx % 2
        
        ax_luminosity = fig.add_subplot(gs[row, col])
        ax_residuals = fig.add_subplot(gs[row + 1, col], sharex=ax_luminosity)
        
        for data, label, linestyle in [(data1, f'Commit {commit_i}', '-'), (data2, f'Commit {commit_j}', '--')]:
            if key in data:
                wavelength = data[key]['wavelength']
                luminosity = data[key]['luminosity']
                ax_luminosity.plot(wavelength, luminosity, linestyle=linestyle, label=label)
        
        ax_luminosity.set_ylabel('Luminosity')
        ax_luminosity.set_title(f'Luminosity for {key}')
        ax_luminosity.legend()
        ax_luminosity.grid(True)
        
        if key in data1 and key in data2:
            wavelength = data1[key]['wavelength']
            luminosity1 = data1[key]['luminosity']
            luminosity2 = data2[key]['luminosity']
            if np.array_equal(wavelength, data2[key]['wavelength']):
                with np.errstate(divide='ignore', invalid='ignore'):
                    fractional_residuals = np.where(luminosity1 != 0, (luminosity2 - luminosity1) / luminosity1, 0)
                ax_residuals.plot(wavelength, fractional_residuals, label='Fractional Residuals', color='purple')
                ax_residuals.axhline(0, color='black', linestyle='--', linewidth=0.8)
        
        ax_residuals.set_xlabel('Wavelength')
        ax_residuals.set_ylabel('Fractional Residuals')
        ax_residuals.legend()
        ax_residuals.grid(True)
        
        ax_luminosity.tick_params(axis='x', labelbottom=False)
        if row != 2:
            ax_residuals.tick_params(axis='x', labelbottom=False)
    
    plt.suptitle(f'Comparison of Spectrum Solvers between Commit {commit_i} and Commit {commit_j}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    file_name = os.path.join(output_dir, f"spectrum_comparison_c{commit_i}_c{commit_j}.png")
    try:
        plt.savefig(file_name)
        print(f"Saved Matplotlib plot as {file_name}")
    except Exception as e:
        print(f"Failed to save {file_name}: {e}")
    plt.close()

def plot_spectrum_comparison_plotly(data1, data2, spectrum_keys, commit_i, commit_j, output_dir):
    has_data = any(key in data1 and key in data2 for key in spectrum_keys)
    if not has_data:
        print(f"No data to plot for commits {commit_i} and {commit_j}.")
        return

    fig = make_subplots(
        rows=4,
        cols=2,
        subplot_titles=[
            f'Luminosity for {key}' for key in spectrum_keys
        ] + [
            'Fractional Residuals' for _ in spectrum_keys
        ],
        vertical_spacing=0.07,
        horizontal_spacing=0.08,
        row_heights=[0.3, 0.15] * 2,
        shared_xaxes=True,
    )

    for idx, key in enumerate(spectrum_keys):
        if key not in data1 or key not in data2: 
            continue

        plot_col = idx % 2 + 1
        plot_row = (idx // 2) * 2 + 1

        for data, name, line_style in [(data1, f'Commit {commit_i}', 'solid'), (data2, f'Commit {commit_j}', 'dash')]:
            if key in data:
                wavelength = data[key]['wavelength']
                luminosity = data[key]['luminosity']
                fig.add_trace(
                    go.Scatter(
                        x=wavelength,
                        y=luminosity,
                        mode='lines',
                        name=name,
                        line=dict(dash=line_style),
                    ),
                    row=plot_row,
                    col=plot_col
                )

        if key in data1 and key in data2:
            wavelength = data1[key]['wavelength']
            luminosity1 = data1[key]['luminosity']
            luminosity2 = data2[key]['luminosity']
            if np.array_equal(wavelength, data2[key]['wavelength']):
                with np.errstate(divide='ignore', invalid='ignore'):
                    fractional_residuals = np.where(luminosity1 != 0, (luminosity2 - luminosity1) / luminosity1, 0)
                fig.add_trace(
                    go.Scatter(
                        x=wavelength,
                        y=fractional_residuals,
                        mode='lines',
                        name='Residuals',
                        line=dict(color='purple'),
                    ),
                    row=plot_row + 1,
                    col=plot_col
                )
                fig.add_hline(
                    y=0,
                    line=dict(color='black', dash='dash', width=0.8),
                    row=plot_row + 1,
                    col=plot_col
                )

    fig.update_layout(
        title=f'Comparison of Spectrum Solvers between Commit {commit_i} and Commit {commit_j}',
        height=900,
        width=1200,
        showlegend=True,
        margin=dict(t=50, b=30, l=50, r=30),
        plot_bgcolor='rgba(240, 240, 255, 0.3)',
    )

    file_name = os.path.join(output_dir, f"spectrum_comparison_c{commit_i}_c{commit_j}.html")
    try:
        fig.write_html(file_name)
        print(f"Saved Plotly plot as {file_name}")
    except Exception as e:
        print(f"Failed to save {file_name}: {e}")

for i, commit in enumerate(commits, 1):
    print(f"Processing commit {i}/{n}: {commit.hexsha}")
 
    tardis_repo.git.checkout(commit.hexsha)

    tardis_repo.git.reset('--hard')
    tardis_repo.git.clean('-fd')

    cmd = [
        "python", "-m", "pytest",
        "tardis/spectrum/tests/test_spectrum_solver.py",
        f"--tardis-regression-data={regression_data_repo_path}",
        "--generate-reference",
        "-x"
    ]
    print(f"Running pytest command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, cwd=tardis_repo_path, capture_output=True, text=True)
        print("Pytest stdout:")
        print(result.stdout)
        print("Pytest stderr:")
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running pytest for commit {commit.hexsha}: {e}")
        print("Pytest stdout:")
        print(e.stdout)
        print("Pytest stderr:")
        print(e.stderr)
        raise

    if not os.path.exists(target_file_path):
        print(f"Error: HDF5 file {target_file_path} was not generated.")
        continue

    regression_repo.git.add(A=True)
    regression_commit = regression_repo.index.commit(f"Regression data for tardis commit {i}")
    regression_commits.append((regression_commit.hexsha, regression_commit.message))

    processed_commits.append(commit.hexsha)

    current_data = load_h5_data(target_file_path, spectrum_keys)
    commit_data.append(current_data)

print("\nProcessed Tardis Commits:")
for hash in processed_commits:
    print(hash)

print("\nRegression Data Commits:")
for hash, msg in regression_commits:
    print(f"{hash}: {msg}")

output_dir = os.path.join(tardis_repo_path, "comparison_plots")
os.makedirs(output_dir, exist_ok=True)

for k in range(n - 1):
    data1 = commit_data[k]
    data2 = commit_data[k + 1]
    plot_spectrum_comparison_matplotlib(data1, data2, spectrum_keys, k + 1, k + 2, output_dir)
    plot_spectrum_comparison_plotly(data1, data2, spectrum_keys, k + 1, k + 2, output_dir)

print(f"\nResetting regression data repo to {original_head}")
regression_repo.git.reset('--hard', original_head)

tardis_repo.git.checkout(branch)