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

matplotlib_colors = [
    'blue', 'red', 'green', 'purple', 'orange',
    'cyan', 'magenta', 'lime', 'brown', 'pink'
]

plotly_colors = [
    '#1f77b4',  
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf'  
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

def plot_all_commits_matplotlib(all_data, spectrum_keys, output_dir):

    fig, axes = plt.subplots(2, 2, figsize=(20, 15), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, key in enumerate(spectrum_keys):
        ax = axes[idx]
        for commit_idx, data in enumerate(all_data, 1):
            if key in data:
                wavelength = data[key]['wavelength']
                luminosity = data[key]['luminosity']
                # Use the color corresponding to the commit index
                color = matplotlib_colors[(commit_idx - 1) % len(matplotlib_colors)]
                ax.plot(wavelength, luminosity, label=f'Commit {commit_idx}', color=color)
        ax.set_title(f'Luminosity for {key}')
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Luminosity')
        ax.legend()
        ax.grid(True)

    plt.suptitle('Comparison of Spectrum Solvers Across All Commits', fontsize=16)
    plt.tight_layout()
    file_name = os.path.join(output_dir, "all_commits_spectrum_comparison.pdf")
    try:
        plt.savefig(file_name)
        print(f"Saved Matplotlib plot as {file_name}")
    except Exception as e:
        print(f"Failed to save {file_name}: {e}")
    plt.close()

def plot_all_commits_plotly(all_data, spectrum_keys, output_dir):

    fig = make_subplots(rows=2, cols=2, subplot_titles=[f'Luminosity for {key}' for key in spectrum_keys])

    for idx, key in enumerate(spectrum_keys):
        row = idx // 2 + 1
        col = idx % 2 + 1
        for commit_idx, data in enumerate(all_data, 1):
            if key in data:
                wavelength = data[key]['wavelength']
                luminosity = data[key]['luminosity']
               
                color = plotly_colors[(commit_idx - 1) % len(plotly_colors)]
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

plot_all_commits_matplotlib(commit_data, spectrum_keys, output_dir)

plot_all_commits_plotly(commit_data, spectrum_keys, output_dir)

print(f"\nResetting regression data repo to {original_head}")
regression_repo.git.reset('--hard', original_head)

tardis_repo.git.checkout(branch)