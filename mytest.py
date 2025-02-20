import shutil
import subprocess
import tempfile
from filecmp import dircmp
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from IPython.display import display

# Configuration
CONFIG = {
    'compare_path': '.',
    'temp_dir_prefix': 'ref_compare_',
}

# Setup logging
logger = logging.getLogger(__name__)

# Utility Functions
def color_print(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def get_last_n_commits(n):
    """Fetch the last N commits in chronological order (oldest to newest)."""
    try:
        result = subprocess.run(['git', 'log', '--format=%H', '-n', str(n)],
                                capture_output=True, text=True, check=True)
        commits = result.stdout.strip().split('\n')
        if len(commits) >= n:
            return commits[::-1]  # Reverse to get oldest to newest
        return []
    except (subprocess.SubprocessError, subprocess.CalledProcessError):
        print("Error: Unable to get git commits.")
        return []

class FileManager:
    def __init__(self):
        self.temp_dir = None

    def setup(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix=CONFIG['temp_dir_prefix']))
        print(f'Created temporary directory at {self.temp_dir}')

    def teardown(self):
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f'Removed temporary directory {self.temp_dir}')
        self.temp_dir = None

    def get_temp_path(self, filename):
        return self.temp_dir / filename

class FileSetup:
    def __init__(self, file_manager, commit_hashes):
        self.file_manager = file_manager
        self.commit_hashes = commit_hashes

    def setup(self):
        for ref_id, ref_hash in enumerate(self.commit_hashes, 1):
            ref_dir = self.file_manager.get_temp_path(f"ref{ref_id}")
            os.makedirs(ref_dir, exist_ok=True)
            if ref_hash:
                self._copy_data_from_hash(ref_hash, ref_dir)
            else:
                subprocess.run(f'cp -r {CONFIG["compare_path"]}/* {ref_dir}', shell=True)

    def _copy_data_from_hash(self, ref_hash, ref_dir):
        git_cmd = ['git', 'archive', ref_hash, '|', 'tar', '-x', '-C', str(ref_dir)]
        subprocess.run(' '.join(git_cmd), shell=True)

class DiffAnalyzer:
    def __init__(self, file_manager):
        self.file_manager = file_manager

    def print_diff_files(self, dcmp):
        dcmp.right = Path(dcmp.right)
        dcmp.left = Path(dcmp.left)
        self._print_new_files(dcmp.right_only, dcmp.right, "ref2")
        self._print_new_files(dcmp.left_only, dcmp.left, "ref1")
        self._print_modified_files(dcmp)
        for sub_dcmp in dcmp.subdirs.values():
            self.print_diff_files(sub_dcmp)

    def _print_new_files(self, files, path, ref):
        for item in files:
            if Path(path, item).is_file():
                print(f"New file detected inside {ref}: {item}")
                print(f"Path: {Path(path, item)}")
                print()

    def _print_modified_files(self, dcmp):
        for name in dcmp.diff_files:
            print(f"Modified file found: {name}")
            left = self._get_relative_path(dcmp.left)
            right = self._get_relative_path(dcmp.right)
            if left == right:
                print(f"Path: {left}")
            print()

    def _get_relative_path(self, path):
        try:
            return str(Path(path).relative_to(self.file_manager.temp_dir))
        except ValueError:
            return str(path)

class HDFComparator:
    def __init__(self, print_path=False):
        self.print_path = print_path

    def summarise_changes_hdf(self, name, path1, path2):
        ref1 = pd.HDFStore(Path(path1) / name)
        ref2 = pd.HDFStore(Path(path2) / name)
        k1, k2 = set(ref1.keys()), set(ref2.keys())
        
        different_keys = len(k1 ^ k2)
        identical_items = []
        identical_name_different_data = []
        identical_name_different_data_dfs = {}

        for item in k1 & k2:
            try:
                if ref1[item].equals(ref2[item]):
                    identical_items.append(item)
                else:
                    identical_name_different_data.append(item)
                    identical_name_different_data_dfs[item] = (ref1[item] - ref2[item]) / ref1[item]
                    self._compare_and_display_differences(ref1[item], ref2[item], item, name, path1, path2)
            except Exception as e:
                print(f"Error comparing item: {item}")
                print(e)

        ref1.close()
        ref2.close()

        if different_keys > 0 or len(identical_name_different_data) > 0:
            print("\n" + "=" * 50)
            print(f"Summary for {name}:")
            print(f"Total keys - ref1: {len(k1)}, ref2: {len(k2)}")
            print(f"Different keys: {different_keys}")
            print(f"Same name, different data: {len(identical_name_different_data)}")
            print(f"Identical keys: {len(identical_items)}")
            print("=" * 50)
            print()

        return {
            "different_keys": different_keys,
            "identical_keys": len(identical_items),
            "identical_keys_diff_data": len(identical_name_different_data),
            "identical_name_different_data_dfs": identical_name_different_data_dfs,
            "ref1_keys": list(k1),
            "ref2_keys": list(k2)
        }

    def _compare_and_display_differences(self, df1, df2, item, name, path1, path2):
        abs_diff = np.fabs(df1 - df2)
        rel_diff = abs_diff / np.maximum(np.fabs(df1), np.fabs(df2))
        FLOAT_UNCERTAINTY = 1e-14
        max_rel_diff = np.nanmax(rel_diff)

        if max_rel_diff > FLOAT_UNCERTAINTY:
            logger.warning(
                f"Significant difference in {name}, key={item}\n"
                f"Max relative diff: {max_rel_diff:.2e} ({max_rel_diff*100:.2e}%)"
            )

        print(f"Heatmap for key {item} in {name}")
        for diff_type, diff in zip(["abs", "rel"], [abs_diff, rel_diff]):
            print(f"Visualizing {'Absolute' if diff_type == 'abs' else 'Relative'} Differences")
            self._display_difference(diff)
        
        if self.print_path:
            print(f"Path1: {path1}" if path1 != path2 else f"Path: {path1}")
            if path1 != path2:
                print(f"Path2: {path2}")

    def _display_difference(self, diff):
        with pd.option_context('display.max_rows', 100, 'display.max_columns', 10):
            if isinstance(diff, pd.Series):
                diff = pd.DataFrame([diff.mean(), diff.max()], index=['mean', 'max'])
            elif isinstance(diff.index, pd.core.indexes.multi.MultiIndex):
                diff = diff.reset_index(drop=True)
            diff = pd.DataFrame([diff.mean(), diff.max()], index=['mean', 'max'])
            display(diff.style.format('{:.2g}').background_gradient(cmap='Reds'))

class SpectrumSolverComparator:
    def __init__(self, ref_paths, commit_labels):
        self.ref_paths = ref_paths
        self.commit_labels = commit_labels
        self.spectrum_keys = [
            'spectrum_integrated',
            'spectrum_real_packets',
            'spectrum_real_packets_reabsorbed',
            'spectrum_virtual_packets'
        ]
        self.data = {}

    def setup(self):
        for ref_name, file_path in zip(self.commit_labels, self.ref_paths):
            self.data[ref_name] = {}
            try:
                with pd.HDFStore(file_path) as hdf:
                    for key in self.spectrum_keys:
                        full_key = f"simulation/spectrum_solver/{key}"
                        self.data[ref_name][key] = {
                            'wavelength': np.array(hdf[f'{full_key}/wavelength']),
                            'luminosity': np.array(hdf[f'{full_key}/luminosity'])
                        }
            except (FileNotFoundError, KeyError) as e:
                print(f"Warning: Issue with {file_path}: {e}")

    def plot_matplotlib(self):
        fig = plt.figure(figsize=(20, 20))
        gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 3, 1], hspace=0.1, wspace=0.3)

        for idx, key in enumerate(self.spectrum_keys):
            row = (idx // 2) * 2
            col = idx % 2
            ax_lum = fig.add_subplot(gs[row, col])
            ax_res = fig.add_subplot(gs[row+1, col], sharex=ax_lum)

            # Plot luminosity for all commits
            for ref_name in self.commit_labels:
                if key in self.data[ref_name]:
                    wavelength = self.data[ref_name][key]['wavelength']
                    luminosity = self.data[ref_name][key]['luminosity']
                    ax_lum.plot(wavelength, luminosity, label=ref_name)

            ax_lum.set_ylabel('Luminosity')
            ax_lum.set_title(f'{key}')
            ax_lum.legend()
            ax_lum.grid(True)

            # Plot residuals relative to first commit
            if key in self.data[self.commit_labels[0]]:
                wavelength = self.data[self.commit_labels[0]][key]['wavelength']
                lum_ref0 = self.data[self.commit_labels[0]][key]['luminosity']
                for ref_name in self.commit_labels[1:]:
                    if key in self.data[ref_name]:
                        lum = self.data[ref_name][key]['luminosity']
                        residuals = np.where(lum_ref0 != 0, (lum - lum_ref0) / lum_ref0, 0)
                        ax_res.plot(wavelength, residuals, label=f'{ref_name} vs {self.commit_labels[0]}')

            ax_res.set_xlabel('Wavelength')
            ax_res.set_ylabel('Residuals')
            ax_res.legend()
            ax_res.grid(True)
            ax_lum.tick_params(axis='x', labelbottom=False)
            if row != 2:
                ax_res.tick_params(axis='x', labelbottom=False)

        plt.suptitle('Spectrum Solver Comparison Across Commits', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

class MultiCommitComparer:
    def __init__(self, num_commits=10, print_path=False):
        self.num_commits = num_commits
        self.print_path = print_path
        self.commit_hashes = get_last_n_commits(num_commits)
        if len(self.commit_hashes) < 2:
            raise ValueError("Not enough commits to compare.")
        self.file_manager = FileManager()
        self.file_setup = FileSetup(self.file_manager, self.commit_hashes)
        self.diff_analyzer = DiffAnalyzer(self.file_manager)
        self.hdf_comparator = HDFComparator(print_path=print_path)
        self.ref_paths = [self.file_manager.get_temp_path(f"ref{i+1}") for i in range(num_commits)]
        self.commit_labels = [f"Commit {i+1} ({h[:8]})" for i, h in enumerate(self.commit_hashes)]

    def setup(self):
        self.file_manager.setup()
        self.file_setup.setup()

    def teardown(self):
        self.file_manager.teardown()

    def compare(self, print_diff=False):
        for i in range(self.num_commits - 1):
            ref1_path = self.ref_paths[i]
            ref2_path = self.ref_paths[i+1]
            print(f"\nComparing {self.commit_labels[i]} vs {self.commit_labels[i+1]}")
            dcmp = dircmp(ref1_path, ref2_path)
            if print_diff:
                self.diff_analyzer.print_diff_files(dcmp)
            self.compare_hdf_files(ref1_path, ref2_path)

    def compare_hdf_files(self, ref1_path, ref2_path):
        for root, _, files in os.walk(ref1_path):
            for file in files:
                file_path = Path(file)
                if file_path.suffix in ('.h5', '.hdf5'):
                    rel_path = Path(root).relative_to(ref1_path)
                    ref2_file_path = ref2_path / rel_path / file
                    if ref2_file_path.exists():
                        self.hdf_comparator.summarise_changes_hdf(file, root, ref2_file_path.parent)

    def compare_spectrum_solvers(self, hdf_file_path="tardis/spectrum/tests/test_spectrum_solver/test_spectrum_solver/TestSpectrumSolver.h5"):
        ref_file_paths = [Path(ref_path) / hdf_file_path for ref_path in self.ref_paths]
        comparator = SpectrumSolverComparator(ref_file_paths, self.commit_labels)
        comparator.setup()
        comparator.plot_matplotlib()

if __name__ == "__main__":
    comparer = MultiCommitComparer(num_commits=10, print_path=True)
    try:
        comparer.setup()
        comparer.compare(print_diff=True)
        comparer.compare_spectrum_solvers()
    finally:
        comparer.teardown()