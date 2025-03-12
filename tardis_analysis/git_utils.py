# tardis_analysis/git_utils.py

import subprocess
import os
from git import Repo

def process_commits(tardis_repo_path, regression_data_repo_path, branch, target_file, n=3):
    """Process commits from tardis repo, run tests, and commit regression data."""
    target_file_path = os.path.join(regression_data_repo_path, target_file)
    tardis_repo = Repo(tardis_repo_path)
    regression_repo = Repo(regression_data_repo_path)

    original_head = regression_repo.head.commit.hexsha
    print(f"Original HEAD of regression data repo: {original_head}")

    commits = list(tardis_repo.iter_commits(branch, max_count=n))
    commits.reverse()

    processed_commits = []
    regression_commits = []

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

            if not os.path.exists(target_file_path):
                print(f"Error: HDF5 file {target_file_path} was not generated.")
                continue

            regression_repo.git.add(A=True)
            regression_commit = regression_repo.index.commit(f"Regression data for tardis commit {i}")
            regression_commits.append(regression_commit.hexsha)
            processed_commits.append(commit.hexsha)
        except subprocess.CalledProcessError as e:
            print(f"Error running pytest for commit {commit.hexsha}: {e}")
            print("Pytest stdout:")
            print(e.stdout)
            print("Pytest stderr:")
            print(e.stderr)
            continue  # Skip to the next commit

    print("\nProcessed Tardis Commits:")
    for hash in processed_commits:
        print(hash)

    print("\nRegression Data Commits:")
    for hash in regression_commits:
        print(hash)

    return processed_commits, regression_commits, original_head, target_file_path