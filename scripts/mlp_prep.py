"""
    mlp_prep.py - prepares training data for MLP predictor
    
    Description:
        This script executes a given trace directory using a specified simulator,
        grabs the output concatenated CSV file, processes the data, and saves the
        processed data to a new CSV file for MLP training.

"""

import os
import argparse
from process_data import ProcessData
from pathlib import Path
import subprocess
import time

def rel_path(p):
    p = Path(p)
    return p if p.is_absolute() else (Path.cwd() / p)

# Set up argument parser
parser = argparse.ArgumentParser(description="Workflow script for MLP data preparation")
parser.add_argument("--trace_dir", type=rel_path, help="Path to the input trace directory", required=True)
parser.add_argument("--output_dir", type=rel_path, help="Path to the output files", required=True)
parser.add_argument("--percentage", type=float, default=100.0,
                    help="Percentage to take from original log (default: 100.0)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for shuffling (default: 42)")
args = parser.parse_args()


relative_path = Path.cwd()

# Gather PATH information
base_path = Path(__file__).parent.resolve()
simulator_path = Path(base_path.parent / "cbp_data")
trace_dir_path = Path(args.trace_dir)
output_dir_path = Path(args.output_dir)
default_concatenated_csv = Path(base_path.parent / "branch_history_log.csv")

print(f"Simulator path: {simulator_path}"
      f"\nTrace directory path: {trace_dir_path}"
      f"\nOutput directory path: {output_dir_path}"
      f"\nDefault concatenated CSV path: {default_concatenated_csv}")

def run_trace_script():
    # Run a subprocess to execute external Python script for trace execution
    
    # Check if simulator, trace directory exist
    assert(simulator_path.is_file()), f"Simulator path {simulator_path} does not exist."
    assert(trace_dir_path.is_dir()), f"Trace directory {trace_dir_path} does not exist."    
    
    exec_path = base_path / "trace_exec_training_list.py"
    output_path = output_dir_path / "traceresults"
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Try to run the subprocess and capture output
    try:
        result = subprocess.run(
            [
                "python3",
                exec_path,
                "--simulator_path",
                str(simulator_path),
                "--trace_dir",
                str(trace_dir_path),
                "--results_dir",
                str(output_path)    
            ],
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running traces.\n"
              f"Command: {e.cmd}\n"
              f"Return code: {e.returncode}\n"
              f"Stderr output:\n{e.stderr}")
        
    print(result.stdout)
    print(result.stderr)
        

def main():
    # Run traces using the specified simulator (implementation not shown)
    run_trace_script()
    
    # Rename output CSV file to timestamped version
    assert(default_concatenated_csv.is_file()), f"Expected output CSV {default_concatenated_csv} does not exist."
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Rename the current output directory (do not create a new one)
    current_output_dir = output_dir_path / "traceresults"
    assert current_output_dir.exists() and current_output_dir.is_dir(), f"Expected output directory {current_output_dir} does not exist."

    renamed_output_dir = output_dir_path / f"traceresults_{timestamp}"
    current_output_dir.rename(renamed_output_dir)

    # Move/rename the concatenated CSV into the renamed folder with timestamp
    timestamped_csv_path = renamed_output_dir / f"branch_history_log_{timestamp}.csv"
    default_concatenated_csv.rename(timestamped_csv_path)

    # Prepare processed output path inside the renamed folder
    processed_path = renamed_output_dir / f"training_data_{timestamp}.csv"

    # Process the generated CSV file
    processor = ProcessData(
        input_file=str(timestamped_csv_path),
        output_file=str(processed_path),
        percentage=args.percentage,
        seed=args.seed
    )

    processor.process()

    print(f"Processed training data saved to: {processed_path}")

if __name__=="__main__":
    main()