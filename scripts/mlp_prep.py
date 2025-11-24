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
import re

# Helper functions for argument parsing
def rel_path(p):
    p = Path(p)
    return p if p.is_absolute() else (Path.cwd() / p)

def restricted_percent(p):
    p = float(p)
    if p < 0.0 or p > 100.0:
        raise argparse.ArgumentTypeError("Percentage must be between 0 and 100.")
    return p

def restricted_size(s):
    s = int(s)
    if s <= 0:
        raise argparse.ArgumentTypeError("Input size must be a positive integer.")
    elif s > 64:
        raise argparse.ArgumentTypeError("Input size must be less than or equal to 64.")
    return s

# Set up argument parser
parser = argparse.ArgumentParser(description="Workflow script for MLP data preparation")
parser.add_argument("--trace_dir", type=rel_path, help="Path to the input trace directory, if simulation is disabled, this is used for the input CSV", required=True)
parser.add_argument("--output_dir", type=rel_path, help="Path to the output files", required=True)
parser.add_argument("--percentage", type=restricted_percent, default=100.0,
                    help="Percentage to take from original log (default: 100.0)")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for shuffling (default: 42)")
parser.add_argument("--input_size", type=restricted_size, default=64, 
                    help="Input size for MLP (default: 64)")
# Skip simulation flag - if set, does not run the trace execution, but needs a valid CSV to be passed as an argument
parser.add_argument("--skip_sim", action="store_true",
                    help="If set, skips the trace execution step and uses CSV in trace_dir")
args = parser.parse_args()

if args.skip_sim:
    # If skipping simulation, ensure the provided trace_dir is a valid CSV file, not a directory
    assert args.trace_dir.is_file(), f"When skipping simulation, trace_dir must be a valid CSV file. Given: {args.trace_dir}"

relative_path = Path.cwd()

# Gather PATH information
base_path = Path(__file__).parent.resolve()
simulator_path = Path(base_path.parent / "cbp_data")
trace_dir_path = Path(args.trace_dir)
output_dir_path = Path(args.output_dir)

search_pattern = re.compile(r'^branch_history_log_p[^/]*')

# Skip simulation, use provided CSV directly
if args.skip_sim:
    default_csv = trace_dir_path
    print(f"Skip Simulation Mode Enabled."
        f"\nCSV path: {default_csv}"
        f"\nOutput directory path: {output_dir_path}")
# For normal operation, use the default concatenated CSV path
else:
    default_csv = Path(base_path.parent / "branch_history_log.csv")
    print(f"Normal Simulation Mode Enabled."
        f"Simulator path: {simulator_path}"
        f"\nTrace directory path: {trace_dir_path}"
        f"\nOutput directory path: {output_dir_path}"
        f"\nDefault concatenated CSV path: {default_csv}")

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
    """
        Process workflow:
            1. Run traces using specified simulator (unless skipped)
            2. Create a list of processed CSV logs
            3. De-dup each log and parse for target columns
            4. Concatenate processed logs into a single CSV file
            5. Save processed CSV file to output directory with timestamp
    """
    
    # Run traces using the specified simulator (implementation not shown)
    if not args.skip_sim:
        run_trace_script()
        # Gather all generated CSV files matching the pattern
        file_matches = [p.resolve() for p in base_path.parent.iterdir() if p.is_file() and search_pattern.match(p.name)]
    else:
        file_matches = [default_csv]
        # Create output directory if it doesn't exist
        output_path = output_dir_path / "traceresults"
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Check that all searched files exist
    for f in file_matches:
        print(f"Found generated CSV file: {f}")
        assert(f.is_file()), f"Expected output CSV {default_csv} does not exist."
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Rename the current output directory (do not create a new one)
    current_output_dir = output_dir_path / "traceresults"
    assert current_output_dir.exists() and current_output_dir.is_dir(), f"Expected output directory {current_output_dir} does not exist."

    renamed_output_dir = output_dir_path / f"traceresults_{timestamp}"
    current_output_dir.rename(renamed_output_dir)
    print(f"Renamed output directory to: {renamed_output_dir}")
    
    renamed_csv_paths = []
    renamed_processed_paths = []
    
    # Move/rename the CSVs into the renamed folder with timestamp
    for index, f in enumerate(file_matches):
        f = Path(f)  # convert string → Path

        name, ext = os.path.splitext(f.name)
        new_csv_path = renamed_output_dir / f"{name}_{timestamp}{ext}"
        
        # Extract PID number from filename
        pid = re.search(r'p(\d+)', name)
        pid_str = pid.group(1) if pid else "unknown"
        new_processed_path = renamed_output_dir / f"training_data_p{pid_str}_{timestamp}{ext}"

        f.rename(new_csv_path)
        renamed_csv_paths.append(new_csv_path)
        renamed_processed_paths.append(new_processed_path)

    # Print renamed paths    
    for index, f in enumerate(renamed_processed_paths):
        print(f"Input CSV [{[index]}]: {renamed_csv_paths[index]}")
        print(f"Output CSV [{[index]}]: {f}")
        
    # Process the generated CSV file
    for index, f in enumerate(renamed_csv_paths):
        processor = ProcessData(
            input_file=str(renamed_csv_paths[index]),
            output_file=str(renamed_processed_paths[index]),
            dedup=True
        )

        processor.process()

        print(f"Processed training data saved to: {renamed_processed_paths[index]}")
        
    # Combine all processed CSV files into a single CSV
    combined_output_path = output_dir_path / f"mlp_training_data_{timestamp}.csv"
    with open(combined_output_path, mode='w', newline='', encoding='utf-8') as combined_file:
        for processed_file in renamed_processed_paths:
            with open(processed_file, mode='r', newline='', encoding='utf-8') as pf:
                combined_file.write(pf.read())
                
    # Processed combined file path with processor with parameters, no dedup
    print(f"Combined processed training data saved to: {combined_output_path}")
    processor = ProcessData(
        input_file=str(combined_output_path),
        output_file=str(combined_output_path),
        percentage=args.percentage,
        seed=args.seed,
        input_size=args.input_size,
        dedup=False
    )

if __name__=="__main__":
    main()