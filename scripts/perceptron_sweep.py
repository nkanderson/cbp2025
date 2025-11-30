"""
Perceptron parameter sweep script.
Builds the perceptron predictor with different table sizes and history lengths,
then runs trace_exec_training_list.py for each configuration.
"""

import os
import subprocess
import sys
from pathlib import Path

# Configuration
TABLE_SIZES = [64, 128, 256, 512, 1024, 2048]
HISTORY_LENGTHS = [0, 1, 10, 20, 30, 40, 50, 60, 62, 64]

# Paths (relative to project root)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
SIMULATOR_NAME = "cbp_perceptron"
SIMULATOR_PATH = PROJECT_ROOT / SIMULATOR_NAME
TRACE_DIR = PROJECT_ROOT / "perceptron_data" / "traces"
RESULTS_BASE_DIR = PROJECT_ROOT / "perceptron_data"
TRACE_EXEC_SCRIPT = SCRIPT_DIR / "trace_exec_training_list.py"


def build_perceptron(table_size, history_length):
    """Build the perceptron predictor with specified parameters."""
    print(f"\n{'='*80}")
    print(
        f"Building perceptron: table_size={table_size}, history_length={history_length}"
    )
    print(f"{'='*80}\n")

    # Change to project root for build
    os.chdir(PROJECT_ROOT)

    # Clean and build
    build_cmd = [
        "make",
        "clean",
        "&&",
        "make",
        "perceptron",
        f"PERCEPTRON_TABLE_SIZE={table_size}",
        f"PERCEPTRON_HISTORY_LENGTH={history_length}",
    ]

    try:
        result = subprocess.run(
            " ".join(build_cmd), shell=True, check=True, capture_output=True, text=True
        )
        print("Build successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Build failed with error:\n{e.stderr}")
        return False


def run_traces(table_size, history_length):
    """Run trace_exec_training_list.py with current configuration."""
    results_dir = RESULTS_BASE_DIR / f"results_{table_size}_{history_length}"

    print(f"\n{'='*80}")
    print(f"Running traces: table_size={table_size}, history_length={history_length}")
    print(f"Results directory: {results_dir}")
    print(f"{'='*80}\n")

    # Build command
    cmd = [
        sys.executable,  # Use same Python interpreter
        str(TRACE_EXEC_SCRIPT),
        "--simulator_path",
        str(SIMULATOR_PATH),
        "--trace_dir",
        str(TRACE_DIR),
        "--results_dir",
        str(results_dir),
    ]

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=False, text=True  # Show output in real-time
        )
        print(f"\nTraces completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTrace execution failed with error code {e.returncode}")
        return False


def main():
    """Main sweep loop."""
    print("Perceptron Parameter Sweep")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Simulator: {SIMULATOR_PATH}")
    print(f"Trace directory: {TRACE_DIR}")
    print(f"Results base directory: {RESULTS_BASE_DIR}")
    print(f"\nTable sizes: {TABLE_SIZES}")
    print(f"History lengths: {HISTORY_LENGTHS}")
    print(f"Total configurations: {len(TABLE_SIZES) * len(HISTORY_LENGTHS)}")

    # Verify paths exist
    if not TRACE_DIR.exists():
        print(f"\nERROR: Trace directory not found: {TRACE_DIR}")
        sys.exit(1)

    if not TRACE_EXEC_SCRIPT.exists():
        print(f"\nERROR: trace_exec_training_list.py not found: {TRACE_EXEC_SCRIPT}")
        sys.exit(1)

    # Track results
    successful_configs = []
    failed_configs = []

    # Sweep through configurations
    config_num = 0
    for table_size in TABLE_SIZES:
        for history_length in HISTORY_LENGTHS:
            config_num += 1
            print(f"\n\n{'#'*80}")
            print(
                f"# Configuration {config_num}/{len(TABLE_SIZES) * len(HISTORY_LENGTHS)}"
            )
            print(f"# Table size: {table_size}, History length: {history_length}")
            print(f"{'#'*80}")

            # Build with current configuration
            if not build_perceptron(table_size, history_length):
                print(f"Skipping traces due to build failure")
                failed_configs.append((table_size, history_length, "build_failed"))
                continue

            # Run traces
            if run_traces(table_size, history_length):
                successful_configs.append((table_size, history_length))
            else:
                failed_configs.append((table_size, history_length, "trace_failed"))

    # Print summary
    print(f"\n\n{'='*80}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*80}")
    print(
        f"Successful: {len(successful_configs)}/{len(TABLE_SIZES) * len(HISTORY_LENGTHS)}"
    )

    if failed_configs:
        print(f"\nFailed configurations:")
        for table_size, history_length, reason in failed_configs:
            print(
                f"  - table_size={table_size}, history_length={history_length} ({reason})"
            )

    print(f"\nResults saved to: {RESULTS_BASE_DIR}")


if __name__ == "__main__":
    main()
