"""
Perceptron parameter sweep visualization script.
Creates heatmaps showing miss rates for different table sizes and history lengths.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse

# Default configuration
DEFAULT_TABLE_SIZES = [64, 128, 256, 512, 1024, 2048]
DEFAULT_HISTORY_LENGTHS = [0, 1, 10, 20, 30, 40, 50, 60, 62, 64]

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_BASE_DIR = PROJECT_ROOT / "perceptron_data"
IMAGES_DIR = PROJECT_ROOT / "images"

# Workload mapping
WORKLOAD_NAMES = {
    "fp": "Floating Point",
    "media": "Media",
    "infra": "Infra",
    "web": "Web",
    "compress": "Compression",
    "int": "Integer",
}


def load_results(table_sizes, history_lengths):
    """
    Load miss rate results from all configuration directories.

    Returns a dictionary mapping (table_size, history_length) -> average miss rate
    """
    results = {}

    for table_size in table_sizes:
        for history_length in history_lengths:
            results_dir = RESULTS_BASE_DIR / f"results_{table_size}_{history_length}"
            results_csv = results_dir / "results.csv"

            if not results_csv.exists():
                print(
                    f"Warning: Results not found for table_size={table_size}, history_length={history_length}"
                )
                results[(table_size, history_length)] = np.nan
                continue

            try:
                df = pd.read_csv(results_csv)

                # Calculate average miss rate across all runs
                # Prefer 50PercMR (warmup-adjusted) over MR (full run)
                mr_column = "50PercMR" if "50PercMR" in df.columns else "MR"
                if mr_column in df.columns:
                    # Strip '%' sign and convert to float
                    avg_mr = df[mr_column].str.rstrip("%").astype(float).mean()
                    results[(table_size, history_length)] = avg_mr
                else:
                    print(f"Warning: No MR column found in {results_csv}")
                    results[(table_size, history_length)] = np.nan

            except Exception as e:
                print(f"Error loading {results_csv}: {e}")
                results[(table_size, history_length)] = np.nan

    return results


def create_heatmap(results, table_sizes, history_lengths, output_file=None):
    """
    Create a heatmap visualization of miss rates.

    Args:
        results: Dictionary mapping (table_size, history_length) -> miss rate
        table_sizes: List of table sizes (rows, top to bottom)
        history_lengths: List of history lengths (columns, left to right)
        output_file: Optional path to save the figure
    """
    # Create a 2D array for the heatmap
    # Rows = table sizes, Columns = history lengths
    heatmap_data = np.zeros((len(table_sizes), len(history_lengths)))

    for i, table_size in enumerate(table_sizes):
        for j, history_length in enumerate(history_lengths):
            heatmap_data[i, j] = results.get((table_size, history_length), np.nan)

    # Create DataFrame for better labeling
    df_heatmap = pd.DataFrame(
        heatmap_data,
        index=[f"{ts}" for ts in table_sizes],
        columns=[f"{hl}" for hl in history_lengths],
    )

    # Calculate figure dimensions that should work with square cells
    n_cols = len(history_lengths)
    n_rows = len(table_sizes)
    cell_size = 1.0  # Base cell size in inches
    fig_width = n_cols * cell_size + 3  # +3 for colorbar and labels
    fig_height = n_rows * cell_size + 2  # +2 for title and labels

    plt.figure(figsize=(fig_width, fig_height))

    # Create heatmap with annotations
    # Use viridis colormap (yellow for high, dark purple for low)
    sns.heatmap(
        df_heatmap,
        annot=True,
        fmt=".3f",
        cmap="viridis",  # Yellow (high) to dark purple (low)
        cbar_kws={"label": "Miss Rate (%)"},
        linewidths=0.5,
        linecolor="gray",
        square=True,  # Force square cells
    )

    # Set labels
    plt.xlabel("History Length", fontsize=12, fontweight="bold")
    plt.ylabel("Table Size", fontsize=12, fontweight="bold")
    plt.title(
        "Perceptron Branch Predictor Miss Rates\n(Lower is Better)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # Tight layout
    plt.tight_layout()

    # Save and show
    if output_file:
        plt.savefig(output_file, format="svg", bbox_inches="tight")
        print(f"Heatmap saved to {output_file}")

    plt.show()

    # Find and print best configuration
    min_mr = np.nanmin(heatmap_data)

    # Check if we have any valid data
    if not np.isnan(min_mr):
        min_idx = np.where(heatmap_data == min_mr)
        best_table_size = table_sizes[min_idx[0][0]]
        best_history_length = history_lengths[min_idx[1][0]]

        print(f"\n{'='*60}")
        print("Best Configuration:")
        print(f"  Table Size: {best_table_size}")
        print(f"  History Length: {best_history_length}")
        print(f"  Miss Rate: {min_mr:.4f}")
        print(f"{'='*60}\n")
    else:
        print("\nWarning: No valid data found in results.\n")


def load_results_by_workload(table_sizes, history_lengths):
    """
    Load miss rate results organized by workload.

    Returns a nested dictionary:
        {table_size: {(workload, history_length): miss_rate}}
    """
    results_by_table = {}

    for table_size in table_sizes:
        results_by_table[table_size] = {}

        for history_length in history_lengths:
            results_dir = RESULTS_BASE_DIR / f"results_{table_size}_{history_length}"
            results_csv = results_dir / "results.csv"

            if not results_csv.exists():
                continue

            try:
                df = pd.read_csv(results_csv)

                # Get miss rate column
                mr_column = "50PercMR" if "50PercMR" in df.columns else "MR"
                if mr_column not in df.columns:
                    continue

                # Extract workload from Run column and get miss rate for each
                for _, row in df.iterrows():
                    run_name = row["Run"]
                    # Extract workload prefix (e.g., "fp_0" -> "fp")
                    workload = run_name.split("_")[0]

                    if workload in WORKLOAD_NAMES:
                        # Strip '%' sign if present and convert to float
                        miss_rate_str = str(row[mr_column])
                        if miss_rate_str.endswith("%"):
                            miss_rate = float(miss_rate_str.rstrip("%"))
                        else:
                            miss_rate = float(miss_rate_str)
                        results_by_table[table_size][
                            (workload, history_length)
                        ] = miss_rate

            except Exception as e:
                print(f"Error loading {results_csv}: {e}")
                continue

    return results_by_table


def create_workload_heatmaps(
    results_by_table,
    table_sizes,
    history_lengths,
    output_file=None,
    shared_scale=True,
    n_cols=2,
):
    """
    Create a grid of heatmaps, one for each table size showing workload performance.

    Args:
        results_by_table: Nested dict {table_size: {(workload, history_length): miss_rate}}
        table_sizes: List of table sizes
        history_lengths: List of history lengths
        output_file: Path to save the combined figure
        shared_scale: If True, all heatmaps use the same color scale (default: True)
        n_cols: Number of columns in the grid (default: 2)
    """
    # Get all unique workloads present in the data
    all_workloads = set()
    for table_results in results_by_table.values():
        for workload, _ in table_results.keys():
            all_workloads.add(workload)

    workloads = sorted(all_workloads)

    if not workloads:
        print("No workload data found!")
        return

    # Filter to only table sizes that have data
    valid_table_sizes = [
        ts for ts in table_sizes if ts in results_by_table and results_by_table[ts]
    ]

    if not valid_table_sizes:
        print("No valid table size data found!")
        return

    # Calculate grid dimensions
    n_tables = len(valid_table_sizes)
    n_cols_actual = min(n_cols, n_tables)  # Use provided n_cols parameter
    n_rows = (n_tables + n_cols_actual - 1) // n_cols_actual  # Ceiling division

    # Create figure with subplots - calculate size for square cells
    # Each subplot needs to accommodate workloads x history_lengths
    n_workload_rows = len(workloads)
    n_hist_cols = len(history_lengths)
    subplot_width = n_hist_cols * 0.8 + 3  # 0.8 inch per cell + space for labels
    subplot_height = n_workload_rows * 0.8 + 2  # 0.8 inch per cell + space for labels

    fig, axes = plt.subplots(
        n_rows,
        n_cols_actual,
        figsize=(subplot_width * n_cols_actual, subplot_height * n_rows),
    )

    # Ensure axes is always a 2D array for consistent indexing
    if n_rows == 1 and n_cols_actual == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols_actual == 1:
        axes = axes.reshape(n_rows, n_cols_actual)

    # First pass: collect all data
    all_heatmap_data = []
    for table_size in valid_table_sizes:
        table_results = results_by_table[table_size]
        heatmap_data = np.full((len(workloads), len(history_lengths)), np.nan)

        for i, workload in enumerate(workloads):
            for j, history_length in enumerate(history_lengths):
                key = (workload, history_length)
                if key in table_results:
                    heatmap_data[i, j] = table_results[key]

        all_heatmap_data.append(heatmap_data)

    # Calculate global min/max across all heatmaps if using shared scale
    if shared_scale:
        global_min = np.nanmin([np.nanmin(data) for data in all_heatmap_data])
        global_max = np.nanmax([np.nanmax(data) for data in all_heatmap_data])
    else:
        global_min = None
        global_max = None

    # Create a heatmap for each table size
    for idx, table_size in enumerate(valid_table_sizes):
        row = idx // n_cols_actual
        col = idx % n_cols_actual
        ax = axes[row, col]

        heatmap_data = all_heatmap_data[idx]

        # Create DataFrame with readable labels
        df_heatmap = pd.DataFrame(
            heatmap_data,
            index=[WORKLOAD_NAMES.get(wl, wl) for wl in workloads],
            columns=[f"{hl}" for hl in history_lengths],
        )

        # Create heatmap with viridis colormap (yellow/high to dark purple/low)
        heatmap_kwargs = {
            "annot": True,
            "fmt": ".3f",
            "cmap": "viridis",
            "cbar_kws": {
                "label": "Miss Rate (%)",
                "shrink": 0.75,  # Shrink colorbar to better match heatmap height
            },
            "linewidths": 0.5,
            "linecolor": "gray",
            "ax": ax,
            "square": True,  # Force square cells
        }

        # Add vmin/vmax only if using shared scale
        if shared_scale:
            heatmap_kwargs["vmin"] = global_min
            heatmap_kwargs["vmax"] = global_max

        sns.heatmap(df_heatmap, **heatmap_kwargs)

        # Set labels
        ax.set_xlabel("History Length", fontsize=10, fontweight="bold")
        ax.set_ylabel("Workload Type", fontsize=10, fontweight="bold")
        ax.set_title(
            f"Table Size: {table_size}",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )

        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

    # Hide any unused subplots, like in the case of an odd number of table sizes
    for idx in range(len(valid_table_sizes), n_rows * n_cols_actual):
        row = idx // n_cols_actual
        col = idx % n_cols_actual
        axes[row, col].axis("off")

    # Overall title with reduced padding
    fig.suptitle(
        "Perceptron Miss Rates by Workload and Table Size",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )

    plt.tight_layout()

    # Save and show (save as SVG)
    if output_file:
        plt.savefig(output_file, format="svg", bbox_inches="tight")
        print(f"Workload heatmaps saved to {output_file}")

    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize perceptron parameter sweep results"
    )
    parser.add_argument(
        "--table_sizes",
        type=int,
        nargs="+",
        default=DEFAULT_TABLE_SIZES,
        help=f"Table sizes to visualize (default: {DEFAULT_TABLE_SIZES})",
    )
    parser.add_argument(
        "--history_lengths",
        type=int,
        nargs="+",
        default=DEFAULT_HISTORY_LENGTHS,
        help=f"History lengths to visualize (default: {DEFAULT_HISTORY_LENGTHS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for the average heatmap (default: images/perceptron_average_heatmap.svg)",
    )
    parser.add_argument(
        "--workload_output",
        type=str,
        default=None,
        help="Output file path for the workload heatmap grid (default: images/perceptron_workload_heatmap.svg)",
    )
    parser.add_argument(
        "--independent_scales",
        action="store_true",
        help="Use independent color scales for each workload heatmap (default: shared scale)",
    )

    args = parser.parse_args()

    # Create images directory if it doesn't exist
    IMAGES_DIR.mkdir(exist_ok=True)

    # Set default output paths if not provided
    avg_output = (
        args.output if args.output else IMAGES_DIR / "perceptron_average_heatmap.svg"
    )
    workload_output = (
        args.workload_output
        if args.workload_output
        else IMAGES_DIR / "perceptron_workload_heatmap.svg"
    )

    print("Perceptron Parameter Sweep Visualization")
    print(f"Results directory: {RESULTS_BASE_DIR}")
    print(f"Images directory: {IMAGES_DIR}")
    print(f"Table sizes: {args.table_sizes}")
    print(f"History lengths: {args.history_lengths}")
    print()

    # Load results for average heatmap
    print("Loading results for average heatmap...")
    results = load_results(args.table_sizes, args.history_lengths)

    # Create average visualization
    print("Creating average heatmap...")
    create_heatmap(results, args.table_sizes, args.history_lengths, avg_output)

    # Load results by workload
    print("\nLoading results by workload...")
    results_by_workload = load_results_by_workload(
        args.table_sizes, args.history_lengths
    )

    # Create workload heatmaps with 3 columns (landscape)
    print("Creating workload-specific heatmaps (3 columns - landscape)...")
    workload_output_landscape = workload_output.parent / (
        workload_output.stem + "_landscape" + workload_output.suffix
    )
    create_workload_heatmaps(
        results_by_workload,
        args.table_sizes,
        args.history_lengths,
        workload_output_landscape,
        shared_scale=not args.independent_scales,
        n_cols=3,
    )

    # Create workload heatmaps with 2 columns (portrait)
    print("Creating workload-specific heatmaps (2 columns - portrait)...")
    workload_output_portrait = workload_output.parent / (
        workload_output.stem + "_portrait" + workload_output.suffix
    )
    create_workload_heatmaps(
        results_by_workload,
        args.table_sizes,
        args.history_lengths,
        workload_output_portrait,
        shared_scale=not args.independent_scales,
        n_cols=2,
    )


if __name__ == "__main__":
    main()
