import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import glob
from datetime import datetime
import matplotlib.patches as mpatches
import numpy as np
import sys
import re 

def create_bar_charts(directory_path, group_by_col, group_by_display_name, main_category_col, subtitle=None):
    """
    Recursively searches a directory for CSV files, combines the data, extracts
    TblSize and HistLen from subdirectory names if missing, and generates bar charts.
    
    Args:
        directory_path (str): The base path to the directory containing the input CSV files.
        group_by_col (str or None): The technical column name used for secondary grouping.
        group_by_display_name (str or None): The user-friendly label for the grouping column.
        main_category_col (str): The primary column for categorization (e.g., 'Workload' or 'Run').
        subtitle (str, optional): A string to be used as a subtitle for the charts.
    """
    
    # Setup Output Directory
    output_dir = os.path.join(directory_path, 'charts')
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Error: Could not create output directory '{output_dir}'. {e}")
        sys.exit(1)

    # Load and Combine Data (MODIFIED FOR RECURSION)
    all_dfs = []
    
    # Use os.walk to recursively search all subdirectories
    print(f"Recursively searching for CSV files in: {directory_path}")
    for root, dirs, files in os.walk(directory_path):
        # Skip the output directory
        if root == output_dir:
            continue
            
        # Regex to capture two consecutive numbers in the subdirectory name (e.g., results_64_10)
        # Finds the first number (TblSize) and the second number (HistLen) separated by an underscore
        match = re.search(r'(\d+)_(\d+)', os.path.basename(root))
        
        # Default values for TblSize and HistLen extracted from directory name
        dir_tblsize = int(match.group(1)) if match and len(match.groups()) >= 1 else None
        dir_histlen = int(match.group(2)) if match and len(match.groups()) >= 2 else None

        # Direct iteration over files and case-insensitive check
        for file_name in files:
            if file_name.lower().endswith(".csv"):
                file_path = os.path.join(root, file_name)
                
                try:
                    df = pd.read_csv(file_path)
                                        
                    # Check for TblSize
                    if 'TblSize' not in df.columns:
                        if dir_tblsize is not None:
                            df['TblSize'] = dir_tblsize
                        else:
                            df['TblSize'] = 'N/A' 
                    
                    # Check for HistLen
                    if 'HistLen' not in df.columns:
                        if dir_histlen is not None:
                            df['HistLen'] = dir_histlen
                        else:
                            df['HistLen'] = 'N/A' 

                    all_dfs.append(df)
                    
                except Exception as e:
                    print(f"Warning: Could not read {file_path}. Skipping. Error: {e}")

    if not all_dfs:
        print(f"Error: No valid CSV files were found in or under the directory: {directory_path}")
        sys.exit(1)
        
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Combined data from {len(all_dfs)} file(s).")
    
    # Check for required columns
    required_cols = [main_category_col, 'IPC', 'MPKI', 'MR']
    if group_by_col:
        required_cols.append(group_by_col)

    if not all(col in df.columns for col in required_cols):
        print(f"Error: Combined data is missing one or more required columns: {required_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    try:
        # Data Cleanup and Aggregation
        df['MR_Clean'] = df['MR'].astype(str).str.replace('%', '', regex=False).astype(float) / 100.0
        
        # Conditional Aggregation
        if group_by_col:
            # Two-Level Grouping
            df[group_by_col] = df[group_by_col].astype(str)
            
            grouped_data_unstacked = df.groupby([group_by_col, main_category_col])[['IPC', 'MPKI', 'MR_Clean']].mean().unstack(fill_value=0)
            
            if len(grouped_data_unstacked.index) == 0:
                print("Error: The requested group-by column is empty")
                sys.exit(1)
            
            main_categories = grouped_data_unstacked.columns.get_level_values(1).unique().tolist()
            x_labels = grouped_data_unstacked.index.tolist()
            plot_title_suffix = f"by {group_by_display_name}"
            
        else:
            # Single-Level Grouping
            grouped_data_flat = df.groupby(main_category_col)[['IPC', 'MPKI', 'MR_Clean']].mean().reset_index()
            main_categories = grouped_data_flat[main_category_col].tolist()
            x_labels = main_categories 
            plot_title_suffix = f"by {main_category_col}"
            
        num_categories = len(main_categories)
        
        # Setup for Plotting
        timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
        bar_colors = plt.cm.get_cmap('tab10').colors[:num_categories]

        metrics = {
            'IPC': {'label': 'Instructions Per Cycle (IPC)', 'scale': 1.0, 'unit': ''},
            'MPKI': {'label': 'Misses Per Kilo Instruction (MPKI)', 'scale': 1.0, 'unit': ''},
            'MR_Clean': {'label': 'Miss Rate (MR) (%)', 'scale': 100.0, 'unit': ' %'} 
        }

        # Generate Charts
        for metric_key, props in metrics.items():
            
            fig, ax = plt.subplots(figsize=(12, 7)) 
            
            if group_by_col:
                # Plotting: Grouped Bars 
                num_factors = len(x_labels)
                bar_width = 0.8 / num_categories
                x_base = np.arange(num_factors)
                
                for i, category in enumerate(main_categories):
                    x_pos = x_base + (i - (num_categories - 1) / 2) * bar_width
                    values = grouped_data_unstacked[metric_key, category].values * props['scale']
                    
                    rects = ax.bar(x_pos, values, bar_width, color=bar_colors[i], label=category)
                    
                    # Add data labels
                    for rect in rects:
                        height = rect.get_height()
                        if height != 0:
                            ax.text(rect.get_x() + rect.get_width() / 2., 
                                    height + ax.get_ylim()[1] * 0.005,
                                    '%.2f' % height,
                                    ha='center', va='bottom', fontsize=8)

                ax.set_xticks(x_base)
                ax.set_xticklabels(x_labels, rotation=0, ha='center') 
                ax.set_xlabel(group_by_display_name, fontsize=12) 

            else:
                # Plotting: Single-Level Bars
                values = grouped_data_flat[metric_key].values * props['scale']
                
                rects = ax.bar(main_categories, values, color=bar_colors)
                
                # Add data labels
                for rect in rects:
                    height = rect.get_height()
                    if height != 0:
                        ax.text(rect.get_x() + rect.get_width() / 2., 
                                height + ax.get_ylim()[1] * 0.005,
                                '%.2f' % height,
                                ha='center', va='bottom', fontsize=10)
                
                ax.set_xticks(np.arange(num_categories))
                ax.set_xticklabels(x_labels, rotation=0, ha='center') 
                ax.set_xlabel(main_category_col, fontsize=12)
            
            # inal Plot Styling (common to both)
            main_title = f'{props["label"]} Comparison {plot_title_suffix}'
            
            if subtitle:
                full_title = f'{main_title}\n({subtitle})'
            else:
                full_title = main_title
                
            ax.set_title(full_title, fontsize=14)
            ax.set_ylabel(f'{props["label"]}{props["unit"]}', fontsize=12)
            
            # Add legend
            legend_patches = [mpatches.Patch(color=c, label=w) for c, w in zip(bar_colors, main_categories)]
            ax.legend(handles=legend_patches, 
                      title=main_category_col, 
                      loc='center left',
                      bbox_to_anchor=(1.05, 0.5),
                      frameon=False)

            ax.grid(axis='y', linestyle='--', alpha=0.7)
            fig.tight_layout() 

            # Save the graph
            filename_key = metric_key.replace('_Clean', '').lower()
            filename_category = main_category_col.lower()
            filename_group = group_by_col.lower() if group_by_col else "single_level"
            output_filename = f'{filename_key}_by_{filename_group}_by_{filename_category}{timestamp}.png'
            
            # Save the file to the new output directory
            plt.savefig(os.path.join(output_dir, output_filename)) 
            print(f"Bar chart for {props['label']} successfully generated and saved to '{os.path.join(output_dir, output_filename)}'")
            
        print("\nAll required charts have been generated.")

    except Exception as e:
        print(f"An unexpected error occurred during processing or plotting: {e}")
        if 'KeyError' in str(e):
             print("\nNote: A KeyError often means a column name is missing or misspelled.")


# Execution
if __name__ == "__main__":
    
    # Mapping for technical column names to display labels
    DISPLAY_NAME_MAPPING = {
        'TblSize': 'Table Size',
        'HistLen': 'History Length',
        'LayerSize': 'Layer Size'
    }
    
    parser = argparse.ArgumentParser(
        description="Generate bar charts by aggregating data from all CSV files within a specified directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        'directory_path', 
        type=str, 
        help="The path to the directory containing the input CSV files (e.g., ./data/)."
    )
    
    parser.add_argument(
        '--sub', 
        type=str, 
        default=None,
        help='Optional subtitle string to be included below the main chart title.'
    )
    
    parser.add_argument('--run', action='store_true', help='Use the "Run" column instead of "Workload" as the primary category (legend and single-level X-axis).')
    
    # Mutually Exclusive Group (Optional) for Grouping Column
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--hist', action='store_true', help='Group bars by the "HistLen" column.')
    group.add_argument('--table', action='store_true', help='Group bars by the "TblSize" column.')
    group.add_argument('--layer', action='store_true', help='Group bars by the "LayerSize" column.')
    
    args = parser.parse_args()
    
    # Determine the grouping column and its display name
    group_by_col = None
    group_by_display_name = None
    
    if args.hist:
        group_by_col = 'HistLen'
    elif args.table:
        group_by_col = 'TblSize'
    elif args.layer:
        group_by_col = 'LayerSize'
        
    if group_by_col:
        group_by_display_name = DISPLAY_NAME_MAPPING.get(group_by_col, group_by_col)

    # Determine the main category column
    main_category_col = 'Run' if args.run else 'Workload'

    # Ensure the path is a valid directory
    if not os.path.isdir(args.directory_path):
        print(f"Error: '{args.directory_path}' is not a valid directory.")
        sys.exit(1)
        
    # Pass all determined arguments to the main function
    create_bar_charts(args.directory_path, group_by_col, group_by_display_name, main_category_col, args.sub)