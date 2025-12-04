"""
    process_data.py - prepares training data for MLP predictor
    
    Description:
        This script loads in a CSV file, shuffles the data, and extracts a specified
        percentage of the data, taking target indicies of each line. The processed data
        is then saved to a new CSV file.
        
        Note: Script uses random to shuffle data, takes first N% of lines, and collapses
        list into a set to remove repeat lines.
"""

import csv
import random
import pandas as pd
import numpy as np

# Indices to target in each line of the CSV (resolved direction and history register)
target_indicies = [0, 3]

class ProcessData:
    """Encapsulate CSV processing parameters and operations."""

    def __init__(self, input_file, output_file, percentage=100.0, seed=None, input_size=64, dedup=True, isProcessed=False):
        self.input_file = input_file
        self.output_file = output_file
        self.percentage = percentage
        self.seed = seed
        self.input_size = input_size
        self.dedup = dedup
        self.isProcessed = isProcessed

    def process(self):
        """Process the CSV according to the configured parameters and write the output CSV."""
        # Read CSV with pandas
        df = pd.read_csv(self.input_file, header=None)
        
        # Shuffle if seed is provided
        if self.seed is not None:
            df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        # Extract target columns and apply mask if not already processed
        if not self.isProcessed:
            df = df.iloc[:, target_indicies]
            history_mask = np.uint64((1 << self.input_size) - 1)
            df.iloc[:, 1] = df.iloc[:, 1].astype('uint64') & history_mask
        
        # Remove duplicates
        if self.dedup:
            df = df.drop_duplicates()
        
        # Take percentage of data
        if self.percentage < 100.0:
            n_rows = int(len(df) * self.percentage / 100)
            df = df.iloc[:n_rows]
        
        # Write output
        df.to_csv(self.output_file, index=False, header=False)