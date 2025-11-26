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
import argparse
import random


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

    def read_csv(self):
        """Read entire CSV file and return a list of rows. Each row is represented as a list."""
        with open(self.input_file, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            return list(reader)

    def shuffle_data(self, data):
        """Shuffle the input list using the instance's random seed."""
        random.seed(self.seed)
        random.shuffle(data)
        return data

    def process(self):
        """Process the CSV according to the configured parameters and write the output CSV."""
        # Read data from input CSV file
        data = self.read_csv()

        # Shuffle data with specified seed
        if self.seed is not None:
            shuffled_data = self.shuffle_data(data)
        else:
            shuffled_data = data
        
        # Extract target columns from the specified number of lines
        processed_data = []
        
        # Parse for target columns
        for row in shuffled_data:
            if self.isProcessed:
                # If data is already processed, use the row as is
                extracted_row = row
            else:
                extracted_row = [row[index] for index in target_indicies]
            
            # Mask history value to keep only the lowest `input_size` bits, but keep decimal representation
            hist_str = extracted_row[1].strip()
            history_value = int(hist_str, 10)
            history_mask = (1 << self.input_size) - 1
            masked = history_value & history_mask
            extracted_row[1] = str(masked)
            
            processed_data.append(extracted_row)
            
        # Remove duplicate rows by converting to a set of tuples
        if self.dedup:
            processed_set = set(tuple(row) for row in processed_data)
        else:
            processed_set = processed_data
        
        # Remove percentage of set 
        num_unique_lines = len(processed_set)
        num_lines_to_take_from_set = int((self.percentage / 100) * num_unique_lines)
        processed_set = list(processed_set)[:num_lines_to_take_from_set]

        # Write processed data to output CSV file
        with open(self.output_file, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(processed_set)