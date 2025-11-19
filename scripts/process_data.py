"""
    process_data.py - prepares training data for MLP predictor
    
    Description:
        This script loads in a CSV file, shuffles the data, and extracts a specified
        percentage of the data, taking target indicies of each line. The processed data
        is then saved to a new CSV file.
"""

import csv
import argparse
import random


# Indices to target in each line of the CSV (resolved direction and history register)
target_indicies = [0, 3]


class ProcessData:
    """Encapsulate CSV processing parameters and operations."""

    def __init__(self, input_file, output_file, percentage, seed):
        self.input_file = input_file
        self.output_file = output_file
        self.percentage = percentage
        self.seed = seed

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
        total_lines = len(data)

        # Shuffle data with specified seed
        shuffled_data = self.shuffle_data(data)

        # Calculate number of lines to process based on percentage
        num_lines_to_process = int((self.percentage / 100.0) * total_lines)

        # Extract target columns from the specified number of lines
        processed_data = []

        try:
            for i in range(num_lines_to_process):
                line = shuffled_data[i]
                if not line:
                    continue  # Skip empty lines
                extracted_line = [line[index] for index in target_indicies]
                processed_data.append(extracted_line)
        except IndexError as e:
            print(f"IndexError encountered: {e}. Check if target indices are valid for the data.")

        processed_set = set(tuple(row) for row in processed_data)

        # Write processed data to output CSV file
        with open(self.output_file, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(processed_set)