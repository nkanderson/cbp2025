#include "mlp.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

// Sigmoid activation function
static inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// MLPPredictor implementation
MLPPredictor::MLPPredictor(void)
    : global_history(0), history_length(0), hidden_size(0), bias_output(0.0) {
  // Initialize to safe defaults
  // Weights will be loaded in setup()
}

void MLPPredictor::setup() {
  if (!load_weights(MLP_WEIGHTS_FILE)) {
    std::cerr << "ERROR: Failed to load MLP weights from " << MLP_WEIGHTS_FILE
              << std::endl;
    std::exit(1);
  }
  std::cout << "MLP predictor initialized: " << history_length << " inputs, "
            << hidden_size << " hidden neurons" << std::endl;
}

bool MLPPredictor::load_weights(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "ERROR: Cannot open weights file: " << filename << std::endl;
    return false;
  }

  std::vector<std::string> lines;
  std::string line;

  // Read all lines from file
  while (std::getline(file, line)) {
    // Skip empty lines
    if (line.empty() ||
        line.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;
    }
    lines.push_back(line);
  }
  file.close();

  if (lines.empty()) {
    std::cerr << "ERROR: Weights file is empty" << std::endl;
    return false;
  }

  // Last line is the output neuron, all others are hidden neurons
  hidden_size = lines.size() - 1;

  if (hidden_size == 0) {
    std::cerr << "ERROR: No hidden layer neurons found" << std::endl;
    return false;
  }

  // Parse first hidden neuron line to determine number of inputs
  std::istringstream first_line(lines[0]);
  std::vector<double> first_neuron_weights;
  double weight;
  while (first_line >> weight) {
    first_neuron_weights.push_back(weight);
  }

  if (first_neuron_weights.size() < 2) {
    std::cerr
        << "ERROR: Hidden neuron must have at least 1 input weight + 1 bias"
        << std::endl;
    return false;
  }

  // Number of inputs = total weights - 1 (bias)
  history_length = first_neuron_weights.size() - 1;

  // Resize weight matrices
  weights_hidden.resize(hidden_size);
  bias_hidden.resize(hidden_size);

  // Parse hidden layer neurons
  for (size_t i = 0; i < hidden_size; i++) {
    std::istringstream hidden_neuron_input(lines[i]);
    weights_hidden[i].resize(history_length);

    // Read input weights
    for (size_t j = 0; j < history_length; j++) {
      if (!(hidden_neuron_input >> weights_hidden[i][j])) {
        std::cerr << "ERROR: Failed to read weight for hidden neuron " << i
                  << " input " << j << std::endl;
        return false;
      }
    }

    // Read bias
    if (!(hidden_neuron_input >> bias_hidden[i])) {
      std::cerr << "ERROR: Failed to read bias for hidden neuron " << i
                << std::endl;
      return false;
    }

    // Check for unexpected extra values
    double extra;
    if (hidden_neuron_input >> extra) {
      std::cerr << "WARNING: Hidden neuron " << i
                << " has extra values (expected " << history_length
                << " weights + 1 bias)" << std::endl;
    }
  }

  // Parse output neuron (last line)
  std::istringstream output_neuron(lines[hidden_size]);
  weights_output.resize(hidden_size);

  // Read weights from hidden layer
  for (size_t i = 0; i < hidden_size; i++) {
    if (!(output_neuron >> weights_output[i])) {
      std::cerr << "ERROR: Failed to read output weight " << i << std::endl;
      return false;
    }
  }

  // Read output bias
  if (!(output_neuron >> bias_output)) {
    std::cerr << "ERROR: Failed to read output bias" << std::endl;
    return false;
  }

  // Check for unexpected extra values
  double extra;
  if (output_neuron >> extra) {
    std::cerr << "WARNING: Output neuron has extra values (expected "
              << hidden_size << " weights + 1 bias)" << std::endl;
  }

  return true;
}

void MLPPredictor::terminate() {}

bool MLPPredictor::predict() {
  // Forward propagation through hidden layer
  std::vector<double> hidden_outputs(hidden_size);

  for (size_t i = 0; i < hidden_size; ++i) {
    // Compute weighted sum (MAC operation) for hidden neuron i
    double sum = 0.0;

    // Add weighted inputs from global history
    for (size_t j = 0; j < history_length; ++j) {
      // Extract bit j from global_history (bit 0 is most recent)
      bool history_bit = (global_history >> j) & 1;
      // Since we know branch history is binary, this allows for some
      // optimization over a typical multiply-accumulation operation.
      // This is particularly useful in translation to hardware, where
      // we can avoid multiplication and instead conditionally add the weight
      // using a MUX or AND gate.
      // If history bit is 1 (taken), add the weight; if 0 (not taken), skip:
      if (history_bit) {
        sum += weights_hidden[i][j];
      }
    }

    // Add bias term
    sum += bias_hidden[i];

    // In hardware, we would likely use a lookup table for sigmoid
    // approximation, or switch to a simpler activation function like ReLU.
    // Apply activation function:
    hidden_outputs[i] = sigmoid(sum);
  }

  // Forward propagation through output layer
  double output_sum = 0.0;

  // Add weighted hidden outputs
  for (size_t i = 0; i < hidden_size; ++i) {
    output_sum += hidden_outputs[i] * weights_output[i];
  }

  // Add bias term
  output_sum += bias_output;

  // Apply activation function and threshold at 0.5
  double output = sigmoid(output_sum);
  return output >= 0.5;
}

void MLPPredictor::history_update(bool taken) {
  // Shift global history left and insert new outcome at LSB (bit 0)
  // This keeps the most recent outcome at bit 0
  global_history = (global_history << 1) | (taken ? 1 : 0);

  // Mask to keep only the lower history_length bits
  global_history &= ((1ULL << history_length) - 1);
}

// Global predictor instance
MLPPredictor mlp_predictor_impl;
