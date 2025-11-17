#ifndef _MLP_H_
#define _MLP_H_

#include <array>
#include <cstdint>
#include <stdlib.h>
#include <string>
#include <unordered_map>
#include <vector>

// MLP configuration - can be overridden at compile time
#ifndef MLP_WEIGHTS_FILE
#define MLP_WEIGHTS_FILE "mlp_2_4.txt"
#endif

class MLPPredictor {
  // Global History Register - stores recent branch outcomes as a bitfield
  // Bit 0 (LSB) represents the most recent branch outcome
  // 1 = taken, 0 = not taken
  uint64_t global_history;

  // MLP architecture parameters (loaded from weights file)
  size_t history_length; // Number of history bits used as input
  size_t hidden_size;    // Number of hidden layer neurons

  // MLP weights
  // weights_hidden: input-to-hidden weights (history_length x hidden_size)
  // weights_output: hidden-to-output weights (hidden_size x 1)
  // bias_hidden: hidden layer biases (hidden_size)
  // bias_output: output layer bias (scalar)
  std::vector<std::vector<double>> weights_hidden;
  std::vector<double> weights_output;
  std::vector<double> bias_hidden;
  double bias_output;

public:
  MLPPredictor(void);

  void setup();

  void terminate();

  bool predict();

  void history_update(bool taken);

  // Getters for architecture parameters
  size_t get_history_length() const { return history_length; }
  size_t get_hidden_size() const { return hidden_size; }

private:
  // Load weights from file
  bool load_weights(const std::string &filename);
};

// Global predictor instance
extern MLPPredictor mlp_predictor_impl;

#endif // _MLP_H_
