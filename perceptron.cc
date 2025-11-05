#include "perceptron.h"
#include <cstdint>

// Perceptron implementation
Perceptron::Perceptron() {
  // Initialize all weights (including bias at index 0) to 0
  weights.fill(0);
}

// PerceptronPredictor implementation
PerceptronPredictor::PerceptronPredictor(void) : global_history(0) {
  // Initialize global history to 0 (all branches not-taken)
}

void PerceptronPredictor::setup() {
  // Initialize perceptron table - all weights start at 0
  // This happens automatically via Perceptron constructor
}

void PerceptronPredictor::terminate() {}

size_t PerceptronPredictor::get_perceptron_index(uint64_t PC) const {
  // Simple modulo hash to map PC to table index
  // Can be improved with better hashing if needed
  return PC % PERCEPTRON_TABLE_SIZE;
}

bool PerceptronPredictor::predict(uint64_t seq_no, uint8_t piece, uint64_t PC) {
  // Get the perceptron for this PC
  size_t index = get_perceptron_index(PC);
  const Perceptron &perceptron = perceptron_table[index];

  // Calculate perceptron output: y = bias + sum(weight_i * history_i)
  int32_t output = perceptron.weights[0]; // Start with bias

  // Add contribution from each history bit
  for (size_t i = 0; i < HISTORY_LENGTH; i++) {
    // Extract bit i from global_history (bit 0 is most recent)
    bool history_bit = (global_history >> i) & 1;

    // If history bit is 1 (taken), add the weight; if 0 (not taken), subtract
    if (history_bit) {
      output += perceptron.weights[i + 1]; // +1 because bias is at index 0
    } else {
      output -= perceptron.weights[i + 1];
    }
  }

  // Predict taken if output >= 0, not taken otherwise
  return output >= 0;
}

void PerceptronPredictor::history_update(uint64_t seq_no, uint8_t piece,
                                         uint64_t PC, bool taken,
                                         uint64_t nextPC) {
  // Shift global history left and insert new outcome at LSB (bit 0)
  // This keeps the most recent outcome at bit 0
  global_history = (global_history << 1) | (taken ? 1 : 0);

  // Mask to keep only the lower HISTORY_LENGTH bits (62 bits)
  global_history &= ((1ULL << HISTORY_LENGTH) - 1);
}

void PerceptronPredictor::update(uint64_t seq_no, uint8_t piece, uint64_t PC,
                                 bool resolveDir, bool predDir,
                                 uint64_t nextPC) {
  // Get the perceptron for this PC
  size_t index = get_perceptron_index(PC);
  Perceptron &perceptron = perceptron_table[index];

  // Calculate the perceptron output using CURRENT history
  // (the history that was used to make the original prediction)
  int32_t output = perceptron.weights[0]; // Start with bias

  for (size_t i = 0; i < HISTORY_LENGTH; i++) {
    bool history_bit = (global_history >> i) & 1;
    if (history_bit) {
      output += perceptron.weights[i + 1];
    } else {
      output -= perceptron.weights[i + 1];
    }
  }

  // Determine if we need to train:
  // 1. Misprediction: predDir != resolveDir
  // 2. Weak prediction: |output| <= threshold
  bool mispredicted = (predDir != resolveDir);
  bool weak_prediction = (output >= 0 ? output : -output) <= THETA;

  if (mispredicted || weak_prediction) {
    // Update bias (index 0)
    // Increment if taken, decrement if not taken
    if (resolveDir) {
      // Branch was taken: increment bias (saturate at INT16_MAX)
      if (perceptron.weights[0] < INT16_MAX) {
        perceptron.weights[0]++;
      }
    } else {
      // Branch was not taken: decrement bias (saturate at INT16_MIN)
      if (perceptron.weights[0] > INT16_MIN) {
        perceptron.weights[0]--;
      }
    }

    // Update weights based on correlation with history
    for (size_t i = 0; i < HISTORY_LENGTH; i++) {
      bool history_bit = (global_history >> i) & 1;

      // Increment weight if:
      // - resolveDir == taken AND history_bit == taken (both 1)
      // - resolveDir == not taken AND history_bit == not taken (both 0)
      // Otherwise decrement
      bool should_increment = (resolveDir == history_bit);

      if (should_increment) {
        // Increment, but saturate at +THETA
        if (perceptron.weights[i + 1] < THETA) {
          perceptron.weights[i + 1]++;
        }
      } else {
        // Decrement, but saturate at -THETA
        if (perceptron.weights[i + 1] > -THETA) {
          perceptron.weights[i + 1]--;
        }
      }
    }
  }

  // Update global history AFTER training (so we train with the correct history)
  // Shift global history left and insert new outcome at LSB (bit 0)
  global_history = (global_history << 1) | (resolveDir ? 1 : 0);

  // Mask to keep only the lower HISTORY_LENGTH bits (62 bits)
  global_history &= ((1ULL << HISTORY_LENGTH) - 1);
}

// Global predictor instance
PerceptronPredictor perceptron_predictor_impl;
