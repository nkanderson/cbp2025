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

uint64_t PerceptronPredictor::get_unique_inst_id(uint64_t seq_no,
                                                 uint8_t piece) const {
  // Combine seq_no and piece into a unique 64-bit ID
  // piece is 4 bits (0-15), so we shift seq_no left by 4 bits
  return (seq_no << 4) | (piece & 0x0F);
}

// See Jiménez and Lin paper, sections 3.2 and 3.5 for prediction algorithm
bool PerceptronPredictor::predict(uint64_t seq_no, uint8_t piece, uint64_t PC) {
  // CHECKPOINT: Save current history for this instruction instance
  // This allows us to retrieve the exact history state later during training.
  // The method for maintaining and retreiving checkpointed histories is
  // modelled on the example in my_cond_branch_predictor.
  uint64_t inst_id = get_unique_inst_id(seq_no, piece);
  pred_time_histories[inst_id] = global_history;

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

// NOTE: We need to update the history only after training is done,
// so this function should be used with care.
void PerceptronPredictor::history_update(uint64_t seq_no, uint8_t piece,
                                         uint64_t PC, bool taken,
                                         uint64_t nextPC) {
  // Shift global history left and insert new outcome at LSB (bit 0)
  // This keeps the most recent outcome at bit 0
  global_history = (global_history << 1) | (taken ? 1 : 0);

  // Mask to keep only the lower HISTORY_LENGTH bits (62 bits)
  global_history &= ((1ULL << HISTORY_LENGTH) - 1);
}

// See Jiménez and Lin paper, sections 3.3 and 3.5 for training algorithm
void PerceptronPredictor::update(uint64_t seq_no, uint8_t piece, uint64_t PC,
                                 bool resolveDir, bool predDir,
                                 uint64_t nextPC) {

  // RETRIEVE: Get the checkpointed history from prediction time
  uint64_t inst_id = get_unique_inst_id(seq_no, piece);
  uint64_t prediction_time_history = pred_time_histories.at(inst_id);

  // Get the perceptron for this PC
  size_t index = get_perceptron_index(PC);
  Perceptron &perceptron = perceptron_table[index];

  // Calculate the perceptron output using PREDICTION-TIME history
  // (NOT the current global_history, which may have been updated by other
  // branches)
  int32_t output = perceptron.weights[0]; // Start with bias

  for (size_t i = 0; i < HISTORY_LENGTH; i++) {
    bool history_bit = (prediction_time_history >> i) & 1;
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

    // Update weights based on correlation with PREDICTION-TIME history
    for (size_t i = 0; i < HISTORY_LENGTH; i++) {
      bool history_bit = (prediction_time_history >> i) & 1;

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

  // CLEANUP: Remove the checkpoint for this instruction (no longer needed)
  pred_time_histories.erase(inst_id);

  // Update global history AFTER training
  history_update(seq_no, piece, PC, resolveDir, nextPC);
}

// Global predictor instance
PerceptronPredictor perceptron_predictor_impl;
