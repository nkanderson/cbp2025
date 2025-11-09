#ifndef _PERCEPTRON_H_
#define _PERCEPTRON_H_

#include <array>
#include <cstdint>
#include <stdlib.h>
#include <unordered_map>
#include <vector>

// Perceptron configuration constants
static constexpr size_t PERCEPTRON_TABLE_SIZE = 1024;
static constexpr size_t HISTORY_LENGTH = 62;
static constexpr size_t WEIGHTS_PER_PERCEPTRON =
    HISTORY_LENGTH + 1; // +1 for bias at index 0

// Training threshold: theta = 1.93 * h + 14
// Using integer arithmetic: theta = (193 * h) / 100 + 14
static constexpr int THETA =
    (193 * HISTORY_LENGTH) / 100 + 14; // = 133 for h=62

// Structure to hold one perceptron's weights (bias + weights)
// Index 0: bias term
// Index 1-62: weights for each history bit
// Using int16_t to accommodate THETA = 133 (int8_t max is 127)
struct Perceptron {
  std::array<int16_t, WEIGHTS_PER_PERCEPTRON> weights;

  Perceptron();
};

class PerceptronPredictor {
  // Global History Register - stores last 62 branch outcomes as a bitfield
  // Bit 0 (LSB) represents the most recent branch outcome
  // 1 = taken, 0 = not taken
  // We use uint64_t which can hold 64 bits (we only use the lower 62 bits)
  uint64_t global_history;

  // Perceptron table indexed by PC
  // Each entry contains bias (index 0) + 62 weights (indices 1-62)
  std::array<Perceptron, PERCEPTRON_TABLE_SIZE> perceptron_table;

  // Checkpoint map: stores the history state at prediction time for each
  // in-flight branch
  // Key: unique instruction ID (seq_no, piece)
  // Value: history at prediction time
  std::unordered_map<uint64_t /*inst_id*/, uint64_t /*history*/>
      pred_time_histories;

public:
  PerceptronPredictor(void);

  void setup();

  void terminate();

  bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC);

  void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken,
                      uint64_t nextPC);

  void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir,
              bool predDir, uint64_t nextPC);

private:
  // Helper function to map PC to perceptron table index
  size_t get_perceptron_index(uint64_t PC) const;

  // Helper function to get unique instruction ID from seq_no and piece
  uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const;
};

// Global predictor instance
extern PerceptronPredictor perceptron_predictor_impl;

#endif // _PERCEPTRON_H_
