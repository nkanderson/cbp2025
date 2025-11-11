#include "always_taken.h"

// SampleHist implementation
SampleHist::SampleHist() : ghist(0) {}

// AlwaysTakenPredictor implementation
AlwaysTakenPredictor::AlwaysTakenPredictor(void) {}

void AlwaysTakenPredictor::setup() {}

void AlwaysTakenPredictor::terminate() {}

bool AlwaysTakenPredictor::predict(uint64_t seq_no, uint8_t piece,
                                   uint64_t PC) {
  return true;
}

// NOTE: This is *not* used in an always-taken predictor, but is included
// here as an example based on my_cond_branch_predictor
void AlwaysTakenPredictor::history_update(uint64_t seq_no, uint8_t piece,
                                          uint64_t PC, bool taken,
                                          uint64_t nextPC) {
  active_hist.ghist = active_hist.ghist << 1;
  if (taken) {
    active_hist.ghist |= 1;
  }
}

void AlwaysTakenPredictor::update(uint64_t seq_no, uint8_t piece, uint64_t PC,
                                  bool resolveDir, bool predDir,
                                  uint64_t nextPC) {}

void AlwaysTakenPredictor::update(uint64_t PC, bool resolveDir, bool pred_taken,
                                  uint64_t nextPC,
                                  const SampleHist &hist_to_use) {}

// Global predictor instance
AlwaysTakenPredictor always_taken_predictor_impl;
