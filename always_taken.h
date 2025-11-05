#ifndef _ALWAYS_TAKEN_H_
#define _ALWAYS_TAKEN_H_

#include <cstdint>
#include <stdlib.h>
#include <unordered_map>

struct SampleHist {
  uint64_t ghist;

  SampleHist();
};

class AlwaysTakenPredictor {
  SampleHist active_hist;
  std::unordered_map<uint64_t /*key*/, SampleHist /*val*/> pred_time_histories;

public:
  AlwaysTakenPredictor(void);

  void setup();

  void terminate();

  bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC);

  void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken,
                      uint64_t nextPC);

  void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir,
              bool predDir, uint64_t nextPC);

  void update(uint64_t PC, bool resolveDir, bool pred_taken, uint64_t nextPC,
              const SampleHist &hist_to_use);
};

// Global predictor instance
extern AlwaysTakenPredictor always_taken_predictor_impl;

#endif // _ALWAYS_TAKEN_H_
