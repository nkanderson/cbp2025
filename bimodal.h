/* Prototypes and constants for bimodal predictor */

#ifndef _BIMODAL_H_
#define _BIMODAL_H_

#include <stdlib.h>
#include <cstdint>
#include <array>
#include <unordered_map>

static constexpr size_t BIMODAL_TABLE_SIZE = 4096;  // Table size
static constexpr u_int64_t INDEX_MASK = 0xFFF;      // Mask for lower bits of PC (log2 of table size -1)


class bimodalPredictor {

    std::array<uint8_t, BIMODAL_TABLE_SIZE> pred_state_table = {0};
                
    public:

        bimodalPredictor(void);

        void setup();

        void terminate();

        bool predict(uint64_t seq_no, uint8_t piece, uint64_t PC);

        // Not used in bimodal predictor
        void history_update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool taken,
                        uint64_t nextPC);

        void update(uint64_t seq_no, uint8_t piece, uint64_t PC, bool resolveDir,
                bool predDir, uint64_t nextPC);

    private:

        // Not used in bimodal predictor
        uint64_t get_unique_inst_id(uint64_t seq_no, uint8_t piece) const;

        size_t get_bimodal_index(uint64_t PC) const;
};

// Global predictor instance
extern bimodalPredictor bimodal_predictor_impl;

#endif // _BIMODAL_H_