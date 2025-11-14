#include "bimodal.h"
#include <cstdint>

bimodalPredictor::bimodalPredictor(void) {
};

void bimodalPredictor::setup(){};

void bimodalPredictor::terminate(){};

// This function is here in case needed, but not currently used
uint64_t bimodalPredictor::get_unique_inst_id(uint64_t seq_no, uint8_t piece) const {
    // Build inst_id
    return (seq_no << 4) | (piece & 0x0F);
};

// Gets index of table based on PC
size_t bimodalPredictor::get_bimodal_index(uint64_t PC) const {
    // Mask lower bits of PC to generate table index
    return PC & INDEX_MASK;
}

bool bimodalPredictor::predict(uint64_t seq_no, uint8_t piece, uint64_t PC) {

    bool pred_taken;

    size_t index = get_bimodal_index(PC);

    // Check counter for current branch and predict accordingly
    if (pred_state_table[index] > 1){
        pred_taken = true;
    }
    else {
        pred_taken = false;
    }

    return pred_taken;

};

void bimodalPredictor::update(uint64_t seq_no, uint8_t piece, uint64_t PC,
                                 bool resolveDir, bool predDir, uint64_t nextPC) {

    size_t index = get_bimodal_index(PC);
    
    // Update saturating counter based on branch direction
    if (resolveDir){
        if (pred_state_table[index] < 3){
            pred_state_table[index]++;
        } 
    }
    else {
        if (pred_state_table[index] > 0){
            pred_state_table[index]--;
        }
    }

};

// Global predictor instance
bimodalPredictor bimodal_predictor_impl;
