# /*
# 
# Copyright (c) 2019, North Carolina State University
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# 3. The names “North Carolina State University”, “NCSU” and any trade-name, personal name,
# trademark, trade device, service mark, symbol, image, icon, or any abbreviation, contraction or
# simulation thereof owned by North Carolina State University must not be used to endorse or promote products derived from this software without prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# */
# 
# // Author: Eric Rotenberg (ericro@ncsu.edu)

CC = g++
OPT = -O3
LIBS = -lcbp -lz
#FLAGS = -std=c++11 -L./lib $(LIBS) $(OPT)
FLAGS = -std=c++17 -L./lib $(LIBS) $(OPT)
CPPFLAGS = -std=c++17 $(OPT)

# MLP configuration
# Example override: make mlp MLP_WEIGHTS_FILE=mlp_10_20.txt
MLP_WEIGHTS_FILE ?= mlp_2_4.txt
MLP_FLAGS = -DMLP_WEIGHTS_FILE=\"$(MLP_WEIGHTS_FILE)\"

# Default object files and dependencies
OBJ = cond_branch_predictor_interface.o my_cond_branch_predictor.o
DEPS = cbp.h cond_branch_predictor_interface.h my_cond_branch_predictor.h

# Target-specific object files and dependencies
# Data generation dependencies
D_OBJ = datagen_interface.o my_cond_branch_predictor.o
D_DEPS = cbp.h datagen_interface.h my_cond_branch_predictor.h

ALWAYS_TAKEN_OBJ = always_taken_bp_interface.o always_taken.o
ALWAYS_TAKEN_DEPS = cbp.h always_taken.h

PERCEPTRON_OBJ = perceptron_interface.o perceptron.o
PERCEPTRON_DEPS = cbp.h perceptron.h

BIMODAL_OBJ = bimodal_interface.o bimodal.o
BIMODAL_DEPS = cbp.h bimodal.h

MLP_OBJ = mlp_interface.o mlp.o
MLP_DEPS = cbp.h mlp.h

DEBUG=0
ifeq ($(DEBUG), 1)
	CC += -ggdb3
endif

ifeq ($(MAKECMDGOALS),cbp_data)
    OBJ = $(D_OBJ)
    DEPS = $(D_DEPS)
endif

ifeq ($(MAKECMDGOALS),always_taken)
    OBJ = $(ALWAYS_TAKEN_OBJ)
    DEPS = $(ALWAYS_TAKEN_DEPS)
endif

ifeq ($(MAKECMDGOALS),perceptron)
    OBJ = $(PERCEPTRON_OBJ)
    DEPS = $(PERCEPTRON_DEPS)
endif

ifeq ($(MAKECMDGOALS),bimodal)
    OBJ = $(BIMODAL_OBJ)
    DEPS = $(BIMODAL_DEPS)
endif

ifeq ($(MAKECMDGOALS),mlp)
    OBJ = $(MLP_OBJ)
    DEPS = $(MLP_DEPS)
endif

.PHONY: clean lib data example always_taken perceptron bimodal mlp

# Default to building the provided example
all: example

lib:
	make -C $@ DEBUG=$(DEBUG)

# Example target (original my_cond_branch_predictor)
example: cbp_example

cbp_example: $(OBJ) | lib
	$(CC) $(FLAGS) -o $@ $^

# Training data generation target
cbp_data: $(OBJ) | lib
	$(CC) $(FLAGS) -o $@ $^

# Always taken target
always_taken: cbp_always_taken

cbp_always_taken: $(OBJ) | lib
	$(CC) $(FLAGS) -o $@ $^

# Perceptron target
perceptron: cbp_perceptron

cbp_perceptron: $(PERCEPTRON_OBJ) | lib
	$(CC) $(FLAGS) -o $@ $^

# Bimodal target
bimodal: cbp_bimodal

cbp_bimodal: $(BIMODAL_OBJ) | lib
	$(CC) $(FLAGS) -o $@ $^

# MLP target
mlp: cbp_mlp

cbp_mlp: $(MLP_OBJ) | lib
	$(CC) $(FLAGS) -o $@ $^

mlp_interface.o: mlp_interface.cc $(MLP_DEPS) # Include MLP_FLAGS
	$(CC) $(FLAGS) $(MLP_FLAGS) -c -o $@ $<

mlp.o: mlp.cc $(MLP_DEPS) # Include MLP_FLAGS
	$(CC) $(FLAGS) $(MLP_FLAGS) -c -o $@ $<

# Generic rule for building object files
%.o: %.cc $(DEPS)
	$(CC) $(FLAGS) -c -o $@ $<

clean:
	rm -f *.o cbp_example cbp_data cbp_always_taken cbp_perceptron cbp_bimodal cbp_mlp
	make -C lib clean
