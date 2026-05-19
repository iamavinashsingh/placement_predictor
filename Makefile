# ─────────────────────────────────────────────────────────────────────────────
#  Makefile — Advanced C++ Neural Network Project
#
#  TECHNIQUE: Performance Tuning
#    -O3          → maximum compiler optimisation (loop unrolling, inlining)
#    -march=native→ generate instructions for the host CPU (enables AVX/SSE)
#    -std=c++17   → required for <filesystem>
#    -pthread     → required for std::thread on Linux/macOS
#    -flto        → Link-Time Optimisation (cross-translation-unit inlining)
#
#  Targets:
#    make            → build both train.exe and predict.exe
#    make train      → build train.exe only
#    make predict    → build predict.exe only
#    make clean      → remove all build artifacts
#    make run        → train, then run predict interactively
# ─────────────────────────────────────────────────────────────────────────────

CXX      = g++

ifeq ($(OS),Windows_NT)
    PTHREAD_FLAGS =
else
    PTHREAD_FLAGS = -pthread
endif

# --- Performance flags ---
CXXFLAGS = -std=c++17 \
            -O3 \
            -march=native \
            -flto \
            -Wall -Wextra \
            $(PTHREAD_FLAGS)

# Include directory where all our .h files live
INCLUDES = -I.

# Linker flags (pthread for multi-threading on Linux/macOS)
LDFLAGS  = $(PTHREAD_FLAGS)

# Source and header files
TRAIN_SRC   = train.cpp
PREDICT_SRC = predict.cpp

HEADERS = include/exceptions.h \
           include/logger.h     \
           include/matrix.h     \
           include/dataset.h    \
           include/layer.h      \
           include/model.h      \
           include/trainer.h    \
           include/distributed.h

# Output binaries
ifeq ($(OS),Windows_NT)
    TRAIN_BIN   = train.exe
    PREDICT_BIN = predict.exe
    RM          = del /Q
else
    TRAIN_BIN   = train
    PREDICT_BIN = predict
    RM          = rm -f
endif

# ─────────────────────────────────────────────────────────────────────────────
#  Targets
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: all train predict clean run distributed help

all: train predict

train: $(TRAIN_SRC) $(HEADERS)
	@echo "[BUILD] Compiling train..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(TRAIN_SRC) -o $(TRAIN_BIN) $(LDFLAGS)
	@echo "[BUILD] Done → $(TRAIN_BIN)"

predict: $(PREDICT_SRC) $(HEADERS)
	@echo "[BUILD] Compiling predict..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(PREDICT_SRC) -o $(PREDICT_BIN) $(LDFLAGS)
	@echo "[BUILD] Done → $(PREDICT_BIN)"

# Run standard multi-threaded training
run: train predict
	./$(TRAIN_BIN) placement_data.csv --epochs 500 --lr 0.01
	./$(PREDICT_BIN)

# Run distributed (parameter-server) training
distributed: train
	./$(TRAIN_BIN) placement_data.csv --distributed --workers 4 --epochs 200

clean:
	$(RM) $(TRAIN_BIN) $(PREDICT_BIN) model.bin scaler.bin train.log
	@echo "[CLEAN] Build artifacts removed"

help:
	@echo ""
	@echo "  Usage:"
	@echo "    make              - build train + predict"
	@echo "    make run          - build and run training then predict"
	@echo "    make distributed  - build and run distributed training"
	@echo "    make clean        - remove binaries and artifacts"
	@echo ""
	@echo "  train.exe flags:"
	@echo "    --epochs N        set number of epochs (default 500)"
	@echo "    --lr F            set learning rate  (default 0.01)"
	@echo "    --threads N       set worker threads (default auto)"
	@echo "    --batch N         set mini-batch size (default 32)"
	@echo "    --distributed     use parameter-server training"
	@echo "    --workers N       number of distributed workers (default 4)"
	@echo ""
