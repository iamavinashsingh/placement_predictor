/**
 * @file train.cpp
 * @brief Training Entry Point
 *
 * This file ties all the advanced techniques together:
 *
 *   Usage:
 *     train.exe [csv_path] [--distributed] [--threads N] [--epochs N] [--lr F]
 *
 *   Flags:
 *     --distributed   Use the ParameterServer (Distributed Systems demo)
 *     --threads N     Number of worker threads (default: all CPU cores)
 *     --epochs N      Training epochs (default: 500)
 *     --lr F          Learning rate (default: 0.01)
 *
 *  ALL TECHNIQUES DEMONSTRATED IN CALL ORDER:
 *   1. STL Advanced     → CLI argument parsing with std::string/find
 *   2. File Systems     → Dataset::fromCSV uses <filesystem> internally
 *   3. OOP Design       → Sequential model with Layer hierarchy
 *   4. Memory Mgmt      → unique_ptr layers, Matrix contiguous memory
 *   5. Concurrency      → Logger mutex, atomic loss in Trainer
 *   6. Multi-threading  → Trainer parallelises mini-batches across threads
 *   7. Performance      → -O3 -march=native flags in Makefile
 *   8. Error Recovery   → Typed exceptions + checkpoint resume
 *   9. Distributed      → ParameterServer with worker threads (--distributed)
 */

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>       // for timing   ← Performance Tuning

#include "include/exceptions.h"
#include "include/logger.h"
#include "include/dataset.h"
#include "include/matrix.h"
#include "include/layer.h"
#include "include/model.h"
#include "include/trainer.h"
#include "include/distributed.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

// STL Advanced Usage — parse a named integer argument from argv
// Example: getIntArg(argc, argv, "--threads", 4)  returns 4 if not set
int getIntArg(int argc, char** argv, const std::string& flag, int def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == flag) {
            try { return std::stoi(argv[i + 1]); }
            catch (...) { return def; }
        }
    return def;
}
double getDblArg(int argc, char** argv, const std::string& flag, double def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == flag) {
            try { return std::stod(argv[i + 1]); }
            catch (...) { return def; }
        }
    return def;
}
bool hasFlag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag) return true;
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Build the neural network architecture
//  5 inputs → Dense(16) → ReLU → Dense(10) → Softmax
// ─────────────────────────────────────────────────────────────────────────────
std::unique_ptr<Sequential> buildModel(std::mt19937& rng) {
    auto model = std::make_unique<Sequential>();

    model->add(std::make_unique<DenseLayer>(5, 16, rng));   // layer 0
    model->add(std::make_unique<ReLULayer>());               // layer 1
    model->add(std::make_unique<DenseLayer>(16, 10, rng));  // layer 2
    model->add(std::make_unique<SoftmaxLayer>());            // layer 3

    model->summary();
    return model;
}

// ─────────────────────────────────────────────────────────────────────────────
//  main()
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    // TECHNIQUE: Error Recovery — wrap everything in a top-level try/catch
    // so the user always gets a clean error message, never a crash dump
    try {
        // ── Logger setup ──────────────────────────────────────────────
        Logger::instance().setLogFile("train.log");
        Logger::instance().setLevel(LogLevel::INFO);
        LOG_INFO("============================================================");
        LOG_INFO(" Placement Predictor — Advanced C++ Neural Network Trainer");
        LOG_INFO("============================================================");

        // ── Parse CLI arguments (STL Advanced Usage) ──────────────────
        std::string csvPath   = (argc > 1 && argv[1][0] != '-')
                                    ? argv[1] : "placement_data.csv";
        bool  useDistributed  = hasFlag(argc, argv, "--distributed");
        int   numThreads      = getIntArg(argc, argv, "--threads", 0);
        int   epochs          = getIntArg(argc, argv, "--epochs",  500);
        double lr             = getDblArg(argc, argv, "--lr",      0.01);
        int   batchSize       = getIntArg(argc, argv, "--batch",    32);
        int   numWorkers      = getIntArg(argc, argv, "--workers",   4);

        LOG_INFO("Config → csv: "  + csvPath +
                 " | epochs: "     + std::to_string(epochs) +
                 " | lr: "         + std::to_string(lr) +
                 " | threads: "    + (numThreads == 0 ? "auto" : std::to_string(numThreads)) +
                 " | distributed: "+ (useDistributed ? "YES" : "no"));

        // ── Load & split dataset (File Systems + Error Recovery) ───────
        Dataset full = Dataset::fromCSV(csvPath, 5 /*features*/);
        std::mt19937 rng(42);
        std::pair<Dataset, Dataset> splitResult = full.split(0.8, rng);
        Dataset& trainSet = splitResult.first;
        Dataset& testSet  = splitResult.second;

        LOG_INFO("Split: " + std::to_string(trainSet.size()) +
                 " train / " + std::to_string(testSet.size()) + " test");

        // ── Normalise (fit on train, apply to both) ────────────────────
        auto scaler = trainSet.fitScaler();
        trainSet.applyScaler(scaler);
        testSet.applyScaler(scaler);

        // ── Save scaler for use during inference ───────────────────────
        {
            std::ofstream sf("scaler.bin", std::ios::binary);
            Dataset::saveScaler(scaler, sf);
            LOG_INFO("Scaler saved to scaler.bin");
        }

        // ── Build model (OOP Design + Memory Management) ───────────────
        auto model = buildModel(rng);

        // ── Performance Tuning: time the full training run ─────────────
        auto t0 = std::chrono::high_resolution_clock::now();

        if (useDistributed) {
            // ── TECHNIQUE: Distributed Systems ────────────────────────
            DistributedTrainer dtrainer(numWorkers, lr, epochs, batchSize);
            dtrainer.train(*model, trainSet, testSet);
        } else {
            // ── TECHNIQUE: Multi-threading + Concurrency ───────────────
            TrainConfig cfg;
            cfg.learningRate       = lr;
            cfg.epochs             = epochs;
            cfg.batchSize          = batchSize;
            cfg.numThreads         = numThreads;
            cfg.logInterval        = 50;
            cfg.checkpointInterval = 100;

            Trainer trainer(cfg);
            trainer.train(*model, trainSet, testSet);
        }

        // ── Performance measurement ────────────────────────────────────
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        LOG_INFO("Training wall-clock time: " + std::to_string(ms) + " ms");

        // ── Final evaluation ───────────────────────────────────────────
        double trainAcc = model->accuracy(trainSet);
        double testAcc  = model->accuracy(testSet);
        LOG_INFO("Final Train Accuracy : " + std::to_string(trainAcc).substr(0,5) + "%");
        LOG_INFO("Final Test  Accuracy : " + std::to_string(testAcc).substr(0,5) + "%");

        // ── Save model (File Systems) ──────────────────────────────────
        model->saveBinary("model.bin");
        LOG_INFO("Done! Run predict.exe to use the model.");

    } catch (const FileException& e) {
        // TECHNIQUE: Error Recovery — specific handlers give actionable info
        std::cerr << "\n[FILE ERROR] " << e.what() << "\n";
        std::cerr << "  → Check the CSV path and file permissions.\n";
        return 1;
    } catch (const DataException& e) {
        std::cerr << "\n[DATA ERROR] " << e.what() << "\n";
        std::cerr << "  → Check that the CSV has 6 columns (5 features + 1 label).\n";
        return 2;
    } catch (const TrainingException& e) {
        std::cerr << "\n[TRAINING ERROR] " << e.what() << "\n";
        std::cerr << "  → Try --lr 0.001 (lower learning rate) or check your data.\n";
        return 3;
    } catch (const NNException& e) {
        std::cerr << "\n[NN ERROR] " << e.what() << "\n";
        return 4;
    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL] " << e.what() << "\n";
        return 99;
    }

    return 0;
}
