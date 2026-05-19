/**
 * @file distributed.h
 * @brief Parameter Server — Distributed Systems
 *
 * TECHNIQUE: Distributed Systems
 *
 * Simulates real-world distributed training (e.g. Google DistBelief,
 * PyTorch DDP) using an in-process Parameter Server (PS) pattern.
 *
 *   ┌────────────────────────────────────┐
 *   │        Parameter Server (PS)       │
 *   │  Holds MASTER weights              │
 *   │  Aggregates worker gradients       │
 *   │  Broadcasts updated weights        │
 *   └──────────────┬─────────────────────┘
 *       push/pull  │
 *   Worker0  Worker1  Worker2  Worker3
 *  (shard0) (shard1) (shard2) (shard3)
 *
 * In production this uses gRPC/MPI/NCCL over the network.
 * Here we simulate using shared memory with Mutex-protected access.
 *
 * TECHNIQUE: Concurrency
 *   - Mutex protects the central weight store
 *   - AtomicInt tracks how many workers have completed an epoch
 *   - Workers are separate Thread objects (Win32 on Windows, pthreads on Linux)
 */

#pragma once

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>

#include "model.h"
#include "dataset.h"
#include "matrix.h"
#include "exceptions.h"
#include "logger.h"
#include "threads.h"

// ─────────────────────────────────────────────────────────────────────────────
//  ParameterServer — central weight manager
// ─────────────────────────────────────────────────────────────────────────────
class ParameterServer {
public:
    ParameterServer(int numWorkers, double lr)
        : numWorkers_(numWorkers), lr_(lr), pendingWorkers_(0), generation_(0)
    {}

    // ── Register a Dense layer's gradient shape ────────────────────────
    void registerLayer(int layerId, int outSize, int inSize) {
        LockGuard<Mutex> guard(mutex_);
        if (layerId >= static_cast<int>(masterW_.size())) {
            masterW_.resize(static_cast<size_t>(layerId + 1), Matrix());
            masterB_.resize(static_cast<size_t>(layerId + 1));
            aggDW_.resize(static_cast<size_t>(layerId + 1), Matrix());
            aggDB_.resize(static_cast<size_t>(layerId + 1));
        }
        masterW_[static_cast<size_t>(layerId)] = Matrix(outSize, inSize);
        masterB_[static_cast<size_t>(layerId)].assign(static_cast<size_t>(outSize), 0.0);
        aggDW_[static_cast<size_t>(layerId)]   = Matrix(outSize, inSize);
        aggDB_[static_cast<size_t>(layerId)].assign(static_cast<size_t>(outSize), 0.0);
    }

    // ── Copy model weights into PS master store ────────────────────────
    void setMasterWeights(Sequential& model) {
        LockGuard<Mutex> guard(mutex_);
        for (int l = 0; l < model.numLayers(); ++l) {
            DenseLayer* dense = dynamic_cast<DenseLayer*>(model.getLayer(l));
            if (!dense) continue;
            size_t li = static_cast<size_t>(l);
            if (li >= masterW_.size() || masterW_[li].rows() == 0) continue;
            masterW_[li] = dense->weights();
            masterB_[li] = dense->biases();
        }
    }

    // ── Worker pushes gradient contribution ───────────────────────────
    // "push" = analogous to gRPC StreamingCall in real distributed systems
    void pushGradient(int layerId, const Matrix& dW,
                      const std::vector<double>& db, int batchSize) {
        LockGuard<Mutex> guard(mutex_);
        size_t li = static_cast<size_t>(layerId);
        if (li >= aggDW_.size() || aggDW_[li].rows() == 0) return;
        double scale = 1.0 / batchSize;
        for (int i = 0; i < dW.rows(); ++i)
            for (int j = 0; j < dW.cols(); ++j)
                aggDW_[li](i, j) += dW(i, j) * scale;
        for (size_t i = 0; i < db.size(); ++i)
            aggDB_[li][i] += db[i] * scale;
    }

    // ── Signal epoch completion; last worker applies and broadcasts ────
    void workerEpochDone() {
        int gen = generation_.load();
        int arrived = ++pendingWorkers_;
        if (arrived == numWorkers_) {
            applyAndBroadcast();
            pendingWorkers_.store(0);
            ++generation_; // wake up the others
        } else {
            // Simple lock-free spin barrier
            while (generation_.load() == gen) {
                Sleep(1);
            }
        }
    }

    // ── Workers pull updated weights from PS ──────────────────────────
    void pullWeights(Sequential& model) {
        // No lock needed: all workers only READ master weights, and they
        // only do this after the barrier ensures applyAndBroadcast is done.
        for (int l = 0; l < model.numLayers(); ++l) {
            DenseLayer* dense = dynamic_cast<DenseLayer*>(model.getLayer(l));
            size_t li = static_cast<size_t>(l);
            if (!dense || li >= masterW_.size() || masterW_[li].rows() == 0) continue;
            dense->weights() = masterW_[li];
            dense->biases()  = masterB_[li];
        }
    }

private:
    // Apply averaged gradients from all workers to master weights
    void applyAndBroadcast() {
        double workerScale = 1.0 / numWorkers_;
        for (size_t l = 0; l < masterW_.size(); ++l) {
            if (masterW_[l].rows() == 0) continue;
            for (int i = 0; i < masterW_[l].rows(); ++i) {
                for (int j = 0; j < masterW_[l].cols(); ++j)
                    masterW_[l](i, j) -= lr_ * aggDW_[l](i, j) * workerScale;
                masterB_[l][static_cast<size_t>(i)] -= lr_ * aggDB_[l][static_cast<size_t>(i)] * workerScale;
            }
            aggDW_[l].zeros();
            std::fill(aggDB_[l].begin(), aggDB_[l].end(), 0.0);
        }
    }

    int             numWorkers_;
    double          lr_;
    AtomicInt       pendingWorkers_;
    AtomicInt       generation_;
    Mutex           mutex_;
    ConditionVariable cv_;

    std::vector<Matrix>              masterW_;
    std::vector<std::vector<double>> masterB_;
    std::vector<Matrix>              aggDW_;
    std::vector<std::vector<double>> aggDB_;
};

// ─────────────────────────────────────────────────────────────────────────────
//  DistributedTrainer — coordinates N workers + 1 ParameterServer
// ─────────────────────────────────────────────────────────────────────────────
class DistributedTrainer {
public:
    DistributedTrainer(int numWorkers, double lr, int epochs, int batchSize)
        : numWorkers_(numWorkers), epochs_(epochs), batchSize_(batchSize),
          lr_(lr), ps_(numWorkers, lr)
    {
        LOG_INFO("=== Distributed Training Mode ===");
        LOG_INFO("Workers: "  + std::to_string(numWorkers) +
                 " | Epochs: " + std::to_string(epochs) +
                 " | Simulates " + std::to_string(numWorkers) + " remote nodes");
    }

    // Train using data-parallel distributed SGD
    void train(Sequential& masterModel, const Dataset& fullDataset,
               const Dataset& testSet)
    {
        registerLayers(masterModel);
        ps_.setMasterWeights(masterModel);
        std::vector<Dataset> shards = partitionData(fullDataset, numWorkers_);

        std::vector<double> workerLoss(static_cast<size_t>(numWorkers_), 0.0);
        std::vector<Thread*> workers;

        for (int w = 0; w < numWorkers_; ++w) {
            auto* thr = new Thread([this, &shards, &workerLoss, &masterModel, &testSet, w]() {
                Sequential localModel = masterModel.clone();
                for (int epoch = 1; epoch <= epochs_; ++epoch) {
                    workerEpoch(w, shards[static_cast<size_t>(w)],
                                localModel, workerLoss[static_cast<size_t>(w)], epoch, testSet);
                }
            });
            workers.push_back(thr);
        }
        for (auto* thr : workers) { thr->join(); delete thr; }

        ps_.pullWeights(masterModel);
        LOG_INFO("Distributed training complete.");
    }

private:
    int numWorkers_, epochs_, batchSize_;
    double lr_;
    ParameterServer ps_;

    void registerLayers(Sequential& model) {
        for (int l = 0; l < model.numLayers(); ++l) {
            DenseLayer* dense = dynamic_cast<DenseLayer*>(model.getLayer(l));
            if (!dense) continue;
            ps_.registerLayer(l, dense->weights().rows(), dense->weights().cols());
        }
    }

    void workerEpoch(int workerId, const Dataset& shard,
                     Sequential& sharedModel, double& lossOut, int epoch, const Dataset& testSet)
    {
        std::mt19937 rng(static_cast<unsigned>(workerId * 1000 + epoch));
        double localLoss = 0.0;

        for (const Batch& batch : shard.batches(batchSize_, &rng)) {
            for (int k = 0; k < batch.size(); ++k) {
                std::vector<double> probs = sharedModel.forward(batch.X[static_cast<size_t>(k)]);
                localLoss += Sequential::crossEntropy(probs, batch.y[static_cast<size_t>(k)]);

                std::vector<double> dOut = probs;
                dOut[static_cast<size_t>(batch.y[static_cast<size_t>(k)])] -= 1.0;
                sharedModel.backward(dOut);
            }

            for (int l = 0; l < sharedModel.numLayers(); ++l) {
                DenseLayer* dense = dynamic_cast<DenseLayer*>(sharedModel.getLayer(l));
                if (!dense) continue;
                ps_.pushGradient(l, dense->weightGrad(), dense->biasGrad(), batch.size());
            }
        }

        ps_.workerEpochDone();
        ps_.pullWeights(sharedModel);

        lossOut = (shard.size() > 0) ? localLoss / shard.size() : 0.0;

        if (workerId == 0 && (epoch % 50 == 0 || epoch == 1)) {
            double acc = sharedModel.accuracy(testSet);
            LOG_INFO("[Distributed] Epoch " + std::to_string(epoch) +
                     " | Test Acc: " + std::to_string(acc).substr(0, 5) + "%");
        }
    }

    std::vector<Dataset> partitionData(const Dataset& ds, int n) {
        std::vector<Dataset> shards(static_cast<size_t>(n));
        for (size_t i = 0; i < shards.size(); ++i)
            shards[i].numFeatures = ds.numFeatures;

        for (int i = 0; i < ds.size(); ++i) {
            int w = i % n;
            shards[static_cast<size_t>(w)].X.push_back(ds.X[static_cast<size_t>(i)]);
            shards[static_cast<size_t>(w)].y.push_back(ds.y[static_cast<size_t>(i)]);
        }

        LOG_INFO("Data partitioned: " + std::to_string(ds.size()) +
                 " samples across " + std::to_string(n) +
                 " workers (~" + std::to_string(ds.size() / n) + " each)");
        return shards;
    }
};
