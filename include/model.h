/**
 * @file model.h
 * @brief Sequential Neural Network Model — OOP Design + File Systems
 *
 * TECHNIQUE: OOP Design
 *   - Sequential model owns a list of Layer pointers via std::unique_ptr
 *     (Memory Management: no raw new/delete in user code).
 *   - add() uses method chaining: model.add(...).add(...).add(...)
 *   - summary() prints the architecture like PyTorch/Keras do.
 *
 * TECHNIQUE: File Systems
 *   - saveBinary() writes a .bin model file with a magic header so we can
 *     detect corrupt files.
 *   - loadBinary() validates the magic header and throws ModelException on
 *     mismatch instead of reading garbage data.
 *   - Uses std::filesystem to ensure the save directory exists.
 *
 * TECHNIQUE: Memory Management
 *   - std::unique_ptr<Layer> ensures each layer is automatically freed
 *     when the model is destroyed — no memory leaks possible.
 */

#pragma once

#include <vector>
#include <memory>       // std::unique_ptr
#include <string>
#include <fstream>
#include <cmath>
#include <cassert>

#include "layer.h"
#include "exceptions.h"
#include "logger.h"
#include "fscompat.h"   // portable filesystem shim

// Magic bytes written at the start of every .bin file
// If these don't match on load, the file is corrupt/wrong version
constexpr uint32_t MODEL_MAGIC   = 0x4E4E4D4C;   // "NNML" in ASCII
constexpr uint32_t MODEL_VERSION = 2;

class Sequential {
public:
    // ── Layer registration ────────────────────────────────────────────
    // Takes ownership of the layer pointer — caller must use std::move or new
    // Returns *this for method chaining:  model.add(...).add(...)
    Sequential& add(std::unique_ptr<Layer> layer) {
        LOG_DEBUG("Adding layer: " + layer->name());
        layers_.push_back(std::move(layer));
        return *this;
    }

    // Convenience: construct-and-add in one line
    template<typename T, typename... Args>
    Sequential& add(Args&&... args) {
        return add(std::make_unique<T>(std::forward<Args>(args)...));
    }

    // Deep copy clone
    Sequential clone() const {
        Sequential copy;
        for (const auto& layer : layers_)
            copy.add(layer->clone());
        return copy;
    }

    // ── Forward pass through all layers ──────────────────────────────
    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> x = input;
        for (auto& layer : layers_)
            x = layer->forward(x);
        return x;   // final output (probabilities after softmax)
    }

    // ── Backward pass — returns gradient w.r.t. model input ──────────
    // dLoss: gradient of loss w.r.t. the last layer's output
    std::vector<double> backward(const std::vector<double>& dLoss) {
        std::vector<double> grad = dLoss;
        // Traverse layers in REVERSE order — that's how backprop works
        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i)
            grad = layers_[i]->backward(grad);
        return grad;
    }

    // ── Apply accumulated gradients across all layers ─────────────────
    void applyUpdate(double lr, int batchSize) {
        for (auto& layer : layers_)
            layer->applyUpdate(lr, batchSize);
    }

    // ── Compute accuracy on a dataset ────────────────────────────────
    double accuracy(const Dataset& ds) {
        int correct = 0;
        for (int i = 0; i < ds.size(); ++i) {
            auto probs = forward(ds.X[i]);
            int pred   = static_cast<int>(
                std::max_element(probs.begin(), probs.end()) - probs.begin());
            if (pred == ds.y[i]) ++correct;
        }
        return 100.0 * correct / ds.size();
    }

    // ── Cross-entropy loss on one sample ─────────────────────────────
    static double crossEntropy(const std::vector<double>& probs, int label) {
        if (label < 0 || label >= (int)probs.size())
            throw DataException("Label " + std::to_string(label) +
                                " out of range [0," +
                                std::to_string(probs.size()) + ")");
        return -std::log(std::max(probs[label], 1e-12));
    }

    // ── Architecture summary (like Keras model.summary()) ─────────────
    void summary() const {
        LOG_INFO("=== Model Summary ===");
        int total = 0;
        for (const auto& l : layers_) {
            int p = l->paramCount();
            total += p;
            LOG_INFO("  " + l->name() + "   params: " + std::to_string(p));
        }
        LOG_INFO("  Total trainable parameters: " + std::to_string(total));
        LOG_INFO("=====================");
    }

    int numLayers() const { return static_cast<int>(layers_.size()); }

    // Direct layer access (needed by Trainer for parameter server)
    Layer* getLayer(int idx) {
        if (idx < 0 || idx >= numLayers())
            throw ModelException("Layer index " + std::to_string(idx) + " out of range");
        return layers_[idx].get();
    }

    // ── Binary save to file ───────────────────────────────────────────
    void saveBinary(const std::string& pathStr) const {
        // TECHNIQUE: File Systems — create parent directories if needed
        auto sep = pathStr.find_last_of("/\\");
        if (sep != std::string::npos)
            fs::create_directories(pathStr.substr(0, sep));

        std::ofstream f(pathStr, std::ios::binary);
        if (!f.is_open())
            throw FileException(pathStr, "cannot open for writing");

        // Write magic header so loadBinary can detect corrupt files
        f.write(reinterpret_cast<const char*>(&MODEL_MAGIC),   sizeof(MODEL_MAGIC));
        f.write(reinterpret_cast<const char*>(&MODEL_VERSION), sizeof(MODEL_VERSION));

        uint32_t numLayers = static_cast<uint32_t>(layers_.size());
        f.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));

        for (const auto& layer : layers_)
            layer->saveBinary(f);

        LOG_INFO("Model saved to: " + pathStr);
    }

    // ── Binary load from file ─────────────────────────────────────────
    void loadBinary(const std::string& pathStr) {
        if (!fs::exists(pathStr))
            throw FileException(pathStr, "model file does not exist");

        std::ifstream f(pathStr, std::ios::binary);
        if (!f.is_open())
            throw FileException(pathStr, "cannot open for reading");

        uint32_t magic, version;
        f.read(reinterpret_cast<char*>(&magic),   sizeof(magic));
        f.read(reinterpret_cast<char*>(&version), sizeof(version));

        // TECHNIQUE: Error Recovery — validate before reading weights
        if (magic != MODEL_MAGIC)
            throw ModelException("Invalid model file (bad magic bytes): " + pathStr);
        if (version != MODEL_VERSION)
            throw ModelException("Model version mismatch: file has v" +
                                 std::to_string(version) + " but code expects v" +
                                 std::to_string(MODEL_VERSION));

        uint32_t numLayers;
        f.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));

        if (numLayers != static_cast<uint32_t>(layers_.size()))
            throw ModelException("Layer count mismatch: file has " +
                                 std::to_string(numLayers) + " but model has " +
                                 std::to_string(layers_.size()));

        for (auto& layer : layers_)
            layer->loadBinary(f);

        LOG_INFO("Model loaded from: " + pathStr);
    }

private:
    // std::unique_ptr enforces single ownership — no memory leaks
    std::vector<std::unique_ptr<Layer>> layers_;
};
