/**
 * @file layer.h
 * @brief Abstract Layer Hierarchy — OOP Design
 *
 * TECHNIQUE: OOP Design
 *
 * Layers form an abstract hierarchy:
 *
 *   Layer  (pure abstract interface)
 *   ├── DenseLayer      — learnable weights W, b; forward = W·x + b
 *   ├── ReLULayer       — element-wise max(0, x); no learnable params
 *   └── SoftmaxLayer    — numerically-stable softmax; no learnable params
 *
 * WHY THIS MATTERS FOR MNCs:
 *   - Open/Closed Principle: adding a new activation (e.g. GELU) means
 *     creating ONE new class — zero changes to existing code.
 *   - Polymorphism: the Sequential model just calls layer->forward(x) without
 *     knowing which concrete type it is.
 *   - Encapsulation: weights are PRIVATE; the only way to touch them is
 *     through addGradient() and applyUpdate(). No stray code can corrupt them.
 */

#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>    // std::max_element
#include <numeric>      // std::accumulate
#include <random>
#include <memory>       // std::unique_ptr — Memory Management

#include "matrix.h"
#include "exceptions.h"
#include "logger.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Abstract interface — every layer MUST implement these three methods
// ─────────────────────────────────────────────────────────────────────────────
class Layer {
public:
    // Virtual destructor is MANDATORY when deleting through a base pointer
    virtual ~Layer() = default;

    // Forward pass: given input x, return output activations
    // const& avoids copying; pure virtual (= 0) forces subclasses to implement
    virtual std::vector<double> forward(const std::vector<double>& x) = 0;

    // Backward pass: given upstream gradient dOut, return gradient w.r.t. input
    // Also accumulates weight gradients internally
    virtual std::vector<double> backward(const std::vector<double>& dOut) = 0;

    // Update weights using accumulated gradients, then reset grad accumulators
    virtual void applyUpdate(double learningRate, int batchSize) = 0;

    // Human-readable description for logging
    virtual std::string name() const = 0;

    // Save/load support for binary model files
    virtual void saveBinary (std::ofstream& f) const = 0;
    virtual void loadBinary (std::ifstream& f)       = 0;

    // Number of trainable parameters (used by model summary)
    virtual int paramCount() const { return 0; }

    // Clone method for deep copying models
    virtual std::unique_ptr<Layer> clone() const = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
//  DenseLayer  (Fully-Connected)
//  y = W · x + b
//  Backprop:
//    dW  += dOut ⊗ x   (outer product)
//    db  += dOut
//    dIn  = Wᵀ · dOut
// ─────────────────────────────────────────────────────────────────────────────
class DenseLayer : public Layer {
public:
    // Constructor
    DenseLayer(int inSize, int outSize, std::mt19937& rng)
        : inSize_(inSize), outSize_(outSize),
          W_(outSize, inSize), b_(outSize, 0.0),
          dW_(outSize, inSize), db_(outSize, 0.0),
          input_(inSize, 0.0)
    {
        // He Kaiming initialisation — reduces vanishing/exploding gradients
        W_.heInit(inSize, rng);
    }

    // ── Forward ───────────────────────────────────────────────────────
    std::vector<double> forward(const std::vector<double>& x) override {
        if ((int)x.size() != inSize_)
            throw DataException(name() + " forward: input size mismatch, expected "
                                + std::to_string(inSize_) + " got "
                                + std::to_string(x.size()));
        input_ = x;                     // cache input for backward pass
        auto out = W_.matvec(x);        // W · x  (uses std::inner_product)
        for (int i = 0; i < outSize_; ++i)
            out[i] += b_[i];            // add bias
        return out;
    }

    // ── Backward ──────────────────────────────────────────────────────
    std::vector<double> backward(const std::vector<double>& dOut) override {
        // Accumulate weight gradients (will be averaged over batch in applyUpdate)
        for (int i = 0; i < outSize_; ++i) {
            for (int j = 0; j < inSize_; ++j)
                dW_(i, j) += dOut[i] * input_[j];   // outer product
            db_[i] += dOut[i];
        }

        // Compute gradient w.r.t. input: dIn = Wᵀ · dOut
        std::vector<double> dIn(inSize_, 0.0);
        for (int j = 0; j < inSize_; ++j)
            for (int i = 0; i < outSize_; ++i)
                dIn[j] += W_(i, j) * dOut[i];

        return dIn;
    }

    // ── Apply accumulated gradients, then reset ───────────────────────
    void applyUpdate(double lr, int batchSize) override {
        double scale = lr / batchSize;
        for (int i = 0; i < outSize_; ++i) {
            for (int j = 0; j < inSize_; ++j) {
                W_(i, j) -= scale * dW_(i, j);
                dW_(i, j) = 0.0;   // reset
            }
            b_[i] -= scale * db_[i];
            db_[i] = 0.0;          // reset
        }
    }

    std::string name() const override {
        return "DenseLayer(" + std::to_string(inSize_) +
               " -> " + std::to_string(outSize_) + ")";
    }
    int paramCount() const override { return outSize_ * inSize_ + outSize_; }

    // ── Binary serialisation ──────────────────────────────────────────
    void saveBinary(std::ofstream& f) const override {
        f.write(reinterpret_cast<const char*>(&inSize_),  sizeof(inSize_));
        f.write(reinterpret_cast<const char*>(&outSize_), sizeof(outSize_));
        W_.writeBinary(f);
        f.write(reinterpret_cast<const char*>(b_.data()), outSize_ * sizeof(double));
    }
    void loadBinary(std::ifstream& f) override {
        f.read(reinterpret_cast<char*>(&inSize_),  sizeof(inSize_));
        f.read(reinterpret_cast<char*>(&outSize_), sizeof(outSize_));
        W_ = Matrix::readBinary(f);
        b_.resize(outSize_);
        f.read(reinterpret_cast<char*>(b_.data()), outSize_ * sizeof(double));
        // Re-size gradient buffers to match
        dW_ = Matrix(outSize_, inSize_);
        db_.assign(outSize_, 0.0);
        input_.assign(inSize_, 0.0);
    }

    // Give the Trainer access to weight copies for parameter server
    Matrix& weights()  { return W_; }
    Matrix& weightGrad() { return dW_; }
    std::vector<double>& biases()   { return b_;  }
    std::vector<double>& biasGrad() { return db_; }

    std::unique_ptr<Layer> clone() const override {
        return std::make_unique<DenseLayer>(*this);
    }

private:
    int inSize_, outSize_;
    Matrix W_;                      // weight matrix (contiguous memory)
    std::vector<double> b_;         // bias vector

    Matrix dW_;                     // gradient accumulators
    std::vector<double> db_;

    std::vector<double> input_;     // cached input for backward
};

// ─────────────────────────────────────────────────────────────────────────────
//  ReLULayer  — element-wise Rectified Linear Unit: f(x) = max(0, x)
// ─────────────────────────────────────────────────────────────────────────────
class ReLULayer : public Layer {
public:
    std::vector<double> forward(const std::vector<double>& x) override {
        mask_ = x;   // cache which elements were positive
        std::vector<double> out(x.size());
        std::transform(x.begin(), x.end(), out.begin(),
                       [](double v){ return std::max(0.0, v); });
        return out;
    }

    std::vector<double> backward(const std::vector<double>& dOut) override {
        std::vector<double> dIn(dOut.size());
        // Gradient is 0 where the input was ≤ 0, else passes through unchanged
        for (size_t i = 0; i < dOut.size(); ++i)
            dIn[i] = (mask_[i] > 0) ? dOut[i] : 0.0;
        return dIn;
    }

    void applyUpdate(double, int) override {}   // no learnable params
    std::string name() const override { return "ReLULayer"; }
    void saveBinary(std::ofstream&) const override {}
    void loadBinary(std::ifstream&) override {}

    std::unique_ptr<Layer> clone() const override {
        return std::make_unique<ReLULayer>(*this);
    }

private:
    std::vector<double> mask_;   // stores input for backward pass
};

// ─────────────────────────────────────────────────────────────────────────────
//  SoftmaxLayer  — numerically-stable softmax + cross-entropy backward fused
//
//  Combined Softmax + CE backward gives dz = p - one_hot(label),
//  which is simpler and more numerically stable than doing them separately.
// ─────────────────────────────────────────────────────────────────────────────
class SoftmaxLayer : public Layer {
public:
    std::vector<double> forward(const std::vector<double>& x) override {
        // Subtract max for numerical stability (prevents exp overflow)
        double maxVal = *std::max_element(x.begin(), x.end());
        std::vector<double> out(x.size());
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            out[i] = std::exp(x[i] - maxVal);
            sum += out[i];
        }
        for (double& v : out) v /= sum;
        probs_ = out;    // cache for backward
        return out;
    }

    // For Softmax + Cross-Entropy combined backward, caller passes dOut as:
    //   dOut[label] -= 1.0  (i.e. probs - one_hot(label))
    // We just pass it through since Jacobian-vector product simplifies.
    std::vector<double> backward(const std::vector<double>& dOut) override {
        return dOut;   // gradient already computed by Trainer (p - y_one_hot)
    }

    void applyUpdate(double, int) override {}
    std::string name() const override { return "SoftmaxLayer"; }
    void saveBinary(std::ofstream&) const override {}
    void loadBinary(std::ifstream&) override {}

    const std::vector<double>& probs() const { return probs_; }

    std::unique_ptr<Layer> clone() const override {
        return std::make_unique<SoftmaxLayer>(*this);
    }

private:
    std::vector<double> probs_;
};
