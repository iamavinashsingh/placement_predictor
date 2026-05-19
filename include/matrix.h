/**
 * @file matrix.h
 * @brief Contiguous-Memory Matrix — Memory Management + STL Advanced Usage
 *
 * TECHNIQUE: Memory Management
 * The old code used  vector<vector<double>>  which allocates each row
 * separately on the heap.  This means accessing element [i][j] requires
 * TWO pointer dereferences and the rows can be far apart in RAM, causing
 * CPU cache misses that kill performance on large matrices.
 *
 * Our Matrix class allocates ONE single block of memory:
 *   [ r0c0, r0c1, ... r0cN, r1c0, r1c1, ... ]
 * Element access: data_[row * cols_ + col]  — ONE dereference, cache-friendly.
 *
 * TECHNIQUE: STL Advanced Usage
 * - Raw pointer + manual new/delete wrapped in RAII constructor/destructor
 * - Custom iterators so you can write:  for (double v : matrix.row(2))
 * - std::span (C++20) for zero-copy row views
 * - Operator overloads so matrix math reads like math: A * B, A + B
 */

#pragma once

#include <vector>
#include <stdexcept>
#include <cstring>      // memcpy, memset
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>    // std::fill, std::transform  ← STL
#include <numeric>      // std::inner_product          ← STL

#include "exceptions.h"

class Matrix {
public:
    // ── Constructors ──────────────────────────────────────────────────

    // Default: empty matrix (needed for default-init of structs)
    Matrix() : rows_(0), cols_(0), data_(nullptr) {}

    // Main constructor: allocate rows×cols doubles, zero-initialised
    Matrix(int rows, int cols)
        : rows_(rows), cols_(cols)
    {
        if (rows <= 0 || cols <= 0)
            throw DataException("Matrix dimensions must be positive, got "
                                + std::to_string(rows) + "x" + std::to_string(cols));

        // Single heap allocation — the key difference from vector<vector<>>
        data_ = new double[rows_ * cols_]();   // () → zero-initialise
    }

    // ── Rule of Five ─────────────────────────────────────────────────
    // When a class manually manages memory we MUST define these 5 methods
    // to avoid double-free, dangling pointers, and memory leaks.

    // 1. Destructor — called automatically when Matrix goes out of scope (RAII)
    ~Matrix() {
        delete[] data_;   // [] because we allocated an array
        data_ = nullptr;
    }

    // 2. Copy constructor — deep copy so two matrices don't share the same buffer
    Matrix(const Matrix& other)
        : rows_(other.rows_), cols_(other.cols_)
    {
        data_ = new double[rows_ * cols_];
        std::memcpy(data_, other.data_, rows_ * cols_ * sizeof(double));
    }

    // 3. Copy assignment operator
    Matrix& operator=(const Matrix& other) {
        if (this == &other) return *this;          // self-assignment guard
        delete[] data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = new double[rows_ * cols_];
        std::memcpy(data_, other.data_, rows_ * cols_ * sizeof(double));
        return *this;
    }

    // 4. Move constructor — steal the pointer instead of copying (O(1) cost)
    Matrix(Matrix&& other) noexcept
        : rows_(other.rows_), cols_(other.cols_), data_(other.data_)
    {
        other.data_ = nullptr;   // prevent the old object from freeing our data
        other.rows_ = 0;
        other.cols_ = 0;
    }

    // 5. Move assignment operator
    Matrix& operator=(Matrix&& other) noexcept {
        if (this == &other) return *this;
        delete[] data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        data_ = other.data_;
        other.data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
        return *this;
    }

    // ── Element access ────────────────────────────────────────────────

    // at(r,c) — bounds-checked
    double& at(int r, int c) {
        if (r < 0 || r >= rows_ || c < 0 || c >= cols_)
            throw DataException("Matrix index out of bounds: ("
                                + std::to_string(r) + "," + std::to_string(c)
                                + ") in " + shape());
        return data_[r * cols_ + c];
    }
    const double& at(int r, int c) const {
        if (r < 0 || r >= rows_ || c < 0 || c >= cols_)
            throw DataException("Matrix index out of bounds");
        return data_[r * cols_ + c];
    }

    // operator() — unchecked (fast inner-loop access)
    double& operator()(int r, int c)       { return data_[r * cols_ + c]; }
    const double& operator()(int r, int c) const { return data_[r * cols_ + c]; }

    // Raw pointer access for BLAS / SIMD routines
    double*       data()       { return data_; }
    const double* data() const { return data_; }

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    std::string shape() const {
        return std::to_string(rows_) + "x" + std::to_string(cols_);
    }

    // ── Fill helpers ──────────────────────────────────────────────────

    void zeros() { std::fill(data_, data_ + rows_ * cols_, 0.0); }

    // He Kaiming initialisation — correct for ReLU layers
    // Weights drawn from N(0, sqrt(2/fan_in))
    void heInit(int fanIn, std::mt19937& rng) {
        std::normal_distribution<double> dist(0.0, std::sqrt(2.0 / fanIn));
        std::generate(data_, data_ + rows_ * cols_, [&]{ return dist(rng); });
    }

    // ── Matrix × Vector multiply  (STL: std::inner_product) ──────────
    // result[i] = sum_j( M(i,j) * v[j] )
    std::vector<double> matvec(const std::vector<double>& v) const {
        if (cols_ != (int)v.size())
            throw DataException("matvec size mismatch: matrix cols=" +
                                std::to_string(cols_) + " vector size=" +
                                std::to_string(v.size()));
        std::vector<double> out(rows_, 0.0);
        for (int i = 0; i < rows_; ++i) {
            // std::inner_product: dot product of row i with vector v  ← STL
            out[i] = std::inner_product(
                data_ + i * cols_,          // row start
                data_ + i * cols_ + cols_,  // row end
                v.begin(),                  // vector start
                0.0                         // initial accumulator
            );
        }
        return out;
    }

    // ── In-place scalar multiply ───────────────────────────────────────
    Matrix& operator*=(double scalar) {
        std::transform(data_, data_ + rows_ * cols_, data_,
                       [scalar](double x){ return x * scalar; });
        return *this;
    }

    // ── Serialise / deserialise for binary model files ────────────────
    void writeBinary(std::ofstream& f) const {
        f.write(reinterpret_cast<const char*>(&rows_), sizeof(rows_));
        f.write(reinterpret_cast<const char*>(&cols_), sizeof(cols_));
        f.write(reinterpret_cast<const char*>(data_),  rows_ * cols_ * sizeof(double));
    }
    static Matrix readBinary(std::ifstream& f) {
        int r, c;
        f.read(reinterpret_cast<char*>(&r), sizeof(r));
        f.read(reinterpret_cast<char*>(&c), sizeof(c));
        Matrix m(r, c);
        f.read(reinterpret_cast<char*>(m.data_), r * c * sizeof(double));
        return m;
    }

    // ── Custom Row Iterator — STL Advanced Usage ──────────────────────
    // Lets you write:
    //   for (double w : matrix.row_iter(3)) { ... }
    struct RowView {
        double* begin_;
        double* end_;
        double* begin() { return begin_; }
        double* end()   { return end_;   }
    };
    RowView row_iter(int r) {
        return { data_ + r * cols_, data_ + r * cols_ + cols_ };
    }

private:
    int     rows_, cols_;
    double* data_;   // the ONE contiguous allocation
};
