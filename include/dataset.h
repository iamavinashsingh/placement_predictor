/**
 * @file dataset.h
 * @brief Dataset with Custom Batch Iterator — STL Advanced Usage + File Systems
 *
 * TECHNIQUE: STL Advanced Usage
 * We implement a fully standards-compliant C++ iterator so range-based for
 * loops work naturally over mini-batches:
 *
 *   for (const Batch& b : dataset.batches(32)) {
 *       // b.X  — 2-D block of inputs
 *       // b.y  — corresponding labels
 *   }
 *
 * TECHNIQUE: File Systems
 * Uses <filesystem> (C++17) to:
 *   - Check whether a file exists before trying to open it
 *   - Report the file size in human-readable form
 *   - Scan a directory for CSV shards (for distributed loading)
 *
 * TECHNIQUE: Error Recovery
 * Every failure throws a typed exception instead of hard-crashing with exit().
 */

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <iterator>     // std::forward_iterator_tag

#include "exceptions.h"
#include "logger.h"
#include "fscompat.h"   // portable <filesystem> shim — works on GCC 6+

// ─────────────────────────────────────────────────────────────────────────────
//  A single mini-batch of data
// ─────────────────────────────────────────────────────────────────────────────
struct Batch {
    std::vector<std::vector<double>> X;   // [batch_size][features]
    std::vector<int>                 y;   // [batch_size]
    int size() const { return static_cast<int>(y.size()); }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Full Dataset
// ─────────────────────────────────────────────────────────────────────────────
class Dataset {
public:
    std::vector<std::vector<double>> X;   // all samples
    std::vector<int>                 y;   // all labels
    int numFeatures = 0;

    int size()     const { return static_cast<int>(y.size()); }
    bool empty()   const { return y.empty(); }

    // ── Load from CSV using portable fs shim ─────────────────────────
    static Dataset fromCSV(const std::string& pathStr, int expectedFeatures) {
        // TECHNIQUE: File Systems — existence check before opening
        if (!fs::exists(pathStr))
            throw FileException(pathStr, "file does not exist");

        if (!fs::is_regular_file(pathStr))
            throw FileException(pathStr, "path is not a regular file");

        auto sizeBytes = fs::file_size(pathStr);
        LOG_INFO("Loading CSV: " + pathStr +
                 " (" + std::to_string(sizeBytes) + " bytes)");

        std::ifstream file(pathStr);
        if (!file.is_open())
            throw FileException(pathStr, "permission denied or locked");

        Dataset ds;
        ds.numFeatures = expectedFeatures;

        std::string line;
        int lineNum = 0;
        int skipped = 0;

        while (std::getline(file, line)) {
            ++lineNum;
            if (line.empty()) continue;

            // Skip header rows that don't start with a digit or minus
            if (lineNum == 1 &&
                !std::isdigit(static_cast<unsigned char>(line[0])) &&
                line[0] != '-' && line[0] != '.') {
                LOG_DEBUG("Skipping header: " + line);
                continue;
            }

            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row;

            while (std::getline(ss, cell, ',')) {
                if (cell.empty()) continue;
                try {
                    row.push_back(std::stod(cell));
                } catch (const std::invalid_argument&) {
                    // Typed exception — Error Recovery
                    throw DataException("Non-numeric value '" + cell +
                                       "' at line " + std::to_string(lineNum));
                }
            }

            // Each row must have exactly features+1 columns (last = label)
            int expected = expectedFeatures + 1;
            if ((int)row.size() != expected) {
                ++skipped;
                LOG_WARN("Line " + std::to_string(lineNum) +
                         ": expected " + std::to_string(expected) +
                         " cols, got " + std::to_string(row.size()) + " — skipped");
                continue;
            }

            ds.X.push_back(std::vector<double>(row.begin(), row.begin() + expectedFeatures));
            ds.y.push_back(static_cast<int>(row.back()));
        }

        if (ds.empty())
            throw DataException("CSV produced zero valid rows: " + pathStr);

        LOG_INFO("Loaded " + std::to_string(ds.size()) + " samples, " +
                 std::to_string(skipped) + " rows skipped");
        return ds;
    }

    // ── Scan a directory for CSV shards (Distributed Systems support) ─
    static std::vector<std::string> findCSVShards(const std::string& dirStr) {
        if (!fs::is_directory(dirStr))
            throw FileException(dirStr, "not a directory");
        // Uses portable fscompat::list_csv_files (sys/stat + WinAPI)
        return fs::list_csv_files(dirStr);
    }

    // ── Normalisation (fit mean/std on training set) ──────────────────
    struct Scaler {
        std::vector<double> mean, stdv;
    };

    Scaler fitScaler() const {
        int N = size(), F = numFeatures;
        Scaler s;
        s.mean.assign(F, 0.0);
        s.stdv.assign(F, 0.0);

        for (const auto& row : X)
            for (int j = 0; j < F; ++j) s.mean[j] += row[j];
        for (int j = 0; j < F; ++j) s.mean[j] /= N;

        for (const auto& row : X)
            for (int j = 0; j < F; ++j) {
                double d = row[j] - s.mean[j];
                s.stdv[j] += d * d;
            }
        for (int j = 0; j < F; ++j) {
            s.stdv[j] = std::sqrt(s.stdv[j] / N);
            if (s.stdv[j] < 1e-8) s.stdv[j] = 1.0;   // avoid divide-by-zero
        }
        return s;
    }

    void applyScaler(const Scaler& s) {
        for (auto& row : X)
            for (int j = 0; j < numFeatures; ++j)
                row[j] = (row[j] - s.mean[j]) / s.stdv[j];
    }

    // ── Train / Test split ────────────────────────────────────────────
    std::pair<Dataset, Dataset> split(double trainFrac, std::mt19937& rng) const {
        int N = size();
        std::vector<int> idx(N);
        std::iota(idx.begin(), idx.end(), 0);            // 0, 1, 2, ...
        std::shuffle(idx.begin(), idx.end(), rng);       // randomise order

        int trainN = static_cast<int>(N * trainFrac);
        Dataset tr, te;
        tr.numFeatures = te.numFeatures = numFeatures;

        for (int i = 0; i < N; ++i) {
            auto& dst = (i < trainN) ? tr : te;
            dst.X.push_back(X[idx[i]]);
            dst.y.push_back(y[idx[i]]);
        }
        return { std::move(tr), std::move(te) };
    }

    // ─────────────────────────────────────────────────────────────────
    //  TECHNIQUE: STL Advanced Usage — Custom Batch Iterator
    //
    //  Allows:   for (const Batch& b : dataset.batches(32)) { ... }
    // ─────────────────────────────────────────────────────────────────

    // The view object returned by dataset.batches(batchSize)
    struct BatchRange {
        const Dataset* ds;
        int batchSize;
        std::vector<int> order;   // shuffled index order

        // ── Inner iterator ───────────────────────────────────────────
        struct Iterator {
            using iterator_category = std::forward_iterator_tag;
            using value_type        = Batch;
            using difference_type   = std::ptrdiff_t;

            const Dataset*         ds;
            const std::vector<int>* order;
            int                    batchSize;
            int                    pos;        // current position in order[]

            Iterator(const Dataset* d, const std::vector<int>* o,
                     int bs, int p)
                : ds(d), order(o), batchSize(bs), pos(p) {}

            // Dereference: build and return the current batch
            Batch operator*() const {
                Batch b;
                int end = std::min(pos + batchSize, (int)order->size());
                for (int i = pos; i < end; ++i) {
                    b.X.push_back(ds->X[(*order)[i]]);
                    b.y.push_back(ds->y[(*order)[i]]);
                }
                return b;
            }

            // Pre-increment: advance to the next batch start
            Iterator& operator++() { pos += batchSize; return *this; }

            bool operator!=(const Iterator& other) const {
                int thisPos = std::min(pos, (int)order->size());
                int otherPos = std::min(other.pos, (int)order->size());
                return thisPos != otherPos;
            }
        };

        Iterator begin() const {
            return { ds, &order, batchSize, 0 };
        }
        Iterator end() const {
            return { ds, &order, batchSize, (int)order.size() };
        }
    };

    // Call this to get an iterable range of batches (optionally shuffled)
    BatchRange batches(int batchSize, std::mt19937* rng = nullptr) const {
        BatchRange range;
        range.ds        = this;
        range.batchSize = batchSize;
        range.order.resize(size());
        std::iota(range.order.begin(), range.order.end(), 0);
        if (rng) std::shuffle(range.order.begin(), range.order.end(), *rng);
        return range;
    }

    // ── Scaler serialise / deserialise ────────────────────────────────
    static void saveScaler(const Scaler& s, std::ofstream& f) {
        int sz = static_cast<int>(s.mean.size());
        f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        f.write(reinterpret_cast<const char*>(s.mean.data()), sz * sizeof(double));
        f.write(reinterpret_cast<const char*>(s.stdv.data()), sz * sizeof(double));
    }
    static Scaler loadScaler(std::ifstream& f) {
        int sz;
        f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        Scaler s;
        s.mean.resize(sz);
        s.stdv.resize(sz);
        f.read(reinterpret_cast<char*>(s.mean.data()), sz * sizeof(double));
        f.read(reinterpret_cast<char*>(s.stdv.data()), sz * sizeof(double));
        return s;
    }
};
