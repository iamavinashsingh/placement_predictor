/**
 * @file exceptions.h
 * @brief Custom Exception Hierarchy — Error Recovery
 *
 * TECHNIQUE: Error Recovery
 * Instead of calling exit(1) or using generic std::exception, we define a
 * proper hierarchy of typed exceptions. This lets callers catch SPECIFIC
 * errors and potentially recover from them instead of crashing hard.
 *
 *   BaseException
 *   ├── FileException       → file not found, permission denied
 *   ├── DataException       → bad CSV rows, wrong column count
 *   ├── ModelException      → corrupt model file, shape mismatch
 *   ├── TrainingException   → NaN loss, divergence
 *   └── NetworkException    → socket/distributed comms errors
 */

#pragma once  // header guard — prevents double-inclusion

#include <stdexcept>
#include <string>

// ─────────────────────────────────────────────
//  Base — every project exception inherits this
// ─────────────────────────────────────────────
class NNException : public std::runtime_error {
public:
    // std::runtime_error already stores the message; we just forward it
    explicit NNException(const std::string& msg)
        : std::runtime_error("[NNException] " + msg) {}
};

// ─────────────────────────────────────────────
//  File-system related errors
// ─────────────────────────────────────────────
class FileException : public NNException {
public:
    explicit FileException(const std::string& path, const std::string& reason)
        : NNException("File error on '" + path + "': " + reason) {}
};

// ─────────────────────────────────────────────
//  Data / CSV parsing errors
// ─────────────────────────────────────────────
class DataException : public NNException {
public:
    explicit DataException(const std::string& detail)
        : NNException("Data error — " + detail) {}
};

// ─────────────────────────────────────────────
//  Model save / load errors
// ─────────────────────────────────────────────
class ModelException : public NNException {
public:
    explicit ModelException(const std::string& detail)
        : NNException("Model error — " + detail) {}
};

// ─────────────────────────────────────────────
//  Training divergence / NaN errors
// ─────────────────────────────────────────────
class TrainingException : public NNException {
public:
    explicit TrainingException(const std::string& detail)
        : NNException("Training error — " + detail) {}
};

// ─────────────────────────────────────────────
//  Distributed / network communication errors
// ─────────────────────────────────────────────
class NetworkException : public NNException {
public:
    explicit NetworkException(const std::string& detail)
        : NNException("Network error — " + detail) {}
};
