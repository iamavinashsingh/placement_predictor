/**
 * @file predict.cpp
 * @brief Inference / Prediction Entry Point
 *
 *  Usage:
 *    predict.exe [model.bin] [scaler.bin]
 *
 *  Then enter space-separated feature values when prompted:
 *    DSA  Projects  IQ  CGPA  Attendance
 *    e.g.:  8 5 110 8.5 90
 *
 *  TECHNIQUES DEMONSTRATED:
 *   - OOP Design       → loads Sequential model, calls model.forward()
 *   - Memory Management→ unique_ptr, Matrix (via model internals)
 *   - File Systems     → validates model/scaler file existence via <filesystem>
 *   - Error Recovery   → typed exceptions with user-friendly messages
 *   - STL Advanced     → std::transform for feature normalisation
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>
#include <random>

#include "include/exceptions.h"
#include "include/logger.h"
#include "include/dataset.h"
#include "include/matrix.h"
#include "include/layer.h"
#include "include/model.h"

// Company names for label -> human-readable output
// Labels 0-9 correspond to companies in the dataset
const std::vector<std::string> COMPANY_NAMES = {
    "Company A (No Placement)",
    "Company B (Tier-3)",
    "Company C (Tier-3)",
    "Company D (Tier-2)",
    "Company E (Tier-2)",
    "Company F (Tier-1)",
    "Company G (Tier-1)",
    "Company H (FAANG)",
    "Company I (FAANG)",
    "Company J (FAANG Top)"
};

// ─────────────────────────────────────────────────────────────────────────────
//  Build the SAME architecture as train.cpp
//  (Architecture must match what was saved — Error Recovery will catch mismatch)
// ─────────────────────────────────────────────────────────────────────────────
std::unique_ptr<Sequential> buildModel() {
    std::mt19937 rng(42);   // weights will be overwritten by loadBinary
    auto model = std::make_unique<Sequential>();
    model->add(std::make_unique<DenseLayer>(5, 16, rng));
    model->add(std::make_unique<ReLULayer>());
    model->add(std::make_unique<DenseLayer>(16, 10, rng));
    model->add(std::make_unique<SoftmaxLayer>());
    return model;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Print a confidence bar for all classes
// ─────────────────────────────────────────────────────────────────────────────
void printProbabilities(const std::vector<double>& probs) {
    std::cout << "\n  Class probabilities:\n";
    for (int i = 0; i < (int)probs.size(); ++i) {
        int bar = static_cast<int>(probs[i] * 30);
        std::string label = (i < (int)COMPANY_NAMES.size())
                                ? COMPANY_NAMES[i] : "Class " + std::to_string(i);
        std::cout << "  [" << i << "] " << std::string(bar, '#')
                  << std::string(30 - bar, '.')
                  << "  " << (int)(probs[i] * 100) << "%"
                  << "  " << label << "\n";
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  main()
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    try {
        Logger::instance().setLevel(LogLevel::WARNING);  // quiet mode for inference

        std::string modelPath  = (argc > 1) ? argv[1] : "model.bin";
        std::string scalerPath = (argc > 2) ? argv[2] : "scaler.bin";

        // ── File Systems: check paths exist before attempting to open ──
        if (!fs::exists(modelPath))
            throw FileException(modelPath,
                "model file not found. Run train.exe first.");
        if (!fs::exists(scalerPath))
            throw FileException(scalerPath,
                "scaler file not found. Run train.exe first.");

        // ── OOP Design: build model skeleton, load binary weights ──────
        auto model = buildModel();
        model->loadBinary(modelPath);

        // ── Load scaler (File Systems + Error Recovery) ────────────────
        Dataset::Scaler scaler;
        {
            std::ifstream sf(scalerPath, std::ios::binary);
            if (!sf.is_open())
                throw FileException(scalerPath, "cannot open for reading");
            scaler = Dataset::loadScaler(sf);
        }

        std::cout << "\n╔══════════════════════════════════════════════╗\n";
        std::cout <<   "║     Placement Predictor — Inference Mode     ║\n";
        std::cout <<   "╚══════════════════════════════════════════════╝\n";
        std::cout << "  Model : " << modelPath << "\n";
        std::cout << "  Input : DSA  Projects  IQ  CGPA  Attendance\n";
        std::cout << "  Type 'q' to quit.\n\n";

        // ── Interactive inference loop ─────────────────────────────────
        while (true) {
            std::cout << "Enter features > ";
            std::string first;
            std::cin >> first;
            if (first == "q" || first == "Q") break;

            std::vector<double> features(5);
            try {
                features[0] = std::stod(first);
                for (int i = 1; i < 5; ++i) {
                    std::string tok;
                    std::cin >> tok;
                    features[i] = std::stod(tok);
                }
            } catch (const std::invalid_argument&) {
                std::cout << "  ✗ Invalid number — please enter 5 numeric values.\n";
                std::cin.clear();
                std::cin.ignore(1000, '\n');
                continue;
            }

            // ── STL Advanced: std::transform for normalisation ─────────
            std::transform(features.begin(), features.end(), features.begin(),
                [&scaler, i = 0](double v) mutable {
                    return (v - scaler.mean[i]) / scaler.stdv[i++];
                });

            // ── Forward pass ───────────────────────────────────────────
            auto probs = model->forward(features);

            int best = static_cast<int>(
                std::max_element(probs.begin(), probs.end()) - probs.begin());

            std::string company = (best < (int)COMPANY_NAMES.size())
                                        ? COMPANY_NAMES[best]
                                        : "Class " + std::to_string(best);

            std::cout << "\n  ✔ Prediction  : " << company << "\n";
            std::cout <<   "  ✔ Confidence  : " << (int)(probs[best] * 100) << "%\n";

            printProbabilities(probs);
            std::cout << "\n";
        }

        std::cout << "Goodbye!\n";

    } catch (const FileException& e) {
        std::cerr << "\n[FILE ERROR] " << e.what() << "\n";
        return 1;
    } catch (const ModelException& e) {
        std::cerr << "\n[MODEL ERROR] " << e.what() << "\n";
        std::cerr << "  → Retrain with train.exe to regenerate a compatible model.\n";
        return 2;
    } catch (const NNException& e) {
        std::cerr << "\n[NN ERROR] " << e.what() << "\n";
        return 3;
    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL] " << e.what() << "\n";
        return 99;
    }

    return 0;
}
