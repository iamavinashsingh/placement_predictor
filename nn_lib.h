#ifndef NN_LIB_H
#define NN_LIB_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>

using namespace std;

// Constants
const int    INPUT_SIZE   = 5;
const int    HIDDEN_SIZE  = 16;
const int    OUTPUT_SIZE  = 10;

// Data Structures
struct Dataset {
    vector<vector<double>> X;
    vector<int>            y;
};

struct Scaler {
    vector<double> mean, stdv;
};

struct NeuralNet {
    vector<vector<double>> W1;
    vector<double>         b1;
    vector<vector<double>> W2;
    vector<double>         b2;

    // Cache for forward/backward
    vector<double> z1, a1;
    vector<double> z2, a2;

    NeuralNet() {
        W1.assign(HIDDEN_SIZE, vector<double>(INPUT_SIZE));
        b1.assign(HIDDEN_SIZE, 0.0);
        W2.assign(OUTPUT_SIZE, vector<double>(HIDDEN_SIZE));
        b2.assign(OUTPUT_SIZE, 0.0);
    }
};

// Functions
inline double randomWeight(int fanIn, mt19937& rng) {
    normal_distribution<double> dist(0.0, sqrt(2.0 / fanIn));
    return dist(rng);
}

inline Dataset loadCSV(const string& path) {
    Dataset data;
    ifstream file(path);
    if (!file.is_open()) {
        cerr << "ERROR: Cannot open file " << path << endl;
        exit(1);
    }

    string line;
    bool firstLine = true;
    while (getline(file, line)) {
        if (line.empty()) continue;
        if (firstLine) {
            firstLine = false;
            if (!isdigit(line[0]) && line[0] != '-' && line[0] != '.') continue;
        }

        stringstream ss(line);
        string cell;
        vector<double> row;
        while (getline(ss, cell, ',')) {
            try { row.push_back(stod(cell)); } catch(...) {}
        }

        if (row.size() != INPUT_SIZE + 1) continue;

        data.X.push_back(vector<double>(row.begin(), row.begin() + INPUT_SIZE));
        data.y.push_back((int)row.back());
    }
    return data;
}

inline Scaler fitScaler(const vector<vector<double>>& X) {
    Scaler s;
    s.mean.assign(INPUT_SIZE, 0.0);
    s.stdv.assign(INPUT_SIZE, 0.0);
    int N = X.size();
    if (N == 0) return s;

    for (const auto& row : X)
        for (int j = 0; j < INPUT_SIZE; j++) s.mean[j] += row[j];
    for (int j = 0; j < INPUT_SIZE; j++) s.mean[j] /= N;

    for (const auto& row : X)
        for (int j = 0; j < INPUT_SIZE; j++)
            s.stdv[j] += (row[j] - s.mean[j]) * (row[j] - s.mean[j]);
    for (int j = 0; j < INPUT_SIZE; j++) {
        s.stdv[j] = sqrt(s.stdv[j] / N);
        if (s.stdv[j] < 1e-8) s.stdv[j] = 1.0;
    }
    return s;
}

inline void applyScaler(vector<vector<double>>& X, const Scaler& s) {
    for (auto& row : X)
        for (int j = 0; j < INPUT_SIZE; j++)
            row[j] = (row[j] - s.mean[j]) / s.stdv[j];
}

#endif
