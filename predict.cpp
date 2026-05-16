#include "nn_lib.h"

struct Model {
    int inputSize, hiddenSize, outputSize;
    Scaler s;
    NeuralNet net;
};

Model loadModel(const string& path) {
    ifstream f(path);
    if (!f.is_open()) {
        cerr << "ERROR: Cannot open model file: " << path << "\n";
        exit(1);
    }
    Model m;
    f >> m.inputSize >> m.hiddenSize >> m.outputSize;
    m.s.mean.resize(m.inputSize);
    m.s.stdv.resize(m.inputSize);
    for (int i = 0; i < m.inputSize; i++) f >> m.s.mean[i];
    for (int i = 0; i < m.inputSize; i++) f >> m.s.stdv[i];

    m.net.W1.assign(m.hiddenSize, vector<double>(m.inputSize));
    for (int i = 0; i < m.hiddenSize; i++)
        for (int j = 0; j < m.inputSize; j++) f >> m.net.W1[i][j];

    m.net.b1.resize(m.hiddenSize);
    for (int i = 0; i < m.hiddenSize; i++) f >> m.net.b1[i];

    m.net.W2.assign(m.outputSize, vector<double>(m.hiddenSize));
    for (int i = 0; i < m.outputSize; i++)
        for (int j = 0; j < m.hiddenSize; j++) f >> m.net.W2[i][j];

    m.net.b2.resize(m.outputSize);
    for (int i = 0; i < m.outputSize; i++) f >> m.net.b2[i];
    return m;
}

vector<double> predict(Model& m, vector<double> x) {
    for (int j = 0; j < m.inputSize; j++)
        x[j] = (x[j] - m.s.mean[j]) / m.s.stdv[j];

    vector<double> a1(m.hiddenSize);
    for (int i = 0; i < m.hiddenSize; i++) {
        double sum = m.net.b1[i];
        for (int j = 0; j < m.inputSize; j++) sum += m.net.W1[i][j] * x[j];
        a1[i] = max(0.0, sum);
    }

    vector<double> z2(m.outputSize);
    for (int i = 0; i < m.outputSize; i++) {
        double sum = m.net.b2[i];
        for (int j = 0; j < m.hiddenSize; j++) sum += m.net.W2[i][j] * a1[j];
        z2[i] = sum;
    }

    double maxL = *max_element(z2.begin(), z2.end());
    vector<double> probs(m.outputSize);
    double sum = 0.0;
    for (int i = 0; i < m.outputSize; i++) {
        probs[i] = exp(z2[i] - maxL);
        sum += probs[i];
    }
    for (int i = 0; i < m.outputSize; i++) probs[i] /= sum;
    return probs;
}

int main(int argc, char** argv) {
    string modelPath = (argc > 1) ? argv[1] : "model.txt";
    Model m = loadModel(modelPath);

    cout << "Placement Predictor loaded (" << m.inputSize << " inputs)\n";
    while (true) {
        cout << "Enter (DSA, Projects, IQ, CGPA, Attendance) or 'q': ";
        string s; cin >> s; if (s == "q") break;
        vector<double> in(5);
        try {
            in[0] = stod(s);
            cin >> in[1] >> in[2] >> in[3] >> in[4];
        } catch(...) { cout << "Invalid input\n"; continue; }

        auto p = predict(m, in);
        int best = max_element(p.begin(), p.end()) - p.begin();
        cout << "Prediction: Company " << best << " (Confidence: " << p[best]*100 << "%)\n";
    }
    return 0;
}
