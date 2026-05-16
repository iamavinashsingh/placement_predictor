#include "nn_lib.h"

// Hyperparameters
const double LEARNING_RATE = 0.01;
const int    EPOCHS       = 500;
const int    BATCH_SIZE   = 32;
const double TRAIN_SPLIT  = 0.8;

mt19937 rng(42);

void initNet(NeuralNet& net) {
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net.W1[i][j] = randomWeight(INPUT_SIZE, rng);

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net.W2[i][j] = randomWeight(HIDDEN_SIZE, rng);
}

vector<double> forward(NeuralNet& net, const vector<double>& x) {
    net.z1.assign(HIDDEN_SIZE, 0.0);
    net.a1.assign(HIDDEN_SIZE, 0.0);
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        double sum = net.b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            sum += net.W1[i][j] * x[j];
        net.z1[i] = sum;
        net.a1[i] = max(0.0, sum);
    }

    net.z2.assign(OUTPUT_SIZE, 0.0);
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        double sum = net.b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            sum += net.W2[i][j] * net.a1[j];
        net.z2[i] = sum;
    }

    double maxLogit = *max_element(net.z2.begin(), net.z2.end());
    net.a2.assign(OUTPUT_SIZE, 0.0);
    double expSum = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        net.a2[i] = exp(net.z2[i] - maxLogit);
        expSum += net.a2[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) net.a2[i] /= expSum;
    return net.a2;
}

void backward(NeuralNet& net, const vector<double>& x, int trueLabel,
              vector<vector<double>>& dW1, vector<double>& db1,
              vector<vector<double>>& dW2, vector<double>& db2) {
    vector<double> dz2(OUTPUT_SIZE);
    for (int i = 0; i < OUTPUT_SIZE; i++)
        dz2[i] = net.a2[i] - (i == trueLabel ? 1.0 : 0.0);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++)
            dW2[i][j] += dz2[i] * net.a1[j];
        db2[i] += dz2[i];
    }

    vector<double> da1(HIDDEN_SIZE, 0.0);
    for (int j = 0; j < HIDDEN_SIZE; j++)
        for (int i = 0; i < OUTPUT_SIZE; i++)
            da1[j] += net.W2[i][j] * dz2[i];

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        double dz1 = (net.z1[j] > 0) ? da1[j] : 0.0;
        for (int k = 0; k < INPUT_SIZE; k++)
            dW1[j][k] += dz1 * x[k];
        db1[j] += dz1;
    }
}

void applyGradients(NeuralNet& net, const vector<vector<double>>& dW1, const vector<double>& db1,
                    const vector<vector<double>>& dW2, const vector<double>& db2, int bSize) {
    double scale = LEARNING_RATE / bSize;
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) net.W1[i][j] -= scale * dW1[i][j];
        net.b1[i] -= scale * db1[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) net.W2[i][j] -= scale * dW2[i][j];
        net.b2[i] -= scale * db2[i];
    }
}

double crossEntropyLoss(const vector<double>& probs, int label) {
    return -log(max(probs[label], 1e-12));
}

double evaluate(NeuralNet& net, const Dataset& data) {
    int correct = 0;
    for (size_t i = 0; i < data.X.size(); i++) {
        auto probs = forward(net, data.X[i]);
        if ((max_element(probs.begin(), probs.end()) - probs.begin()) == data.y[i])
            correct++;
    }
    return 100.0 * correct / data.X.size();
}

pair<Dataset, Dataset> split(const Dataset& data, double frac) {
    int N = data.X.size();
    vector<int> idx(N);
    iota(idx.begin(), idx.end(), 0);
    shuffle(idx.begin(), idx.end(), rng);
    int trainN = (int)(N * frac);
    Dataset tr, te;
    for (int i = 0; i < N; i++) {
        if (i < trainN) { tr.X.push_back(data.X[idx[i]]); tr.y.push_back(data.y[idx[i]]); }
        else { te.X.push_back(data.X[idx[i]]); te.y.push_back(data.y[idx[i]]); }
    }
    return make_pair(tr, te);
}

void saveModel(const NeuralNet& net, const Scaler& s, const string& path) {
    ofstream f(path);
    f << fixed << setprecision(8) << INPUT_SIZE << " " << HIDDEN_SIZE << " " << OUTPUT_SIZE << "\n";
    for (double v : s.mean) f << v << " "; f << "\n";
    for (double v : s.stdv) f << v << " "; f << "\n";
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) f << net.W1[i][j] << " ";
        f << "\n";
    }
    for (double v : net.b1) f << v << " "; f << "\n";
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) f << net.W2[i][j] << " ";
        f << "\n";
    }
    for (double v : net.b2) f << v << " "; f << "\n";
}

int main(int argc, char** argv) {
    string path = (argc > 1) ? argv[1] : "placement_data.csv";
    Dataset data = loadCSV(path);
    pair<Dataset, Dataset> splitData = split(data, TRAIN_SPLIT);
    Dataset train = splitData.first;
    Dataset test = splitData.second;

    Scaler s = fitScaler(train.X);
    applyScaler(train.X, s);
    applyScaler(test.X, s);

    NeuralNet net;
    initNet(net);

    cout << "Training...\n";
    vector<int> indices(train.X.size());
    iota(indices.begin(), indices.end(), 0);

    for (int epoch = 1; epoch <= EPOCHS; epoch++) {
        shuffle(indices.begin(), indices.end(), rng);
        double loss = 0;
        for (size_t start = 0; start < indices.size(); start += BATCH_SIZE) {
            size_t end = min(start + BATCH_SIZE, indices.size());
            vector<vector<double>> dW1(HIDDEN_SIZE, vector<double>(INPUT_SIZE, 0.0));
            vector<double> db1(HIDDEN_SIZE, 0.0);
            vector<vector<double>> dW2(OUTPUT_SIZE, vector<double>(HIDDEN_SIZE, 0.0));
            vector<double> db2(OUTPUT_SIZE, 0.0);

            for (size_t k = start; k < end; k++) {
                int i = indices[k];
                auto p = forward(net, train.X[i]);
                loss += crossEntropyLoss(p, train.y[i]);
                backward(net, train.X[i], train.y[i], dW1, db1, dW2, db2);
            }
            applyGradients(net, dW1, db1, dW2, db2, end - start);
        }
        if (epoch % 100 == 0 || epoch == 1)
            cout << "Epoch " << epoch << " | Loss: " << loss/train.X.size() 
                 << " | Train Acc: " << evaluate(net, train) << "%"
                 << " | Test Acc: " << evaluate(net, test) << "%\n";
    }
    saveModel(net, s, "model.txt");
    cout << "Done! Model saved to model.txt\n";
    return 0;
}