# 🚀 Multiclass Neural Network from Scratch in C++

A complete, production-ready implementation of a Multilayer Perceptron (MLP) Neural Network built **entirely from scratch** in C++ without the use of any external Machine Learning libraries (like TensorFlow, PyTorch, or Scikit-Learn). 

This project is designed to predict **student placement outcomes** (which company a student will be placed in) based on their academic and skill metrics.

---

## 🧠 Why Build From Scratch?

Modern ML frameworks abstract away the core mathematical principles of Neural Networks. This project strips away the "black box" to demonstrate exactly how data flows through a network, how error is calculated, and how backpropagation updates weights via calculus.

By looking at this code, you will understand:
- How **Forward Propagation** works with matrices and vectors.
- How **Backpropagation** actually calculates gradients layer-by-layer.
- Why **Data Normalization** (Z-score) is absolutely critical for model convergence.
- How **Softmax** combined with **Cross-Entropy Loss** works for multiclass classification.

---

## 🏗️ Neural Network Architecture

The model is a fully connected Feedforward Neural Network designed for a 10-class classification problem.

- **Input Layer:** 5 Neurons
  - `DSA Score` (0 - 100)
  - `Projects Built` (Integer count)
  - `IQ` (80 - 150)
  - `CGPA` (0.0 - 10.0)
  - `Attendance` (0% - 100%)
- **Hidden Layer:** 16 Neurons
  - **Activation:** `ReLU` (Rectified Linear Unit) — Introduces non-linearity to learn complex patterns.
  - **Initialization:** He/Xavier Initialization — Prevents vanishing/exploding gradients.
- **Output Layer:** 10 Neurons (representing 10 different companies)
  - **Activation:** `Softmax` — Converts the raw output scores (logits) into a probability distribution summing to 100%.

### Hyperparameters
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Mini-batch Gradient Descent
- **Learning Rate:** 0.01
- **Batch Size:** 32
- **Epochs:** 500
- **Train/Test Split:** 80% / 20%

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `nn_lib.h` | The core "engine". Contains the math, `NeuralNet` struct, data loaders, and the `Scaler` for normalization. |
| `placement_nn.cpp` | The **Trainer**. Loads the CSV, trains the model using backpropagation, prints accuracy metrics, and saves the learned weights to `model.txt`. |
| `predict.cpp` | The **Inference Engine**. Loads the pre-trained `model.txt` and provides an interactive terminal for you to input new student data and get live predictions. |
| `placement_data.csv` | The raw dataset containing 2,000 records of student metrics and their target companies. |
| `model.txt` | (Generated after training) Contains the saved weights, biases, and normalization parameters. |
| `Makefile` | Build script for automated compilation. |

---

## ⚙️ Setup and Installation

### Prerequisites
You need a C++ compiler (`g++`) installed on your system.
- **Windows:** Install MinGW.
- **Mac:** Install Xcode Command Line Tools (`xcode-select --install`).
- **Linux:** Install build-essential (`sudo apt install build-essential`).

---

## 🚀 How to Run (Windows PowerShell)

### Step 1: Compile the Code
You can compile the training and prediction scripts manually:
```powershell
# Compile the training script
g++ placement_nn.cpp -o train.exe

# Compile the prediction script
g++ predict.cpp -o predict.exe
```

### Step 2: Train the Model
Run the trainer to analyze the dataset and generate the model weights:
```powershell
.\train.exe
```
*Expected Output: You will see the training progress epoch by epoch, showing the loss decreasing and accuracy increasing.*

### Step 3: Make Predictions
Once `model.txt` is generated, run the predictor:
```powershell
.\predict.exe
```

---

## 🚀 How to Run (Linux / Mac)

If you are on a Unix-based system (or using Git Bash/WSL on Windows), you can use the provided `Makefile`:

```bash
# Compile everything
make

# Train the model
./train

# Make predictions
./predict
```

---

## 💻 Usage Example

When you run the prediction engine, the interactive terminal will ask for 5 inputs sequentially. You can type them one by one and press Enter, or copy-paste a space-separated list.

```text
Placement Predictor loaded (5 inputs)
Enter (DSA, Projects, IQ, CGPA, Attendance) or 'q': 
85
4
120
8.5
90

Prediction: Company 8 (Confidence: 87.5%)
```

*(Type `q` and press Enter to quit the program).*

---

## 🔧 Future Improvements
- **Dynamic Architecture:** Allow reading the number of hidden layers/neurons from a config file.
- **Save/Load Binary:** Currently, `model.txt` is plain text for educational readability. Converting this to a binary format would make loading faster.
- **Company Mapping:** Map the output IDs (0-9) to actual company names (e.g., Google, Microsoft, Amazon) for better User Experience.
