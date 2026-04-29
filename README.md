# 🧠 Deep Learning Debugging & Optimization (CIFAR-10 → CIFAR-100)

## 📌 Project Overview

This project simulates a real-world deep learning engineering scenario where I inherited a broken and poorly maintained codebase and was tasked with:

1. **Stabilizing the system** by identifying and fixing critical bugs  
2. **Improving and optimizing the model** for a more complex dataset  

The project was completed as part of my Master's in Artificial Intelligence at THWS.

---

## 🎯 Objectives

### 🔧 Task 1 – Debug & Stabilize (CIFAR-10)
- Identify and fix bugs in an existing PyTorch codebase
- Restore correct functionality with minimal code changes
- Ensure models meet performance targets:
  - MLP → >50% accuracy
  - CNN → >75% accuracy

---

### 🚀 Task 2 – Optimize & Scale (CIFAR-100)
- Extend the stabilized system to a more complex dataset
- Improve performance using systematic experimentation
- Target:
  - ≥50% accuracy (≥65% considered strong)

---

## ⚙️ Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib

---

## 🧱 Model Architectures

### 🔹 Multi-Layer Perceptron (MLP)
- 2 hidden layers
- Batch Normalization
- ReLU activation
- Dropout for regularization

---

### 🔹 Convolutional Neural Network (CNN)
- 2 convolutional blocks:
  - Conv → BatchNorm → ReLU
  - Conv → BatchNorm → ReLU
  - MaxPooling
- Fully connected layers for classification

---

## 🐛 Task 1 – Debugging the Codebase

The original codebase contained multiple subtle bugs introduced through small changes.

### 🔍 Approach:
- Carefully reviewed code line-by-line
- Tested components individually
- Fixed issues with minimal modifications
- Maintained original structure (as required)

---

### ✅ Deliverables:
- Clean and working codebase
- Semantic commit history
- Tagged final version: `task1-final`

---

### 📊 Performance (CIFAR-10)
- MLP achieved >50% accuracy
- CNN achieved >75% accuracy

(See `performance_report.md` for details)

---

### 📄 Changelog
All fixes are documented in:

👉 `changelog.md`

Includes:
- Commit SHA
- Bug description
- Root cause
- Fix applied

---

## 🚀 Task 2 – Optimization & Improvements

After stabilizing the system, I improved model performance on CIFAR-100.

---

### 🔬 Approach:
- Incremental experimentation
- Changed one component at a time
- Tracked performance after each change

---

### ⚡ Improvements Explored:
- Hyperparameter tuning (learning rate, dropout, weight decay)
- Data augmentation
- Architecture extensions
- Training improvements (e.g., scheduling)

---

### 📊 Results (CIFAR-100)
- Achieved competitive performance (>50% accuracy)
- Significant improvement over baseline

---

### 📈 Ablation Study
All experiments are documented in:

👉 `ablation_study.md`

---


