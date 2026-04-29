# Performance Report — Task 1 (CIFAR-10)

This report validates the final stabilized codebase by presenting training,
validation, and test performance for both the MLP and CNN models on the
CIFAR-10 dataset. The objective is to verify that both models meet the
required performance targets.

---


## 1. MLP Performance

### 1.1 Training and Validation Curves

The following figure shows the training and validation loss and accuracy of MLP over 5 epochs:

- `checkpoints/training_history_mlp.png`
---

### 1.2 Final Evaluation Metrics (MLP)

| Metric | Value |
|------|------|
| Best Validation Accuracy | 50.22% |
| Test Accuracy | 51.28% |
| Test Loss | 1.3675 |

**Comment:**
The MLP exceeds the required **50% test accuracy** threshold. Validation and
test accuracy are consistent, indicating stable generalization.

---

## 2. CNN Performance

### 2.1 Training and Validation Curves

The following figure shows the training and validation loss and accuracy of CNN
over 5 epochs:

- `checkpoints/training_history_cnn.png`

---

### 2.2 Final Evaluation Metrics (CNN)

| Metric | Value |
|------|------|
| Best Validation Accuracy | 77.10% |
| Test Accuracy | 76.44% |
| Test Loss | 0.6943 |

**Comment:**
The CNN exceeds the required **75% test accuracy** threshold. Validation and test
accuracy are closely aligned, confirming correct evaluation behavior.

---

## 3. Performance Target Verification

| Model | Required Accuracy | Achieved Test Accuracy | Target Met |
|------|------------------|------------------------|------------|
| MLP | ≥ 50% | 51.28% | Yes |
| CNN | ≥ 75% | 76.44% | Yes |

---

## 4. Conclusion

- The stabilized codebase trains and evaluates correctly on CIFAR-10.
- Both models meet or exceed the required performance targets.
- Loss curves and accuracy trends demonstrate stable optimization and valid
  evaluation.
- Task 1 performance requirements are fully satisfied.
