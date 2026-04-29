# Changelog — IDL25 Exam Retake (Task 1)

I have listed all the fixes applied to stabilize the original codebase and restore correct training behavior on the CIFAR-10 dataset.  


## Fix Log

| Commit SHA | Commit Message | File / Location | Root Cause | Fix Description | How Issue Was Discovered |
|------------|---------------|-----------------|------------|------------------|--------------------------|

| `6ccfb98cdce063583460d40f6344253e50a06aac` | Fix: Correct CIFAR-10 normalisation values (mean and Standard Deviation) | `data_loader.py` (Line 25 & 26) | Incorrect dataset statistics were used (`negative mean` and `std = 1e-5`), causing unstable inputs leading to unstable optimization. | Replaced normalization values with official CIFAR-10 mean and standard deviation: `[0.4914, 0.4822, 0.4465]`, `[0.2470, 0.2435, 0.2616]`. | Training loss behaved erratically and accuracy stayed near chance. Also used torchvision CIFAR-10 reference statistics to check what should be the mean and std dev in actual. |


| `a58d4e907c39279b52228fbb1059ea2f8ce0b6c6` | Fix: Remove Final RELU layer from MLP and CNN classification output | `models.py` (Line 28 for MLP & 71 for CNN) | A ReLU layer was applied after the final classification (linear) layer, applying a non-linear activation as the final layer which restricts output values and limits the model’s ability to represent class score differences. | Removed the final ReLU so both models output raw class scores (logits). | Found out during architectural review and by observing that the output layer unnecessarily clipped negative class scores. |


| `7a956df9a5de407247f54cda13acc93e858b80c6` | Fix: Added RELU layer between CNN layer to add nonlinearity in classifier | `models.py` (Line 70) | The CNN classifier lacked a non-linear activation between fully connected layers, making it effectively linear. | Inserted a ReLU activation between the CNN’s fully connected layers, after BatchNorm1D Layer which improves training stability and ensures proper non-linearity in the classifier | Found out during architectural review when there was a RELU layer at the end and not in the middle. |


| `79bd2f81dc2cf42400a0643411f16d14f4c89d89` | Fix: change CNN and MLP input layer feature size. | `models.py` (Line 14 for MLP and 68 for CNN ) | Input feature dimensions did not match the actual tensor sizes produced by CIFAR-10 images and convolutional layers. | Updated MLP input size to 3×32×32 and CNN fully connected input size to 64×8×8. | Found when output showed error for incorrect matrix multiplication and by manually tracing tensor shapes through the network and validating layer dimensions. |


| `11b7508887f6060fb24a5d1e29bc9191491df90d` | Fix: replace NLLoss with CrossEntropyLoss | `train.py` (Line 74) | The loss function expected log-probabilities, while the model produced raw class scores. NLLLoss was used without LogSoftmax, resulting in incorrect loss computation. | Replaced NLLLoss with CrossEntropyLoss, which internally applies LogSoftmax. | Discovered by reviewing the loss definition and comparing it with the model’s output format. Verified by inspecting loss definition and PyTorch documentation. |


| `89b854f6384f71d95042b01776fc06c9157f4403` | Fix: added zero_grad before backpropagation | `utils.py` (Line 31) | Gradients were implicitly accumulated across batches, leading to incorrect parameter updates. | Added `optimizer.zero_grad()` before backpropagation to reset gradients each iteration. | Observed unstable training behavior and verified PyTorch’s default gradient accumulation mechanism. |


| `c16823c7d2accca407c9ed8cf42b871b6bea1741` | Fix: set model to train mode during training | `utils.py` (Line 24) | The model was not explicitly set to training mode, causing BatchNorm and Dropout layers to behave incorrectly during training. | Added `model.train()` at the start of each training epoch. | Identified by reviewing training logic and recognizing that BatchNorm and Dropout layers require explicit training mode to behave correctly.|


| `34deffdaf883e8b8e4112e7323947b0c425b9967` | Fix: changed train_loader to test_loader for it to evaluation on test set | `train.py (evaluation section)` (Line 146) | Evaluation was mistakenly performed on the training dataset, resulting in misleading performance metrics. | Updated evaluation logic to use the test data loader instead of the training loader.| Detected by reviewing evaluation code and noticing that reported “test accuracy” was computed on training data. |




