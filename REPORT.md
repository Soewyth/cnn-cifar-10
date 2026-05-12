# Report — ResNet18 on CIFAR-10

## Environment

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA RTX 3070 Ti |
| Batch size | 128 |
| Max epochs | 15 |


---

## Results Summary

| Run | Input | Freeze | LR | Best Acc | Best Epoch | Epochs Run | Time | VRAM | GPU Util |
|-----|-------|--------|----|----------|------------|------------|------|------|----------|
| 1 | 32×32 | Yes | 0.001 | 41.3% | 5 | 10/15 | 24s | 737 MB | 74% |
| 2 | 224×224 | Yes | 0.001 | 80.5% | 14 | 15/15 | 389s | 1956 MB | 93% |
| 3 | 224×224 | No | 0.0001 | 95.5% | 7 | 12/15 | 779s | 5743 MB | 94% |

---

## Run Notes

### Run 1 — Wrong input size
Input resized to 32×32 before feeding a ResNet18 pretrained on ImageNet (expected 224×224). The backbone cannot extract meaningful features at that resolution, leading to unusable representations and low accuracy (41.3%). No overfitting observed.

### Run 2 — Frozen backbone, correct input size
Input correctly resized to 224×224. Backbone frozen, only the classifier head is trained. Good convergence: both train and test loss decrease steadily, reaching ~0.6. Accuracy reaches **80.5%**. No overfitting. ~26s per epoch. VRAM usage increases due to larger input size, but remains manageable.

### Run 3 — Full fine-tuning (backbone unfrozen)
Backbone unfrozen with a lower LR (0.0001). Strong convergence early on, but **overfitting starts at epoch 8** — train loss drops to ~0.03 while test loss plateaus at ~0.17. Best accuracy of **95.5%** reached at epoch 7, after which performance degrades. ~64s per epoch due to full backprop through the backbone. VRAM usage is significantly higher due to full fine-tuning.

# Limitations & Improvements
- Progressively fine tuning : start with frozen backbone and adjust gradually some parameters like the lr, the number of epochs, the batch size (to reduce VRAM usage but it will increase training time), and the number of layers to unfreeze.
- More data augmentation to reduce overfitting.
