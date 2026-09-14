# Phase 5D: EfficientNet-B3 Baseline — Final Comprehensive Report

**Project**: Automated Classification of Martian Surface Images Captured by NASA's Curiosity Rover Using Machine Learning  
**Platform**: NASA Mars Science Laboratory (MSL Curiosity Rover)  
**Model Architecture**: EfficientNet-B3 (Compound-Scaled Convolutional Neural Network)  
**Pretrained Source**: PyTorch Torchvision EfficientNet_B3_Weights.DEFAULT (IMAGENET1K_V1)  
**Hardware Engine**: Apple M3 Pro (Metal Performance Shaders / MPS)  
**Execution Timestamp**: 2026-09-13 18:28:13  

---

## 1. Objective
Phase 5D implements and evaluates an **ImageNet-pretrained EfficientNet-B3** image classifier on the NASA Curiosity Rover dataset. The primary scientific objective is to determine experimentally whether compound scaling (jointly balancing network depth, width, and resolution at $300 \times 300$) can surpass the project champion (Phase 5A ResNet-50) and other established architectures (Phase 4B KNN-4, Phase 5B MRSCAtt, Phase 5C ViT-B/16).

---

## 2. Dataset
- **Repository Path**: `Mars rover data/`
- **Dataset Origin**: NASA Mars Science Laboratory (MSL) Curiosity rover browse imagery (Sols 3 to 1060).
- **Format**: Calibrated JPEG browse images standardized across Mastcam and MAHLI instruments.

---

## 3. Dataset Integrity
- **Physical Image Verification**: All 6,691 referenced images physically exist in `Mars rover data/calibrated/`.
- **Zero Modifications**: No images were modified, cropped, deleted, or generated synthetically.
- **Split File Immutability**: `train-calibrated-shuffled.txt`, `val-calibrated-shuffled.txt`, and `test-calibrated-shuffled.txt` were used exactly as officially distributed.
- **Split Overlap**: Programmatically verified zero intersection across train, validation, and test splits.

---

## 4. Class Distribution & Active Classes
- **Total Synset Classes**: 25 classes defined in `msl_synset_words-indexed.txt` (IDs 0..24).
- **Active Classes**: Exactly **24 active classes**.
- **Zero-Instance Inactive Class**: Class 22 (`sun`) contains 0 samples across all calibrated splits and was strictly excluded.
- **Official Split Sample Counts**:
  - Training Set: **3,746** images
  - Validation Set: **1,640** images
  - Test Set: **1,305** images
  - Total Verified: **6,691** images

---

## 5. Preprocessing Pipeline
1. Load raw image via PIL directly from disk and convert to 3-channel RGB.
2. Apply aspect-ratio-preserving `LetterboxResize` to $300 \times 300$ with constant neutral black padding.
3. Convert uint8 $[0, 255]$ pixels to float32 tensor $[0.0, 1.0]$.
4. Normalize using standard ImageNet statistics ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
5. Validation and test preprocessing pipelines are strictly deterministic with zero random operations.

---

## 6. Training Augmentation
Applied exclusively to the training split:
- Letterbox resize to $343 \times 343$
- RandomResizedCrop to $300 \times 300$ (scale 0.85–1.0, aspect ratio 0.9–1.1)
- Small random rotation ($\pm 10^\circ$)
- Mild ColorJitter (brightness $\pm 10\%$, contrast $\pm 10\%$)
- Random horizontal flip ($p = 0.5$)
- Zero stochastic transformations applied to validation or test splits.

---

## 7. EfficientNet-B3 Architecture
- **Input Resolution**: $300 \times 300 \times 3$
- **Stem Convolution**: $3 \times 3$ Conv2d ($3 \to 40$ channels, stride 2) + BatchNorm2d + SiLU.
- **Backbone Stages**: 7 MBConv stages utilizing depthwise separable convolutions, squeeze-and-excitation (SE) attention, and inverted residuals.
- **Head Feature Map**: 1,536 channels after final $1 \times 1$ convolution.
- **Classifier Head**: Replaced 1,000-class ImageNet head with `Sequential(Dropout(p=0.3), Linear(1536, 24))` outputting 24 active NASA classes.

---

## 8. Parameter Counts (Programmatically Calculated)
- **Total Parameters**: **10,733,120**
- **Trainable Parameters**: **10,733,120**
- **Frozen Parameters**: **0**
- *(Calculated directly from model instance using `sum(p.numel() for p in model.parameters())`)*.

---

## 9. Training Configuration
- **Optimizer**: AdamW ($	ext{lr} = 10^{-4}$, weight decay $= 10^{-4}$)
- **Learning Rate Schedule**: `CosineAnnealingLR` decaying from $10^{-4}$ to $10^{-6}$ across 15 epochs
- **Batch Size**: 32 samples per batch (117 batches per epoch)
- **Epochs**: 15
- **Random Seed**: 42 (fixed across Python, NumPy, and PyTorch)
- **Hardware Acceleration**: mps

---

## 10. Class Weighting Strategy
Standard balanced inverse-frequency weights computed strictly and exclusively from `y_train`:
$$w_c = \frac{N_{train}}{K \cdot N_{train,c}} \quad (K=24, N_{train}=3746)$$
Zero arbitrary capping, clipping, or manual modifications were applied. Validation and test labels were never accessed for weighting.

---

## 11. Validation Methodology
After every epoch, the model was evaluated exclusively on all 1,640 validation images.
Metrics computed per epoch:
- Validation Loss
- Validation Accuracy
- Macro Precision, Recall, and Macro-F1
- Weighted-F1
- Per-class precision, recall, and F1

---

## 12. Complete Training History
The per-epoch training and validation telemetry was logged and saved to `training_history.csv`.
- Best Validation Macro-F1 reached at **Epoch 6**.
- Curves plotted and saved to `results/phase5d/plots/`.

---

## 13. Best Checkpoint
- **Filename**: `best_efficientnet_b3.pth`
- **Location**: `saved_models/deep_learning/efficientnet_b3/best_efficientnet_b3.pth`
- **Selection Criterion**: Highest **Validation Macro-F1** (primary), Validation Accuracy (secondary), earlier epoch (tie-breaker).
- **Checkpoint Contents**: State dict, epoch, validation metrics, model metadata, class mapping, random seed.

---

## 14. Best Validation Metrics
- **Best Epoch**: **Epoch 6**
- **Validation Accuracy**: **97.38%**
- **Validation Macro-F1**: **0.7417**
- **Validation Macro Precision**: **0.7444**
- **Validation Macro Recall**: **0.7606**
- **Validation Weighted-F1**: **0.9755**
- **Validation Loss**: **0.2897**

---

## 15. Final Test Protocol
The official test split ($N=1,305$) was isolated throughout the entire training process.
It was evaluated **EXACTLY ONCE** after:
1. Completing all training epochs.
2. Saving validation history.
3. Locking the best checkpoint.
4. Reloading and verifying the best checkpoint.
Zero hyperparameter tuning or model selection used the test set.

---

## 16. Final Test Metrics
- **Test Accuracy**: **80.61%** (1052/1,305 correct)
- **Test Macro-F1**: **0.7083**
- **Test Macro Precision**: **0.7307**
- **Test Macro Recall**: **0.7487**
- **Test Weighted-F1**: **0.8288**
- **Test Weighted Precision**: **0.8988**
- **Test Weighted Recall**: **0.8061**
- **Test Cross-Entropy Loss**: **1.0690**
- **Total Test Inference Time**: **20.34 seconds**
- **Inference Latency Per Sample**: **15.59 ms** (64.2 images/second)

---

## 17. Complete Per-Class Classification Report

| Active Idx | Class Name | Precision | Recall | F1-Score | Support | Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
|  0 | `apxs` | 0.9444 | 1.0000 | **0.9714** | 34 | Active |
|  1 | `apxs cal target` | 1.0000 | 1.0000 | **1.0000** | 14 | Active |
|  2 | `chemcam cal target` | 1.0000 | 1.0000 | **1.0000** | 21 | Active |
|  3 | `chemin inlet open` | 1.0000 | 0.6190 | **0.7647** | 84 | Active |
|  4 | `drill` | 0.5556 | 0.7500 | **0.6383** | 20 | Active |
|  5 | `drill holes` | N/A | N/A | **N/A** | 0 | Zero test support in official split |
|  6 | `drt front` | 0.9512 | 0.6500 | **0.7723** | 60 | Active |
|  7 | `drt side` | 1.0000 | 0.6000 | **0.7500** | 150 | Active |
|  8 | `ground` | 0.9841 | 0.7323 | **0.8397** | 254 | Active |
|  9 | `horizon` | 0.5702 | 0.9583 | **0.7150** | 72 | Active |
| 10 | `inlet` | 0.1389 | 0.3125 | **0.1923** | 16 | Active |
| 11 | `mahli` | 0.5263 | 0.8333 | **0.6452** | 12 | Active |
| 12 | `mahli cal target` | 1.0000 | 0.8070 | **0.8932** | 57 | Active |
| 13 | `mastcam` | 1.0000 | 0.8438 | **0.9153** | 32 | Active |
| 14 | `mastcam cal target` | 1.0000 | 0.8333 | **0.9091** | 48 | Active |
| 15 | `observation tray` | 0.8000 | 1.0000 | **0.8889** | 12 | Active |
| 16 | `portion box` | 1.0000 | 0.8333 | **0.9091** | 48 | Active |
| 17 | `portion tube` | 1.0000 | 0.7778 | **0.8750** | 9 | Active |
| 18 | `portion tube opening` | 0.6667 | 1.0000 | **0.8000** | 2 | Active |
| 19 | `rems uv sensor` | 1.0000 | 0.6667 | **0.8000** | 36 | Active |
| 20 | `rover rear deck` | 0.3871 | 0.8571 | **0.5333** | 14 | Active |
| 21 | `scoop` | 0.1579 | 0.9000 | **0.2687** | 10 | Active |
| 22 | `turret` | N/A | N/A | **N/A** | 0 | Zero test support in official split |
| 23 | `wheel` | 0.8539 | 0.9933 | **0.9183** | 300 | Active |

---

## 18. Confusion Matrix Analysis
- Raw count confusion matrix saved to `results/phase5d/confusion_matrices/test_confusion_matrix.png`.
- High precision and recall were achieved across distinct mechanical components (`apxs`, `apxs cal target`, `chemcam cal target`, `observation tray`).

---

## 19. Normalized Confusion Matrix Analysis
- Normalized recall matrix saved to `results/phase5d/confusion_matrices/test_confusion_matrix_normalized.png`.
- Examines true positive recall rates per class independent of support discrepancies.

---

## 20. Runtime Performance
- **Total Training Time**: **3038.0 seconds** (50.63 minutes) on Apple M3 Pro GPU.
- **Single-Pass Test Inference**: **20.34 seconds** for all 1,305 test images.
- **Throughput**: **64.2 samples/second**.

---

## 21. Comparison with Phase 4B (Classical ML Champion)
Phase 4B KNN-4 (8,186 handcrafted features + PCA-100) achieved **47.20% Test Accuracy** and **0.5407 Test Macro-F1**.  
EfficientNet-B3 achieved **80.61% Test Accuracy** and **0.7083 Test Macro-F1**, demonstrating a dramatic performance improvement over classical handcrafted pipelines.

---

## 22. Comparison with Phase 5A (ResNet-50 Baseline)
Phase 5A ResNet-50 achieved **77.78% Test Accuracy** and **0.6631 Test Macro-F1**.  
Comparison:
- ResNet-50 Test Accuracy: 77.78% vs EfficientNet-B3: **80.61%** (Delta: +2.83%)
- ResNet-50 Test Macro-F1: 0.6631 vs EfficientNet-B3: **0.7083** (Delta: +0.0452)

---

## 23. Comparison with Phase 5B (MRSCAtt)
Phase 5B MRSCAtt achieved **64.29% Test Accuracy** and **0.5851 Test Macro-F1**.  
Comparison:
- MRSCAtt Test Accuracy: 64.29% vs EfficientNet-B3: **80.61%** (Delta: +16.32%)
- MRSCAtt Test Macro-F1: 0.5851 vs EfficientNet-B3: **0.7083** (Delta: +0.1232)

---

## 24. Comparison with Phase 5C (ViT-B/16 Baseline)
Phase 5C ViT-B/16 achieved **73.26% Test Accuracy** and **0.6536 Test Macro-F1**.  
Comparison:
- ViT-B/16 Test Accuracy: 73.26% vs EfficientNet-B3: **80.61%** (Delta: +7.35%)
- ViT-B/16 Test Macro-F1: 0.6536 vs EfficientNet-B3: **0.7083** (Delta: +0.0547)

---

## 25. Actual Strengths of EfficientNet-B3 on Martian Imagery
1. **Compound Scaling Efficiency**: Balances depth, width, and resolution ($300 \times 300$) with only 10.7M parameters—substantially fewer than ResNet-50 (23.6M) and ViT-B/16 (85.8M).
2. **Squeeze-and-Excitation Mechanisms**: Channel-wise attention recalibration within each MBConv block allows selective enhancement of informative Martian spectral and textural channels.
3. **High-Resolution Feature Representation**: $300 \times 300$ input captures finer rock grain textures and calibration target details than $224 \times 224$ models.

---

## 26. Actual Weaknesses of EfficientNet-B3
1. **Depthwise Convolution Computational Overhead**: Despite fewer FLOPs and parameters, depthwise separable convolutions exhibit lower GPU memory bandwidth arithmetic intensity compared to dense convolutions.
2. **Resolution-Induced Memory Footprint**: Operating at $300 \times 300$ increases activation tensor sizes across early stages.

---

## 27. Actual Failure Cases & Confusions
- **Visually Similar Apertures**: Fine-grained confusions between circular drill and inlet apertures (`inlet` vs `chemin inlet open`).
- **Dust Accumulation on Metal**: Rover hardware heavily coated with atmospheric dust can occasionally be confused with background `ground`.

---

## 28. Whether the 80% Target Was Achieved
- **Target**: 80.00% Test Accuracy.
- **Actual Phase 5D Test Accuracy**: **80.61%**.
- **Conclusion**: **Phase 5D achieved the 80% test-accuracy target.**

---

## 29. Integrity Verification Results
- [x] Original dataset completely unchanged
- [x] Official train/validation/test split files unchanged
- [x] 3,746 training references verified
- [x] 1,640 validation references verified
- [x] 1,305 test references verified
- [x] Exactly 24 active classes mapped
- [x] Class 22 excluded (0 samples)
- [x] Zero train/test overlap; zero validation/test overlap
- [x] Zero test evaluation during training
- [x] Best checkpoint selected strictly using validation Macro-F1
- [x] Final test evaluated strictly ONCE
- [x] All test predictions, labels, and probabilities saved
- [x] Metrics recomputed from saved predictions match within tolerance $10^{-5}$
- [x] Checkpoint reload verified with 100% parameter alignment
- [x] Phase 4B, Phase 5A, Phase 5B, and Phase 5C artifacts completely untouched
- [x] All reported numbers generated from actual execution

---

## 30. Artifacts Created
- Code: `deep_learning/models/efficientnet_b3.py`, `run_phase5d.py`
- Models: `saved_models/deep_learning/efficientnet_b3/best_efficientnet_b3.pth`, `last_efficientnet_b3.pth`
- Reports: `results/phase5d/classification_reports/` (CSV and TXT)
- Confusion Matrices: `results/phase5d/confusion_matrices/` (raw NPY, raw PNG, normalized PNG)
- Plots: `results/phase5d/plots/` (loss, accuracy, Macro-F1, per-class F1)
- Predictions: `results/phase5d/predictions/` (predictions, labels, probabilities)
- History & Metrics: `results/phase5d/training_history.csv`, `final_metrics.json`, `metadata.json`
- Benchmark: `results/phase5d/benchmark_comparison.csv`
- Notebook: `notebooks/05d_efficientnet_b3.ipynb`

---

## 31. Final Phase 5D Conclusion
EfficientNet-B3 was successfully implemented, fine-tuned, and evaluated on the official NASA Curiosity Rover dataset. The model achieved **80.61% Test Accuracy** and **0.7083 Test Macro-F1**, confirming its effectiveness for automated Martian surface image classification and providing a rigorous compound-scaled benchmark for comparison with ResNet-50, MRSCAtt, and ViT-B/16.
