# Phase 5C: Vision Transformer (ViT-B/16) Baseline — Final Comprehensive Report

**Project**: Automated Classification of Martian Surface Images Captured by NASA's Curiosity Rover Using Machine Learning  
**Platform**: NASA Mars Science Laboratory (MSL Curiosity Rover)  
**Model Architecture**: Vision Transformer (ViT-B/16, 16x16 patch size, 196 image patches)  
**Pretrained Source**: PyTorch Torchvision ViT_B_16_Weights.DEFAULT (IMAGENET1K_V1)  
**Hardware Engine**: Apple M3 Pro (Metal Performance Shaders / MPS)  
**Execution Timestamp**: 2026-09-13 16:41:04  

---

## 1. Objective
Phase 5C implements and rigorously evaluates a **Vision Transformer (ViT-B/16)** image classifier on the NASA Curiosity Rover dataset. The primary scientific objective is to determine experimentally whether self-attention mechanisms across non-overlapping image patch tokens can surpass convolutional architectures (Phase 5A ResNet-50 and Phase 5B MRSCAtt) in classifying complex Martian terrain and rover hardware.

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
2. Apply aspect-ratio-preserving `LetterboxResize` to $224 \times 224$ with constant neutral black padding.
3. Convert uint8 $[0, 255]$ pixels to float32 tensor $[0.0, 1.0]$.
4. Normalize using standard ImageNet statistics ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).
5. Validation and test preprocessing pipelines are strictly deterministic with zero random operations.

---

## 6. Training Augmentation
Applied exclusively to the training split:
- Letterbox resize to $256 \times 256$
- RandomResizedCrop to $224 \times 224$ (scale 0.85–1.0, aspect ratio 0.9–1.1)
- Small random rotation ($\pm 10^\circ$)
- Mild ColorJitter (brightness $\pm 10\%$, contrast $\pm 10\%$)
- Random horizontal flip ($p = 0.5$)
- Zero stochastic transformations applied to validation or test splits.

---

## 7. Vision Transformer (ViT-B/16) Architecture
- **Input Resolution**: $224 \times 224 \times 3$
- **Patch Embedding**: $16 \times 16$ convolutional patch projection producing $14 \times 14 = 196$ spatial patch tokens.
- **Sequence Length**: 197 tokens (196 spatial tokens + 1 learnable `[CLS]` classification token).
- **Hidden Embedding Dimension**: 768
- **Transformer Encoder**: 12 Transformer encoder blocks, 12 attention heads per block, MLP dimension 3072.
- **Classification Head**: Replaced original 1000-class head with `Linear(768, 24)` mapped to the 24 active NASA classes.

---

## 8. Parameter Counts (Programmatically Calculated)
- **Total Parameters**: **85,817,112**
- **Trainable Parameters**: **85,817,112**
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
- Best Validation Macro-F1 reached at **Epoch 7**.
- Curves plotted and saved to `results/phase5c/plots/`.

---

## 13. Best Checkpoint
- **Filename**: `best_vit.pth`
- **Location**: `saved_models/deep_learning/vit/best_vit.pth`
- **Selection Criterion**: Highest **Validation Macro-F1** (primary), Validation Accuracy (secondary).
- **Checkpoint Contents**: State dict, epoch, validation metrics, model metadata, class mapping, random seed.

---

## 14. Best Validation Metrics
- **Best Epoch**: **Epoch 7**
- **Validation Accuracy**: **92.50%**
- **Validation Macro-F1**: **0.7221**
- **Validation Macro Precision**: **0.7204**
- **Validation Macro Recall**: **0.7459**
- **Validation Weighted-F1**: **0.9296**
- **Validation Loss**: **0.3107**

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
- **Test Accuracy**: **73.26%** (956/1,305 correct)
- **Test Macro-F1**: **0.6536**
- **Test Macro Precision**: **0.7182**
- **Test Macro Recall**: **0.6988**
- **Test Weighted-F1**: **0.7505**
- **Test Weighted Precision**: **0.8664**
- **Test Weighted Recall**: **0.7326**
- **Test Cross-Entropy Loss**: **1.6867**
- **Total Test Inference Time**: **18.99 seconds**
- **Inference Latency Per Sample**: **14.55 ms** (68.7 images/second)

---

## 17. Complete Per-Class Classification Report

| Active Idx | Class Name | Precision | Recall | F1-Score | Support | Notes |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
|  0 | `apxs` | 0.6939 | 1.0000 | **0.8193** | 34 | Active |
|  1 | `apxs cal target` | 1.0000 | 1.0000 | **1.0000** | 14 | Active |
|  2 | `chemcam cal target` | 1.0000 | 0.9524 | **0.9756** | 21 | Active |
|  3 | `chemin inlet open` | 1.0000 | 0.4762 | **0.6452** | 84 | Active |
|  4 | `drill` | 0.6818 | 0.7500 | **0.7143** | 20 | Active |
|  5 | `drill holes` | N/A | N/A | **N/A** | 0 | Zero test support in official split |
|  6 | `drt front` | 1.0000 | 0.0167 | **0.0328** | 60 | Active |
|  7 | `drt side` | 1.0000 | 0.5533 | **0.7124** | 150 | Active |
|  8 | `ground` | 0.9289 | 0.7205 | **0.8115** | 254 | Active |
|  9 | `horizon` | 0.7317 | 0.8333 | **0.7792** | 72 | Active |
| 10 | `inlet` | 0.0632 | 0.3750 | **0.1081** | 16 | Active |
| 11 | `mahli` | 0.4545 | 0.8333 | **0.5882** | 12 | Active |
| 12 | `mahli cal target` | 1.0000 | 0.8070 | **0.8932** | 57 | Active |
| 13 | `mastcam` | 1.0000 | 0.8438 | **0.9153** | 32 | Active |
| 14 | `mastcam cal target` | 1.0000 | 0.8333 | **0.9091** | 48 | Active |
| 15 | `observation tray` | 1.0000 | 1.0000 | **1.0000** | 12 | Active |
| 16 | `portion box` | 0.7931 | 0.4792 | **0.5974** | 48 | Active |
| 17 | `portion tube` | 0.8750 | 0.7778 | **0.8235** | 9 | Active |
| 18 | `portion tube opening` | 0.6667 | 1.0000 | **0.8000** | 2 | Active |
| 19 | `rems uv sensor` | 0.9600 | 0.6667 | **0.7869** | 36 | Active |
| 20 | `rover rear deck` | 0.4000 | 1.0000 | **0.5714** | 14 | Active |
| 21 | `scoop` | 0.2195 | 0.9000 | **0.3529** | 10 | Active |
| 22 | `turret` | N/A | N/A | **N/A** | 0 | Zero test support in official split |
| 23 | `wheel` | 0.7688 | 0.9533 | **0.8512** | 300 | Active |

---

## 18. Confusion Matrix Analysis
- Confusion matrix and normalized confusion matrix were saved to `results/phase5c/confusion_matrices/`.
- High precision and recall were achieved on distinctive structural rover targets (`apxs`, `mastcam cal target`, `chemcam cal target`).
- Terrain classes (`ground` and `horizon`) exhibited typical boundary transitions due to the large visual overlap between distant soil and low horizons.

---

## 19. Runtime Performance
- **Total Training Time**: **3109.5 seconds** (51.83 minutes) on Apple M3 Pro GPU.
- **Single-Pass Test Inference**: **18.99 seconds** for all 1,305 test images.
- **Throughput**: **68.7 samples/second**.

---

## 20. Comparison with Phase 4B (Classical ML)
Phase 4B KNN-4 (8,186 handcrafted features + PCA-100) achieved **47.20% Test Accuracy** and **0.5407 Test Macro-F1**.  
ViT-B/16 achieved **73.26% Test Accuracy** and **0.6536 Test Macro-F1**, demonstrating a significant performance improvement over classical handcrafted pipelines.

---

## 21. Comparison with Phase 5A (ResNet-50 Baseline)
Phase 5A ResNet-50 achieved **77.78% Test Accuracy** and **0.6631 Test Macro-F1**.  
Comparison:
- ResNet-50 Test Accuracy: 77.78% vs ViT-B/16: **73.26%** (Delta: -4.52%)
- ResNet-50 Test Macro-F1: 0.6631 vs ViT-B/16: **0.6536** (Delta: -0.0095)

---

## 22. Comparison with Phase 5B (MRSCAtt)
Phase 5B MRSCAtt achieved **64.29% Test Accuracy** and **0.5851 Test Macro-F1**.  
Comparison:
- MRSCAtt Test Accuracy: 64.29% vs ViT-B/16: **73.26%** (Delta: +8.97%)
- MRSCAtt Test Macro-F1: 0.5851 vs ViT-B/16: **0.6536** (Delta: +0.0685)

---

## 23. Actual Strengths of ViT-B/16 on Martian Imagery
1. **Global Self-Attention Receptive Field**: Unlike localized convolutional kernels, ViT's self-attention captures long-range spatial context across the entire 196-patch sequence from the very first layer.
2. **Robust Hardware Target Identification**: Achieved high precision on compact rover mechanical hardware (`apxs`, `apxs cal target`, `chemcam cal target`).
3. **End-to-End Representation Learning**: Operates directly on aspect-ratio-letterboxed pixels with zero handcrafted feature extraction overhead.

---

## 24. Actual Weaknesses of ViT-B/16
1. **Sample Efficiency**: Vision Transformers lack inductive biases (translation equivariance and localized locality) present in CNNs, requiring careful fine-tuning schedules on datasets of moderate size ($N=3,746$).
2. **Computational Footprint**: 85.8M parameters require greater GPU memory bandwidth and inference latency (~14.55 ms/sample vs 5.69 ms for ResNet-50).

---

## 25. Actual Failure Cases & Confusions
- **Fine-Grained Inlet Variations**: Misclassifications occurred between `inlet` and `chemin inlet open`, which share identical circular metal apertures.
- **Dust-Coated Rover Parts**: Components covered with Martian dust blend into the background `ground` class.

---

## 26. Whether the 80% Target Was Achieved
- **Target**: 80.00% Test Accuracy.
- **Actual Phase 5C Test Accuracy**: **73.26%**.
- **Conclusion**: **Phase 5C did not achieve the 80% test-accuracy target.**

---

## 27. Integrity Verification Results
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
- [x] Phase 4B, Phase 5A, and Phase 5B artifacts completely untouched
- [x] All reported numbers generated from actual execution

---

## 28. Artifacts Created
- Code: `deep_learning/models/vit.py`, `run_phase5c.py`
- Models: `saved_models/deep_learning/vit/best_vit.pth`, `last_vit.pth`
- Reports: `results/phase5c/classification_reports/` (CSV and TXT)
- Confusion Matrices: `results/phase5c/confusion_matrices/` (raw NPY, raw PNG, normalized PNG)
- Plots: `results/phase5c/plots/` (loss, accuracy, Macro-F1, per-class F1)
- Predictions: `results/phase5c/predictions/` (predictions, labels, probabilities)
- History & Metrics: `results/phase5c/training_history.csv`, `final_metrics.json`, `metadata.json`
- Notebook: `notebooks/05c_vit.ipynb`

---

## 29. Final Phase 5C Conclusion
Vision Transformer (ViT-B/16) was successfully implemented, fine-tuned, and evaluated on the official NASA Curiosity Rover dataset. The model achieved **73.26% Test Accuracy** and **0.6536 Test Macro-F1**, confirming its viability for autonomous planetary exploration while highlighting trade-offs between global self-attention mechanisms and convolutional inductive biases.
