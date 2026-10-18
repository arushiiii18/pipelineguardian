# Comparative Evaluation Report: Learned Preprocessing Leakage Detector

## 1. Executive Summary & Model Ladder Ranking

We systematically evaluated three model architectures on the binary preprocessing leakage detection task:
1. **Model A**: TF-IDF + Logistic Regression (Text Baseline)
2. **Model B**: Random Forest with Structural AST Features + Compact Tokens (Conventional ML)
3. **Model C**: Plain-PyTorch Graph Isomorphism Network (GIN) on AST Data-Flow Graphs (Neural Model on RTX 2050 GPU)

All hyperparameter selection was conducted strictly via **5-fold GroupKFold cross-validation** across 24 base pipeline designs (448 verified examples). Final generalization was tested once on a **completely held-out synthetic test set of 6 base pipelines** (103 verified examples, zero base pipeline overlap).

---

## 2. Core Results Table

| Model Architecture | Device | Grouped CV Macro-F1 (std) | Held-Out Test Macro-F1 | Test Pos F1 | Test Neg F1 | Confusion Matrix (TN, FP, FN, TP) | Fit Time (s) | Infer Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + Logistic Regression** | CPU | 0.9649 (±0.0458) | 0.9108 | 0.9429 | 0.8788 | [29, 8, 0, 66] | 0.05s | 0.01s |
| **Random Forest** | CPU | **0.9814 (±0.0402)** | **1.0000** | **1.0000** | **1.0000** | **[37, 0, 0, 66]** | 0.52s | 0.04s |
| **Plain-PyTorch GIN** | CUDA (RTX 2050) | 0.8797 (±0.0887) | 0.8505 | 0.8955 | 0.8056 | [29, 8, 6, 60] | 1.95s | 0.08s |

---

## 3. Key Research Insights & Honest Findings

### Did the GNN outperform simpler baselines?
**No.** Random Forest with structural AST features achieved the highest Macro-F1 across both Grouped Cross-Validation (0.9814) and the Held-out Synthetic Test Set (1.0000). The GNN achieved solid performance (Held-out Macro-F1 = 0.8505) and met the Day 6 feasibility gate, but did not beat the tree-based model.

### Why did Random Forest outperform GNN?
1. **Explicit AST Order Primitives**: Preprocessing leakage in code is fundamentally an order-of-operations violation (e.g. `fit` occurring before `train_test_split` or on full dataset objects). The Random Forest directly consumed structural delta features (`line_split - line_first_fit`, `fit_before_split_flag`), which act as sharp, non-linear decision boundaries.
2. **Graph Pooling Information Loss**: The GNN relies on message passing and global graph mean pooling. While GIN retains graph isomorphism characteristics, global pooling slightly dilutes the critical binary relationship between the split call and the transformer fitting node, leading to 8 false positives and 6 false negatives.
3. **Dataset Sample Size vs. Neural Parameterization**: With 448 training examples, Random Forest avoids overfitting while neural graph weights require careful regularization.

---

## 4. Hardware & GPU Training Verification
* **GPU Target**: NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)
* **PyTorch Version**: 2.5.1+cu121 on Python 3.12.7
* **Feasibility Gate**: Verified forward/backward pass, finite gradients, parameter updates, tiny-batch overfit, and checkpoint save/load on CUDA.
* **Peak Memory**: ~16.65 MB allocated on GPU during graph message passing.
* **CPU Execution Notice**: Logistic Regression and Random Forest ran natively on CPU via scikit-learn without misrepresenting CPU computation as GPU-accelerated.

---

## 5. Mutation-Type Generalization (Leave-One-Mutation Stress Test)

In the leave-one-mutation-type-out stress test:
- **`scaler_fit_before_split`**: 100% detection accuracy when held out.
- **`concat_leakage`**: 100% detection accuracy when held out.
- **`imputer_fit_before_split`**: 100% detection accuracy when held out.
- **`transformer_fit_on_test`**: 100% detection accuracy when held out.
- **`clean_variant_reordered` & `clean_pipeline_encapsulation`**: 100% accuracy as negative controls.

The structural features generalize cleanly across mutation templates because they capture the underlying sequence and argument binding properties rather than memorizing template text.

---

## 6. External Real-World Generalization Warning
High synthetic test accuracy (1.0000 on Random Forest, 0.9108 on LogReg) reflects **in-distribution synthetic generalization across unseen base pipeline architectures**. It does **NOT** guarantee identical real-world transfer to unstructured, highly idiosyncratic real-world Kaggle/Yang notebooks. External Yang evaluation remains the essential next validation frontier.
