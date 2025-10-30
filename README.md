
---

# Hybrid CNN–ViT Wrist Abnormality Classifier

This repository provides the official implementation used in our study on **hybrid convolutional–transformer architectures** for automatic wrist X-ray anomaly detection.

The project integrates **Convolutional Neural Networks (CNNs)** and **Vision Transformers (ViTs)** through **Parallel** and **Sequential** fusion strategies to assess **classification performance**, **interpretability**, and **computational efficiency** across both internal (MURA wrist) and external (Al-Huda wrist) datasets.

---

## 1. Overview

This repository enables reproducibility of:

* **Model architectures:** Hybrid CNN–ViT fusion (Parallel / Sequential)
* **Training utilities:** Multi-stage fine-tuning and layer freezing control
* **Custom optimizers:** AdaBoB [1]
* **Evaluation pipeline:** Image- and patient-level metrics, McNemar and Wilcoxon tests
* **Interpretability:** LayerCAM [2] and attention rollout [3]
* **Model loading:** Unified checkpoint loading and reconstruction (`model_loader.py`)

> Patient radiographs are **not included**.
> The codebase is modular and can be applied to any dataset following the same folder and label structure.

---

## 2. Experimental Pipeline

The full workflow mirrors the experimental pipeline presented in the paper and is divided into **four reproducible stages**:

---

### **Stage 1 – Standalone Backbone Pretraining**

**Script:** `scripts/train_stage1_mura_nonwrist.py`

Pre-trains standalone CNN and ViT models on the **MURA non-wrist** subset:

* Example (1): `DeiTClassifier()` – transformer baseline
* Example (2): `DenseNet201Classifier()` – convolutional baseline

Training uses AdamW optimizer, cosine learning rate scheduling, early stopping, and mixed precision (FP16) training where supported.

---

### **Stage 2 – Wrist Fine-tuning**

**Script:** `scripts/train_stage2_mura_wrist.py`

Fine-tunes the Stage 1 backbones on the **MURA wrist subset**:

1. Fine-tunes **Xception** from a Stage 1 checkpoint (freeze ~30%, train last 70%).
2. Fine-tunes **DeiT-B** from a Stage 1 checkpoint (freeze embeddings, train last 60%).

Both examples apply partial freezing using:

```python
m.set_trainable_fraction(model, kind="cnn" or "vit", fraction=0.6)
```

and maintain consistent transforms and loss functions.

---

### **Stage 3 – Hybrid Model Training**

**Script:** `scripts/train_stage3_hybrids.py`

Builds and trains hybrid CNN–ViT models using pre-trained backbones:

* **Xception + DeiT (Parallel Fusion)**
* **DenseNet + ViT (Sequential Fusion)**

Models are constructed using:

```python
from hybrid_vision import CNNExtractor, ViTFeatureLayer, ParallelHybridClassifier, SequentialHybridClassifier
```

and optionally loaded via `model_loader.py` for reproducibility.

All hybrid models follow the same training structure (AdamW / cosine scheduler / early stopping).

---

### **Stage 4 – External Fine-tuning and MTL Evaluation**

**Script:** `scripts/train_stage4_external.py`

Fine-tunes hybrids on the **external Al-Huda wrist dataset** to evaluate domain shift and MTL (multi-task learning) generalization.
Two examples are provided:

* **XC–DeiT (Parallel)** hybrid fine-tuned externally
* **DN–ViT (Parallel)** hybrid fine-tuned externally

This stage also compares **Proposed MTL** vs **General MTL** strategies using patient-level aggregation and statistical tests.

---

## 3. Evaluation and Statistical Testing

Implemented in `evaluation.py`, covering:

* Image-level and patient-level metrics
* Majority voting with mean-probability tie-breaking
* McNemar’s exact test for significance between parallel and sequential hybrids
* Wilcoxon signed-rank test for image–patient paired results
* Inference time and GPU memory profiling

Example usage:

```python
from model_loader import load_hybrid_from_checkpoint
from evaluation import evaluate_and_store_metrics_torch
from data_utils import SingleImageDataset, _val_transforms

# Load pretrained hybrid
model = load_hybrid_from_checkpoint(
    fusion_type="parallel",
    pair_name="xc_deit",
    hybrid_ckpt_path="models/hybrid/par_hybrid_xc_deit_base.pth",
    cnn_ckpt_path="models/wrist_standalones/xception.pth",
    vit_ckpt_path="models/wrist_standalones/deit.pth",
)

# Evaluate
test_ds = SingleImageDataset(x_test, y_test, transform=_val_transforms)
metrics = evaluate_and_store_metrics_torch(model, test_ds, model_name="Xception–DeiT (P)")
```

---

## 4. Repository Structure

```
├── additional_optimizers.py       # Custom optimizers (AdaBoB)
├── data_utils.py                  # Image preprocessing, transforms, dataset loaders
├── hybrid_vision.py               # Hybrid CNN–ViT model definitions
├── interpretability.py            # LayerCAM and attention rollout methods
├── model_loader.py                # Unified model assembly and checkpoint loading
├── evaluation.py                  # Evaluation metrics and statistical analysis
├── scripts/
│   ├── train_stage1_mura_nonwrist.py
│   ├── train_stage2_mura_wrist.py
│   ├── train_stage3_hybrids.py
│   └── train_stage4_external.py
└── README.md
```

All scripts follow the same pattern for reproducibility:

```python
Trainer(model, args, train_dataset, eval_dataset, optimizers=(optimizer, scheduler))
```

---

## 5. Environment

All experiments were run in **Google Colab Pro+** with:

* **GPU:** NVIDIA A100 (40 GB VRAM)
* **RAM:** 83.5 GB
* **Disk:** 235.7 GB

Key dependencies:

| Library      | Version | Purpose            |
| ------------ | ------- | ------------------ |
| PyTorch      | 2.8.0   | Core framework     |
| CUDA         | 12.6    | GPU backend        |
| timm         | 1.0.20  | CNN backbones      |
| Transformers | 4.57.1  | ViT/DeiT           |
| scikit-learn | 1.6.1   | Evaluation metrics |
| NumPy        | 2.0.2   | Array ops          |
| Matplotlib   | 3.10.0  | Visualization      |

Mixed-precision (FP16) training and inference were enabled where available.

---

## 6. Reproducibility Notes

* Each stage logs training and validation metrics via the Hugging Face `Trainer` API.
* All seeds and library versions are fixed for deterministic runs.
* The **patient-level evaluation** follows majority voting with mean probability as a tie-breaker.
* Statistical tests (McNemar’s and Wilcoxon) use image-level predictions to ensure sufficient paired counts.

---

## 7. References

[1] Xiang, Q.; Wang, X.; Song, Y.; Lei, L.
*Dynamic Bound Adaptive Gradient Methods with Belief in Observed Gradients.*
**Pattern Recognition**, 168 (2025) 111819.
(Implemented in `additional_optimizers.py`.)

[2] Jiang, X.; Zhang, Y.; Liu, S.; et al.
*LayerCAM: Exploring Hierarchical Class Activation Map for Localization.*
**IEEE Transactions on Image Processing**, 30 (2021) 5875–5888.
(Used in `interpretability.py`.)

[3] Abnar, S.; Zuidema, W.
*Quantifying Attention Flow in Transformers.*
**ACL**, 2020.
(Attention rollout implementation.)

[4] Xiang, Q.; Wang, X.; Lai, J.; Lei, L.; Song, Y.; He, J.; Li, R.
*Quadruplet Depth-Wise Separable Fusion CNN for Ballistic Target Recognition with Limited Samples.*
**Expert Systems with Applications**, 235 (2024) 121182.

[5] Hu, J.; Shen, L.; Sun, G.
*Squeeze-and-Excitation Networks.*
**CVPR**, 2018, pp. 7132–7141.
[https://doi.org/10.1109/CVPR.2018.00745](https://doi.org/10.1109/CVPR.2018.00745)

---

