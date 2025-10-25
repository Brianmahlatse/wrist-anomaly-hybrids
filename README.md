# Hybrid CNN + ViT Wrist Abnormality Classifier

This repository contains the implementation used in our study on hybrid convolutional transformer architectures for automatic wrist X-ray anomaly detection.

The project integrates CNNs (Xception, DenseNet201) with Vision Transformers (ViT-B/16, DeiT-B) in Parallel and Sequential fusion configurations to evaluate classification performance, interpretability, and computational efficiency.

## Overview

This repository enables reproducibility of:
- Model architectures (CNN–ViT hybrids)
- Training utilities and partial freezing functions
- Custom optimizer (AdaBoB [1])
- Inference and evaluation routines
- Interpretability visualizations, including LayerCAM [2] and attention rollout [3]

Patient X-ray data is not included. The codebase is modular for external use on new datasets.

## Environment

All experiments were performed in Google Colab Pro+ using an NVIDIA A100 GPU (40 GB VRAM), 83.5 GB RAM, and 235.7 GB disk space.  
The implementation was developed in PyTorch v2.0 (CUDA 11.8) using Transformers (v4.42.4), timm (v0.9.16), NumPy, and scikit-learn.  
Mixed-precision (FP16) training was applied when GPU support was available.

## References

[1] Q. Xiang, X. Wang, Y. Song, L. Lei. Dynamic Bound Adaptive Gradient Methods with Belief in Observed Gradients. Pattern Recognition, 168, 111819, 2025.  
(AdaBoB optimizer implementation used in `additional_optimizers.py`.)

[2] X. Jiang, Y. Zhang, S. Liu, et al. LayerCAM: Exploring Hierarchical Class Activation Map for Localization. IEEE Transactions on Image Processing, 30:5875–5888, 2021.  
(LayerCAM-based interpretability used in `interpretability.py`.)

[3] S. Abnar and W. Zuidema. Quantifying Attention Flow in Transformers. ACL, 2020.  
(Attention rollout implementation adapted for ViT/DeiT in `interpretability.py`.)
