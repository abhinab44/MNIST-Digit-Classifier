# MNIST Handwritten Digit Classification — MLP vs CNN vs CNN+Augmentation

> A progressive deep learning study on MNIST — benchmarking a fully-connected MLP against a CNN and an augmented CNN, with full training curves, confusion matrices, and error analysis on 70,000 handwritten digit images.

---

## GitHub Repository Description

> Deep learning pipeline on MNIST — MLP baseline vs CNN (Conv2D + BatchNorm + MaxPooling) vs CNN+Augmentation (rotation, shift, zoom). Includes training curves, confusion matrix, per-class classification report, model comparison, and error analysis of misclassified digit pairs.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Steps](#pipeline-steps)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Key Findings](#key-findings)

---

## Overview

This project progressively builds three neural network architectures to classify handwritten digits (0–9) from the MNIST dataset. Starting from a fully-connected MLP baseline, it advances through a convolutional network and finally a data-augmented CNN — demonstrating how spatial feature extraction and augmentation-driven regularization each contribute to better generalization.

---

## Dataset

| Property | Details |
|---|---|
| **Source** | `tf.keras.datasets.mnist` (originally from NIST) |
| **Training set** | 60,000 grayscale images |
| **Test set** | 10,000 grayscale images |
| **Image shape** | 28 × 28 pixels, single channel |
| **Pixel range** | 0–255 (normalized to 0–1) |
| **Classes** | 10 (digits 0–9) |
| **Label encoding** | One-hot encoded |

>  No manual download needed — the dataset is fetched automatically via Keras.

---

## Pipeline Steps

```
1. Data Loading & Exploration
        ↓
2. Preprocessing       (normalization to [0,1], one-hot encoding, reshape for CNN)
        ↓
3. Model 1 — MLP       (Flatten → Dense → Dropout, baseline)
        ↓
4. Model 2 — CNN       (Conv2D + BatchNorm + MaxPooling + Dropout)
        ↓
5. Model 3 — CNN+Aug   (same CNN + ImageDataGenerator: rotation, shift, zoom)
        ↓
6. Model Comparison    (accuracy, loss, error count)
        ↓
7. Error Analysis      (confusion matrix, most confused digit pairs, samples)
```

---

## Model Architectures

### Model 1 — MLP (Baseline)

| Layer | Output Shape | Params |
|---|---|---|
| Flatten | (None, 784) | 0 |
| Dense (ReLU) | (None, 256) | 200,960 |
| Dropout (0.2) | (None, 256) | 0 |
| Dense (ReLU) | (None, 128) | 32,896 |
| Dropout (0.2) | (None, 128) | 0 |
| Dense (Softmax) | (None, 10) | 1,290 |

**Total params:** 235,146 · **Optimizer:** Adam · **Loss:** Categorical Crossentropy · **Epochs:** 10

---

### Model 2 — CNN

| Block | Layers |
|---|---|
| Block 1 | Conv2D(32, 3×3, same) → BN → Conv2D(32, 3×3, same) → BN → MaxPool(2×2) → Dropout(0.25) |
| Block 2 | Conv2D(64, 3×3, same) → BN → Conv2D(64, 3×3, same) → BN → MaxPool(2×2) → Dropout(0.25) |
| Head | Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(10, Softmax) |

**Total params:** 871,402 · **Optimizer:** Adam · **Epochs:** 10

---

### Model 3 — CNN + Data Augmentation

Same architecture as CNN, trained with `ImageDataGenerator`:

```python
ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)
```

**Epochs:** 15 (extra epochs to compensate for harder augmented training distribution)

---

## 📈 Results

| Model | Test Accuracy | Test Loss | Errors (out of 10k) |
|---|---|---|---|
| MLP (baseline) | 97.95% | 0.0784 | 205 |
| CNN | 98.89% | 0.0396 | 110 |
| **CNN + Augmentation** | **99.16%** | **0.0292** | **84** |

**Best model:** CNN + Augmentation (99.16% test accuracy)

### Top Confused Digit Pairs (CNN+Aug)

| True → Predicted | Count |
|---|---|
| 9 → 4 | 11 |
| 5 → 3 | 6 |
| 1 → 8 | 5 |
| 6 → 1 | 5 |
| 9 → 7 | 5 |

Most errors involve visually similar digit pairs — ambiguities that are challenging even for humans in poor handwriting.

---

## Project Structure

```
mnist-digit-classifier/
│
├── mnist_digit_classifier.ipynb   # Main notebook (full pipeline)
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

> No `data/` folder needed — MNIST is downloaded automatically by Keras on first run.

---

## Requirements

```
tensorflow>=2.10
numpy
matplotlib
scikit-learn
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

---

## How to Run

**Option 1 — Jupyter Notebook (local):**

```bash
git clone https://github.com/abhinab44/mnist-digit-classifier.git
cd mnist-digit-classifier
jupyter notebook mnist_digit_classifier.ipynb
```

The MNIST dataset (~11 MB) is downloaded automatically on first run — no setup required.

**Option 2 — Google Colab:**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> CNN training takes ~3–4 minutes per model on Colab GPU. Enable GPU via **Runtime → Change runtime type → T4 GPU** for faster training.

---

## Key Findings

- **Spatial features matter**: The jump from MLP (97.95%) to CNN (98.89%) confirms that local pattern detection via convolution — detecting edges, curves, and corners — is far more effective than treating every pixel independently.
- **Augmentation reduces overfitting**: CNN+Aug shows a smaller train–val accuracy gap compared to the vanilla CNN, while achieving the best test accuracy (99.16%) — confirming that training on harder augmented samples improves generalization.
- **Errors are genuinely hard**: The remaining 84 misclassifications by CNN+Aug are concentrated in visually ambiguous pairs (9↔4, 5↔3, 7↔2) — cases that even humans would find difficult with messy handwriting.
- **Diminishing returns on MNIST**: The gain from CNN → CNN+Aug (~0.27%) is much smaller than MLP → CNN (~0.94%), suggesting the dataset is nearly saturated and further gains require deeper architectures or stronger regularization.
- **BatchNormalization stabilizes training**: The CNN converges quickly — val_accuracy reaches ~98.9% by epoch 2, reflecting the stabilizing effect of BatchNorm after each Conv2D layer.

---

## Concepts Demonstrated

- Pixel normalization and one-hot label encoding
- Fully-connected MLP as an image classification baseline
- Convolutional blocks with `Conv2D`, `BatchNormalization`, `MaxPooling2D`, `Dropout`
- `ImageDataGenerator` for on-the-fly data augmentation (rotation, shift, zoom)
- Training curve visualization (loss & accuracy per epoch)
- Confusion matrix and per-class classification report (`sklearn`)
- Error analysis of the most confused digit pairs with prediction confidence scores

---

## License

This project is open-source under the [MIT License](LICENSE).

---

*Built with Python 3.10 · TensorFlow 2.19.0 · Keras · NumPy · Matplotlib · scikit-learn*
