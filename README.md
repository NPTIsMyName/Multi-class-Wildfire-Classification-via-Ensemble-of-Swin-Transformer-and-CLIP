---
# Wildfire Multi-class Classification with CLIP & Swin Transformer

![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Vision Transformer](https://img.shields.io/badge/Vision-Transformers-green)
![Status](https://img.shields.io/badge/status-research-orange)

Deep learning system for **multi-class wildfire image classification** using **Vision Transformers and ensemble learning**.

The project fine-tunes two powerful pretrained models:

* **CLIP (ViT-L/14@336px)**
* **Swin Transformer V2-B**

Predictions are combined using a **weighted ensemble strategy** to improve classification accuracy.

---

# Project Highlights

* Fine-tuned **large-scale vision transformer models**
* Applied **data augmentation and training stabilization techniques**
* Implemented **ensemble learning with weighted soft voting**
* Performed **grid search to find optimal ensemble weights**
* Evaluated with **accuracy, classification report, and confusion matrix**

---

# Pipeline Overview

```
Dataset
   │
   ▼
Image Preprocessing + Data Augmentation
   │
   ▼
Fine-tuning Models
   ├── CLIP (ViT-L/14)
   └── Swin Transformer V2-B
   │
   ▼
Model Predictions
   │
   ▼
Weighted Ensemble (Soft Voting)
   │
   ▼
Final Multi-class Prediction
```

---

# Dataset

The wildfire dataset contains **5 classes** related to fire and smoke conditions.

Dataset link:
https://www.kaggle.com/datasets/elmadafri/the-wildfire-dataset

| ID | Class Name                         |
| -- | ---------------------------------- |
| 0  | Both Smoke & Fire                  |
| 1  | Smoke from Fires                   |
| 2  | Fire Confounding Elements          |
| 3  | Forested Areas without Confounding |
| 4  | Smoke Confounding Elements         |

Dataset structure:

```
the_wildfire_dataset/

train/
val/
test/

fire/
nofire/

subclasses/
```

---

# Models

## CLIP Vision Transformer

* Model: **ViT-L/14@336px**
* Pretrained on **image-text pairs**
* Fine-tuned for wildfire classification
* Input size: **336 × 336**

Architecture:

```
CLIP Visual Encoder
        │
        ▼
Fully Connected Layer
        │
        ▼
Dropout
        │
        ▼
5-class classifier
```

---

## Swin Transformer V2-B

* Hierarchical vision transformer
* Pretrained on **ImageNet**
* Fine-tuned for wildfire classification
* Input size: **384 × 384**

Architecture:

```
Swin Transformer Backbone
        │
        ▼
Fully Connected Layer
        │
        ▼
Dropout
        │
        ▼
5-class classifier
```

---

# Training Strategy

### Optimizer

```
AdamW
```

### Learning Rate

| Model | Learning Rate |
| ----- | ------------- |
| CLIP  | 3e-6          |
| Swin  | 5e-6          |

### Training Techniques

* Data Augmentation
* Early Stopping
* Learning Rate Scheduler
* Dropout Regularization

### Data Augmentation

```
RandomAffine
ColorJitter
RandomHorizontalFlip
Resize
Normalization
```

---

# Ensemble Method

After training both models, predictions are combined using **weighted soft voting**.

Formula:

```
P_final = w1 * P_swin + w2 * P_clip
```

Where:

* `P_swin` = Swin Transformer probabilities
* `P_clip` = CLIP probabilities
* `w1 + w2 = 1`

---

# Ensemble Weight Optimization

A **grid search** is used to find optimal ensemble weights.

Example:

```
w_swin = 0.65
w_clip = 0.35
```

This improves prediction robustness by combining complementary features learned by both models.

---

# Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

Example outputs:

* Confusion matrix visualization
* Training loss curves
* Ensemble weight vs accuracy plot

---

# Kaggle Implementation

Due to dataset size and training requirements, **all experiments were implemented on Kaggle**.

Kaggle notebooks include:

* Model fine-tuning
* Training pipeline
* Ensemble inference
* Evaluation and visualization

Kaggle resources:

* CLIP fine-tuning notebook
* Swin Transformer fine-tuning notebook
* Ensemble inference notebook
---

# Technologies

* PyTorch
* Vision Transformers
* CLIP
* Swin Transformer
* Scikit-learn
* Matplotlib
* Seaborn

---

# Future Work

* Add **ConvNeXt / ViT ensembles**
* Explore **knowledge distillation**
* Build **real-time wildfire detection system**
* Deploy model for monitoring systems
