 

```
[![DOI](https://zenodo.org/badge/1172485554.svg)](https://doi.org/10.5281/zenodo.18862439)
```
Code Archive
GitHub repository:
https://github.com/surekhareddy123/AdjLeafGNN

Permanent archived release (Zenodo DOI):
https://doi.org/10.5281/zenodo.18862439


  


# AdjLeafGNN: Graph Neural Network Framework for Leaf Disease Detection and Spread Prediction

## Overview

AdjLeafGNN is a deep learning and graph neural network framework designed for plant leaf disease detection and infection spread prediction. The framework integrates deep convolutional feature extraction with graph-based relational learning, enabling the model to capture both visual disease characteristics and contextual relationships between leaves.

Traditional leaf disease classification models treat each leaf independently. However, in real agricultural scenarios, disease propagation occurs across neighboring leaves and plants. AdjLeafGNN models these dependencies by constructing a **K-Nearest Neighbor (KNN) graph** in embedding space and applying **Graph Neural Networks (GNNs)** to reason about infection propagation patterns.

The framework consists of:

* LDDNet Feature Encoder
* ASPP (Atrous Spatial Pyramid Pooling)
* CSAM (Channel–Spatial Attention Module)
* KNN Graph Construction
* Graph Neural Network (GCN or GAT)
* Dual Prediction Heads

  * Disease classification
  * Infection spread probability

The model is trained using a **multi-task learning strategy** combining classification loss and spread prediction loss.

---

# Key Features

* Hybrid CNN + Graph Neural Network architecture
* Attention-enhanced feature extraction using CSAM
* Multi-scale contextual feature modeling using ASPP
* Graph-based relational reasoning among leaf samples
* Multi-task learning framework
* Flexible GNN backend (GCN or GAT)
* Stratified 5-fold cross-validation
* Spread risk estimation for disease propagation
* Lightweight implementation without PyTorch Geometric dependency

---

# System Architecture

## 1. Image Feature Extraction (LDDNet)

Input leaf images are processed using a convolutional encoder that extracts hierarchical visual features.

Key components:

* Convolutional backbone
* ASPP for multi-scale feature extraction
* CSAM attention for spatial and channel refinement
* Global Average Pooling (GAP)

Output:

Embedding vector per image (node feature)

---

## 2. Graph Construction

Leaf embeddings are used to construct a **K-Nearest Neighbor graph**.

Steps:

1. Compute pairwise Euclidean distances
2. Identify K nearest neighbors
3. Construct adjacency matrix
4. Add self-loops

Default:

```
k = 5
```

---

## 3. Graph Neural Network Reasoning

Two supported models.

### GCN

```
H1 = ReLU(A_hat X W1)
H2 = A_hat H1 W2
```

### GAT

```
alpha_ij = softmax(a^T [Wh_i || Wh_j])
h_i = sum(alpha_ij Wh_j)
```

---

## 4. Dual Task Prediction

### Disease Classification

Softmax classifier.

### Spread Prediction

Sigmoid classifier predicting infection probability.

---

## 5. Multi-Task Loss

```
L = λ1 * L_classification + λ2 * L_spread
```

Default weights:

```
λ1 = 1.0
λ2 = 0.5
```

---

# Repository Structure

```
AdjLeafGNN/
│
├── configs/
│   └── default.yaml
│
├── data/
│   ├── datasets.py
│   ├── transforms.py
│   ├── splits.py
│   └── spread_labels.py
│
├── models/
│   ├── lddnet/
│   │   ├── backbone.py
│   │   ├── aspp.py
│   │   ├── csam.py
│   │   └── encoder.py
│   │
│   ├── gnn/
│   │   ├── graph_builder.py
│   │   └── layers.py
│   │
│   └── adjleafgnn.py
│
├── losses/
│   └── multitask_loss.py
│
├── train/
│   ├── trainer.py
│   └── crossval.py
│
├── inference/
│   └── realtime.py
│
├── utils/
│   ├── seed.py
│   ├── device.py
│   ├── metrics.py
│   └── logging.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone repository

```
git clone https://github.com/username/AdjLeafGNN.git
cd AdjLeafGNN
```

Install dependencies

```
pip install -r requirements.txt
```

Dependencies:

* Python 3.9+
* PyTorch
* Torchvision
* NumPy
* Scikit-learn
* PyYAML
* tqdm

---

# Dataset Preparation

This study uses the **PlantVillage dataset** for plant leaf disease classification and spread prediction experiments.

**Dataset source used in this work:**

Kaggle PlantVillage distribution:
https://www.kaggle.com/datasets/mohitsingh1804/plantvillage

**Canonical dataset citation:**

Hughes, D.P. and Salathé, M., 2015.
An open access repository of images on plant health to enable the development of mobile disease diagnostics.
*arXiv preprint arXiv:1511.08060.*

**Version / access information:**

The dataset was downloaded from the above Kaggle source and accessed in **2026** for this study.

**Dataset contents used:**

The dataset consists of class-labeled RGB images of healthy and diseased plant leaves. In this project, images are organized into class-specific folders and loaded using the standard **ImageFolder format in PyTorch**.

**Directory structure used by the code:**

```
dataset/
├── class_1/
│   ├── img001.jpg
│   ├── img002.jpg
├── class_2/
│   ├── img003.jpg
│   ├── img004.jpg
├── healthy/
│   ├── img005.jpg
```

**Experimental split protocol:**

The study does not use a fixed train/test folder split. Instead, the full dataset is organized in class folders and evaluated using **stratified 5-fold cross-validation**, ensuring balanced class distribution across training and validation folds.

**Preprocessing and loading:**

* Images are resized to the configured input resolution.
* Standard normalization is applied.
* Data augmentation is applied only to training folds.
* The dataset is loaded through the **PyTorch ImageFolder pipeline**.

**License / usage note:**

Users should consult the original PlantVillage project page and the Kaggle dataset page for the applicable dataset terms, licensing, and usage conditions before reuse.

---

# Training

```
python main.py \
--mode train \
--data_root path/to/dataset \
--out runs/experiment1
```

Pipeline:

1. Dataset loading
2. Stratified folds
3. Graph construction
4. GNN training
5. Validation evaluation
6. Model checkpointing

---

# Evaluation

```
python main.py \
--mode eval \
--data_root path/to/dataset \
--ckpt runs/experiment1/fold0/best.pt
```

Metrics:

Disease classification

* Accuracy
* Precision
* Recall
* F1

Spread prediction

* AUC-ROC
* Precision
* Recall
* F1
* MCC

---

# Inference

```
python main.py \
--mode infer \
--ckpt runs/experiment1/fold0/best.pt \
--image test_leaf.jpg \
--data_root path/to/dataset
```

Example output

```
Predicted Class: Apple Scab
Spread Probability: 0.73
```

---

# Configuration

```
configs/default.yaml
```

Example parameters:

```
img_size: 224
batch_size: 32
epochs: 150
learning_rate: 1e-4
k_nn: 5
gnn_type: gcn
dropout: 0.3
```

---

# Experimental Protocol

Cross-validation:

```
5-fold stratified cross validation
```

Training setup:

```
Optimizer: Adam
Learning rate: 1e-4
Batch size: 32
Early stopping: 15 epochs
```

---

# Results Output

```
runs/
   fold0/
      best.pt
      last.pt
      metrics.json
      logs.txt
   fold1/
   fold2/
```

Summary file:

```
crossval_summary.json
```

---

# Reproducibility

To reproduce the experiments:
1.	Download the PlantVillage dataset from the Kaggle source listed above.
2.	Organize the images into class-specific folders as shown above.
3.	Update the dataset path in the training command.
4.	Run the code with the provided configuration file.
5.	The reported results are based on stratified 5-fold cross-validation, not on a single random split.

---

# Research Applications

* Smart agriculture disease monitoring
* Precision crop protection
* Plant pathology research
* Agricultural IoT systems
* Disease spread prediction in crop fields

---

# Future Extensions

* Temporal disease progression modeling
* Field-level spatial graph modeling
* Drone imagery integration
* Federated agricultural disease monitoring
* Real-time mobile deployment

---


# Citation

If you are using this repository or the corresponding research work, please cite the following paper:

```
@article{AdjLeafGNN2026,
title={AdjLeafGNN: A Graph Neural Network Framework for Leaf Disease Detection and Infection Spread Prediction}
}
```

---

# License

This project is licensed under the MIT License.

See the LICENSE file for details.


