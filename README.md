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

Expected structure:

```
dataset/
│
├── apple_scab/
│   ├── img1.jpg
│   ├── img2.jpg
│
├── apple_black_rot/
│   ├── img3.jpg
│
├── healthy/
│   ├── img4.jpg
```

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

To reproduce experiments:

```
python main.py --mode train --cfg configs/default.yaml
```

The framework ensures reproducibility using:

* Fixed random seeds
* Deterministic training
* Stratified cross-validation
* Configuration files

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

```
if you are using my paper side this title 
@article{AdjLeafGNN2026},
title={AdjLeafGNN: A Graph Neural Network Framework for Leaf Disease Detection and Infection Spread Prediction},

