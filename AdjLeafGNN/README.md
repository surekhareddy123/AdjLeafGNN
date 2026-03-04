# AdjLeafGNN (LDDNet + KNN Graph + 2-layer GCN/GAT) — Reference Implementation

This repository implements the pipeline described in your methodology:
- **LDDNet encoder** with **ASPP** + **CSAM** and **GAP** embedding output.
- **KNN graph** construction in embedding space (default **k=5**), undirected + self-loops.
- **2-layer GCN** (default) or optional **2-layer GAT** (lightweight, no PyG dependency).
- **Dual heads**:
  - Disease classification (softmax over C classes)
  - Spread probability (sigmoid; binary)
- **Multi-task loss**: `L = λ1 * CE + λ2 * BCE`, default `λ1=1.0`, `λ2=0.5`.
- **5-fold stratified cross-validation** + metrics.

> NOTE: This is a clean, dependency-light implementation. It **does not require torch-geometric**.

---

## 1) Environment

Recommended (Python 3.10+):

```bash
pip install -r requirements.txt
```

If you use CUDA, install the correct PyTorch wheel from the official website.

---

## 2) Data layout (ImageFolder)

Expected dataset root:

```
DATA_ROOT/
  class_1/
    img001.jpg
    img002.jpg
  class_2/
    ...
```

Example:
- PlantVillage-like folder of leaf images by disease class.

---

## 3) Quick start

### Train with 5-fold CV

```bash
python main.py --mode train --data_root /path/to/DATA_ROOT --out runs/exp1
```

### Evaluate a saved checkpoint

```bash
python main.py --mode eval --data_root /path/to/DATA_ROOT --ckpt runs/exp1/fold0/best.pt
```

### Real-time inference (single image)

```bash
python main.py --mode infer --ckpt runs/exp1/fold0/best.pt --image /path/to/img.jpg
```

---

## 4) Configuration

Edit `configs/default.yaml` or pass overrides via CLI:

- `k_nn`: K in KNN graph builder (default 5)
- `spread_label_mode`: `"neighbor_any"` or `"distance_threshold"`
- `spread_delta`: distance threshold δ (used only if `distance_threshold`)
- `model.gnn_type`: `"gcn"` or `"gat"`

---

## 5) Outputs

Each fold writes to:

```
OUT/
  fold0/
    best.pt
    last.pt
    metrics.json
    logs.txt
  fold1/
  ...
```

---

## 6) Notes on spread labels

Since most image datasets do not include explicit spread labels, this repo supports two **method-aligned** synthetic label strategies:

1. **neighbor_any**: mark a sample as spread-positive if any of its KNN neighbors is from an infected/diseased class.
2. **distance_threshold**: mark as positive if it lies within δ of any infected sample in embedding space.

You can modify `data/spread_labels.py` to incorporate dataset-specific ground truth if available.

---

## License
For research/academic use.
