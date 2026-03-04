from __future__ import annotations
import numpy as np

def infer_infected_mask(y_cls: np.ndarray, classes: list[str], policy: str = "non_healthy") -> np.ndarray:
    """Return boolean mask of infected samples.

    policy:
      - non_healthy: any class name not containing 'healthy' is infected
      - all: treat all classes as infected (not typical; mostly for debugging)
    """
    if policy == "all":
        return np.ones_like(y_cls, dtype=bool)

    # heuristic for PlantVillage-like names
    infected = np.ones_like(y_cls, dtype=bool)
    for idx, name in enumerate(classes):
        if "healthy" in name.lower():
            infected[y_cls == idx] = False
    return infected

def spread_labels_neighbor_any(adj: np.ndarray, infected_mask: np.ndarray) -> np.ndarray:
    """y_spread(i)=1 if any neighbor j is infected."""
    # adj is 0/1 with self-loops possibly
    neigh_infected = (adj @ infected_mask.astype(np.int32)) > 0
    return neigh_infected.astype(np.int64)

def spread_labels_distance_threshold(dist_mat: np.ndarray, infected_mask: np.ndarray, delta: float) -> np.ndarray:
    """y_spread(i)=1 if exists infected j with d(i,j) <= delta."""
    # ignore self distance by requiring j!=i using large diag; caller can handle too
    infected_idx = np.where(infected_mask)[0]
    if infected_idx.size == 0:
        return np.zeros(dist_mat.shape[0], dtype=np.int64)
    near = (dist_mat[:, infected_idx] <= float(delta)).any(axis=1)
    return near.astype(np.int64)
