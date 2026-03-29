"""Dynamic MetaGate: per-timestep MLP-based selector among 4 experts.

This module implements Mode B: at each timestep the MLP meta-gate:
  1. Runs all 4 expert selectors (BN, TopK, Sens, GNN)
  2. Extracts features from TM statistics + expert outputs
  3. A trained MLP classifier predicts which expert yields lowest MLU
  4. Runs ONLY the predicted expert's OD selection through LP

The classifier is trained on oracle labels from known topologies only.
Unseen topologies (Germany50, VtlWavenet2011) are evaluated using the
unified gate trained on the 6 known topologies — zero training on unseen.

Architecture:
  Input features (~46 dim per timestep):
    - TM statistics: mean, std, max, skew, entropy, top-10 share (8)
    - Per-expert demand stats: mean, std, max, coverage for each of 4 experts (16)
    - Cross-expert overlaps: 6 pairwise + 4 unique + 1 all-agree (11)
    - Topology features: log_nodes, log_edges, density (3)
    - GNN diagnostics: alpha, confidence, correction_mean (3)
  Output: 4-class softmax {0=Bottleneck, 1=TopK, 2=Sensitivity, 3=GNN}
  Model: 2-layer MLP with dropout
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SELECTOR_NAMES = ["bottleneck", "topk", "sensitivity", "gnn"]
NUM_SELECTORS = 4


@dataclass
class MetaGateConfig:
    hidden_dim: int = 64
    dropout: float = 0.2
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 32


@dataclass
class MetaGateDecision:
    """Record of one meta-gate decision."""
    timestep: int
    predicted_selector: str
    confidence: float
    oracle_selector: str
    predicted_mlu: float
    oracle_mlu: float


def extract_features(
    tm_vector: np.ndarray,
    selector_results: Dict[str, List[int]],
    num_nodes: int,
    num_edges: int,
    k_crit: int,
    gnn_info: Optional[Dict] = None,
    ecmp_link_utils: Optional[np.ndarray] = None,
    capacities: Optional[np.ndarray] = None,
    path_library=None,
) -> np.ndarray:
    """Extract feature vector for the MLP meta-gate classifier.

    Features (49 dim):
      TM stats (8): mean, std, max, min_nonzero, skew, kurtosis, entropy, top10_share
      Per-expert (4x4=16): mean_demand, std_demand, max_demand, coverage
      Cross-expert (11): 6 pairwise overlaps + 4 unique counts + 1 all-agree
      Topology (3): log_nodes, log_edges, density
      GNN diagnostics (5): alpha, confidence, correction_mean, w_bottleneck, w_sensitivity
      Demand concentration (4): per-expert selected-demand share of total
      ECMP baseline (2): ecmp_max_util, ecmp_mean_util
    Total: 49 dims
    """
    tm = np.maximum(tm_vector, 0.0)
    total = float(np.sum(tm))
    nonzero = tm[tm > 0]

    # --- TM statistics (8) ---
    tm_mean = float(np.mean(tm)) if tm.size else 0.0
    tm_std = float(np.std(tm)) if tm.size else 0.0
    tm_max = float(np.max(tm)) if tm.size else 0.0
    tm_min_nz = float(np.min(nonzero)) if nonzero.size else 0.0
    if nonzero.size > 2:
        from scipy.stats import skew, kurtosis
        tm_skew = float(skew(nonzero))
        tm_kurt = float(kurtosis(nonzero))
    else:
        tm_skew = 0.0
        tm_kurt = 0.0
    if total > 0 and nonzero.size > 0:
        probs = nonzero / total
        tm_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        tm_entropy = 0.0
    sorted_tm = np.sort(tm)[::-1]
    top10_share = float(np.sum(sorted_tm[:10]) / max(total, 1e-12))

    tm_feats = [tm_mean, tm_std, tm_max, tm_min_nz, tm_skew, tm_kurt, tm_entropy, top10_share]

    # --- Per-expert features (4x4 = 16) ---
    selector_feats = []
    sets = {}
    for name in SELECTOR_NAMES:
        selected = selector_results.get(name, [])
        s = set(selected)
        sets[name] = s
        if selected:
            demands = tm[selected]
            selector_feats.extend([
                float(np.mean(demands)),
                float(np.std(demands)),
                float(np.max(demands)),
                len(selected) / max(len(tm), 1),  # coverage
            ])
        else:
            selector_feats.extend([0.0, 0.0, 0.0, 0.0])

    # --- Cross-expert features (11) ---
    bn_set = sets.get("bottleneck", set())
    topk_set = sets.get("topk", set())
    sens_set = sets.get("sensitivity", set())
    gnn_set = sets.get("gnn", set())

    def overlap(a, b):
        if not a or not b:
            return 0.0
        return len(a & b) / max(len(a | b), 1)

    cross_feats = [
        overlap(bn_set, topk_set),
        overlap(bn_set, sens_set),
        overlap(bn_set, gnn_set),
        overlap(topk_set, sens_set),
        overlap(topk_set, gnn_set),
        overlap(sens_set, gnn_set),
        len(bn_set - topk_set - sens_set - gnn_set) / max(k_crit, 1),
        len(topk_set - bn_set - sens_set - gnn_set) / max(k_crit, 1),
        len(sens_set - bn_set - topk_set - gnn_set) / max(k_crit, 1),
        len(gnn_set - bn_set - topk_set - sens_set) / max(k_crit, 1),
        len(bn_set & topk_set & sens_set & gnn_set) / max(k_crit, 1),
    ]

    # --- Topology features (3) ---
    density = (2 * num_edges) / max(num_nodes * (num_nodes - 1), 1)
    topo_feats = [np.log1p(num_nodes), np.log1p(num_edges), density]

    # --- GNN diagnostics (5) ---
    if gnn_info is not None:
        gnn_diag = [
            float(gnn_info.get("alpha", 0.0)),
            float(gnn_info.get("confidence", 0.0)),
            float(gnn_info.get("gnn_correction_mean", 0.0)),
            float(gnn_info.get("w_bottleneck", 0.5)),
            float(gnn_info.get("w_sensitivity", 0.5)),
        ]
    else:
        gnn_diag = [0.0, 0.0, 0.0, 0.5, 0.5]

    # --- Demand concentration per expert (4) ---
    # What fraction of total demand does each expert's selection capture?
    demand_shares = []
    for name in SELECTOR_NAMES:
        selected = selector_results.get(name, [])
        if selected and total > 0:
            demand_shares.append(float(np.sum(tm[selected])) / total)
        else:
            demand_shares.append(0.0)

    # --- ECMP baseline utilization (2) ---
    if ecmp_link_utils is not None:
        ecmp_feats = [float(np.max(ecmp_link_utils)), float(np.mean(ecmp_link_utils))]
    else:
        ecmp_feats = [0.0, 0.0]

    all_feats = (tm_feats + selector_feats + cross_feats + topo_feats +
                 gnn_diag + demand_shares + ecmp_feats)
    return np.array(all_feats, dtype=np.float32)


class DynamicMetaGate:
    """Per-timestep MLP meta-gate that selects among 4 experts.

    Experts: {Bottleneck, TopK, Sensitivity, GNN}
    Trained on oracle labels from known topologies only.
    Applied to both known and unseen topologies at test time.
    """

    def __init__(self, config: MetaGateConfig = None):
        self.config = config or MetaGateConfig()
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self._is_trained = False

    def train(self, features: np.ndarray, labels: np.ndarray, val_features=None, val_labels=None):
        """Train the meta-gate classifier.

        Args:
            features: [N, feat_dim] feature matrix
            labels: [N] integer labels {0=BN, 1=TopK, 2=Sens, 3=GNN}

        Uses inverse-frequency class weights to prevent majority-class collapse.
        """
        import torch
        import torch.nn as nn

        self.scaler_mean = features.mean(axis=0)
        self.scaler_std = features.std(axis=0) + 1e-8
        X = (features - self.scaler_mean) / self.scaler_std
        y = labels.astype(np.int64)

        feat_dim = X.shape[1]
        h = self.config.hidden_dim

        self.model = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, NUM_SELECTORS),
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                     weight_decay=1e-4)

        # Compute inverse-frequency class weights to handle imbalanced oracle labels
        class_counts = np.bincount(y, minlength=NUM_SELECTORS).astype(np.float64)
        class_counts = np.maximum(class_counts, 1.0)
        class_weights = len(y) / (NUM_SELECTORS * class_counts)
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        best_val_acc = 0.0
        best_state = None

        for epoch in range(self.config.num_epochs):
            self.model.train()
            perm = torch.randperm(len(X_t))
            total_loss = 0.0
            for i in range(0, len(X_t), self.config.batch_size):
                idx = perm[i:i + self.config.batch_size]
                logits = self.model(X_t[idx])
                loss = criterion(logits, y_t[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if val_features is not None and (epoch + 1) % 10 == 0:
                val_acc = self._eval_accuracy(val_features, val_labels)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self._is_trained = True

        train_acc = self._eval_accuracy(features, labels)
        logger.info(f"MetaGate trained: train_acc={train_acc:.3f}, val_acc={best_val_acc:.3f}")
        return train_acc, best_val_acc

    def _eval_accuracy(self, features, labels):
        import torch
        X = (features - self.scaler_mean) / self.scaler_std
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t)
            preds = logits.argmax(dim=1).numpy()
        return float((preds == labels).mean())

    def calibrate(self, expert_win_counts: np.ndarray, smoothing: float = 1.0,
                  strength: float = 3.0):
        """Set a calibration prior from a small validation run on the target topology.

        Args:
            expert_win_counts: array of how many times each expert won on calibration set.
            smoothing: Laplace smoothing to avoid zero priors.
            strength: exponent applied to prior before fusion. Higher = trust calibration more.
                      At strength=1, standard Bayesian fusion. At strength=3, calibration
                      dominates over MLP when prior is strongly skewed.

        Combined with MLP via: final_prob[i] = MLP_prob[i] * prior[i]^strength / Z
        """
        counts = np.array(expert_win_counts, dtype=np.float64) + smoothing
        self._calibration_prior = counts / counts.sum()
        self._calibration_strength = strength
        logger.info(f"MetaGate calibrated: prior={self._calibration_prior}, strength={strength}")

    def clear_calibration(self):
        """Remove calibration prior."""
        self._calibration_prior = None

    def predict(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict which expert to use, with optional calibration fusion.

        If calibration prior is set: final_prob[i] = MLP_prob[i] * prior[i] / Z

        Returns: (predicted_class, probabilities)
        """
        import torch

        if not self._is_trained:
            return 0, np.array([1.0, 0.0, 0.0, 0.0])

        X = (features - self.scaler_mean) / self.scaler_std
        X_t = torch.tensor(X.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).numpy()[0]

        if hasattr(self, '_calibration_prior') and self._calibration_prior is not None:
            alpha = getattr(self, '_calibration_strength', 1.0)
            boosted_prior = self._calibration_prior ** alpha
            fused = probs * boosted_prior
            fused = fused / (fused.sum() + 1e-12)
            return int(np.argmax(fused)), fused

        return int(np.argmax(probs)), probs

    def save(self, path: Path):
        """Save trained model."""
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "config": {
                "hidden_dim": self.config.hidden_dim,
                "dropout": self.config.dropout,
            },
            "selector_names": SELECTOR_NAMES,
            "num_selectors": NUM_SELECTORS,
        }, str(path))
        logger.info(f"MetaGate saved to {path}")

    def load(self, path: Path, feat_dim: int):
        """Load trained model."""
        import torch
        import torch.nn as nn

        payload = torch.load(str(path), map_location="cpu")
        self.scaler_mean = payload["scaler_mean"]
        self.scaler_std = payload["scaler_std"]

        h = payload["config"]["hidden_dim"]
        n_sel = payload.get("num_selectors", NUM_SELECTORS)
        dr = payload["config"]["dropout"]
        self.model = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, n_sel),
        )
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self._is_trained = True
        logger.info(f"MetaGate loaded from {path}")
