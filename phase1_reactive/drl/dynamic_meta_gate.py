"""Dynamic MetaGate: per-timestep selector among {Bottleneck, TopK, Sensitivity}.

This module implements the TRUE Mode B meta-selector. At each timestep:
  1. Compute selector scores for all 3 heuristics (fast, no LP)
  2. Extract features from the score distributions + TM statistics
  3. A trained classifier predicts which selector will yield lowest MLU
  4. Run ONLY the predicted selector → LP → routing

The classifier is trained on oracle labels: for each training TM, all 3
selectors are evaluated with LP, and the winner is the label.

Architecture:
  Input features (per timestep):
    - TM statistics: mean, std, max, skew, entropy, top-10 share
    - Selector score statistics: mean, std, max of BN/TopK/Sens scores
    - Score agreement: overlap between selectors' top-K sets
    - Topology features: num_nodes, num_edges, density
  Output: 3-class softmax {0=Bottleneck, 1=TopK, 2=Sensitivity}
  Model: 2-layer MLP with dropout
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SELECTOR_NAMES = ["bottleneck", "topk", "sensitivity"]
NUM_SELECTORS = 3


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
    oracle_selector: str  # for evaluation
    predicted_mlu: float
    oracle_mlu: float


def compute_selector_scores(
    tm_vector: np.ndarray,
    ecmp_base: list,
    path_library,
    capacities: np.ndarray,
    k_crit: int,
) -> Dict[str, Tuple[List[int], np.ndarray]]:
    """Run all 3 selectors to get their selected ODs and scores. No LP needed."""
    from te.baselines import (
        select_bottleneck_critical,
        select_topk_by_demand,
        select_sensitivity_critical,
    )

    bn_selected = select_bottleneck_critical(tm_vector, ecmp_base, path_library, capacities, k_crit)
    topk_selected = select_topk_by_demand(tm_vector, k_crit)
    sens_selected = select_sensitivity_critical(tm_vector, ecmp_base, path_library, capacities, k_crit)

    return {
        "bottleneck": bn_selected,
        "topk": topk_selected,
        "sensitivity": sens_selected,
    }


def extract_features(
    tm_vector: np.ndarray,
    selector_results: Dict[str, List[int]],
    num_nodes: int,
    num_edges: int,
    k_crit: int,
) -> np.ndarray:
    """Extract feature vector for the meta-gate classifier.

    Features (34-dim):
      TM stats (8): mean, std, max, min_nonzero, skew, kurtosis, entropy, top10_share
      Per-selector (3x5=15): mean_demand, std_demand, max_demand, coverage, overlap_with_others
      Topology (3): log_nodes, log_edges, density
      Cross-selector (8): pairwise overlaps, unique counts
    """
    tm = np.maximum(tm_vector, 0.0)
    total = float(np.sum(tm))
    nonzero = tm[tm > 0]

    # TM statistics
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
    # Entropy of demand distribution
    if total > 0 and nonzero.size > 0:
        probs = nonzero / total
        tm_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
    else:
        tm_entropy = 0.0
    # Top-10 demand share
    sorted_tm = np.sort(tm)[::-1]
    top10_share = float(np.sum(sorted_tm[:10]) / max(total, 1e-12))

    tm_feats = [tm_mean, tm_std, tm_max, tm_min_nz, tm_skew, tm_kurt, tm_entropy, top10_share]

    # Per-selector features
    selector_feats = []
    sets = {}
    for name in SELECTOR_NAMES:
        selected = selector_results[name]
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

    # Cross-selector features
    bn_set = sets["bottleneck"]
    topk_set = sets["topk"]
    sens_set = sets["sensitivity"]

    def overlap(a, b):
        if not a or not b:
            return 0.0
        return len(a & b) / max(len(a | b), 1)

    cross_feats = [
        overlap(bn_set, topk_set),
        overlap(bn_set, sens_set),
        overlap(topk_set, sens_set),
        overlap(bn_set & topk_set, sens_set),  # triple overlap
        len(bn_set - topk_set - sens_set) / max(k_crit, 1),  # unique to BN
        len(topk_set - bn_set - sens_set) / max(k_crit, 1),  # unique to TopK
        len(sens_set - bn_set - topk_set) / max(k_crit, 1),  # unique to Sens
        len(bn_set & topk_set & sens_set) / max(k_crit, 1),  # all agree
    ]

    # Topology features
    density = (2 * num_edges) / max(num_nodes * (num_nodes - 1), 1)
    topo_feats = [np.log1p(num_nodes), np.log1p(num_edges), density]

    return np.array(tm_feats + selector_feats + cross_feats + topo_feats, dtype=np.float32)


class DynamicMetaGate:
    """Per-timestep meta-selector using a trained MLP classifier.

    This is the TRUE Mode B implementation: at each timestep, it predicts
    which of {Bottleneck, TopK, Sensitivity} will yield the lowest MLU,
    then runs ONLY that selector.
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
            labels: [N] integer labels {0=BN, 1=TopK, 2=Sens}
        """
        import torch
        import torch.nn as nn

        # Normalize features
        self.scaler_mean = features.mean(axis=0)
        self.scaler_std = features.std(axis=0) + 1e-8
        X = (features - self.scaler_mean) / self.scaler_std
        y = labels.astype(np.int64)

        feat_dim = X.shape[1]
        h = self.config.hidden_dim

        self.model = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(h // 2, NUM_SELECTORS),
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        best_val_acc = 0.0
        best_state = None

        for epoch in range(self.config.num_epochs):
            self.model.train()
            # Mini-batch
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

    def predict(self, features: np.ndarray) -> Tuple[int, np.ndarray]:
        """Predict which selector to use.

        Returns: (predicted_class, probabilities)
        """
        import torch

        if not self._is_trained:
            return 0, np.array([1.0, 0.0, 0.0])  # default to BN

        X = (features - self.scaler_mean) / self.scaler_std
        X_t = torch.tensor(X.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).numpy()[0]
        return int(np.argmax(probs)), probs

    def select(
        self,
        tm_vector: np.ndarray,
        ecmp_base: list,
        path_library,
        capacities: np.ndarray,
        k_crit: int,
        num_nodes: int,
        num_edges: int,
    ) -> Tuple[str, List[int], float]:
        """Full meta-gate inference: predict best selector, return its OD selection.

        Returns: (selector_name, selected_ods, confidence)
        """
        # Step 1: Run all 3 selectors (fast, no LP)
        selector_results = compute_selector_scores(
            tm_vector, ecmp_base, path_library, capacities, k_crit
        )

        # Step 2: Extract features
        feats = extract_features(tm_vector, selector_results, num_nodes, num_edges, k_crit)

        # Step 3: Predict
        pred_class, probs = self.predict(feats)
        selector_name = SELECTOR_NAMES[pred_class]
        confidence = float(probs[pred_class])

        # Step 4: Return the predicted selector's OD selection
        selected_ods = selector_results[selector_name]

        return selector_name, selected_ods, confidence

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
        self.model = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.ReLU(),
            nn.Dropout(payload["config"]["dropout"]),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Dropout(payload["config"]["dropout"]),
            nn.Linear(h // 2, NUM_SELECTORS),
        )
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()
        self._is_trained = True
        logger.info(f"MetaGate loaded from {path}")
