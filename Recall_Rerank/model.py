from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from Recall_Rerank.features import FEATURE_NAMES


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    denom = np.sum(exps)
    if denom <= 0:
        return np.ones_like(logits) / max(1, logits.shape[0])
    return exps / denom


@dataclass
class ListwiseGroup:
    query_id: str
    features: np.ndarray
    target_index: int


class LinearListwiseCEModel:
    def __init__(
        self,
        feature_names: list[str],
        weights: np.ndarray,
        bias: float,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.feature_names = list(feature_names)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.bias = float(bias)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_std = np.asarray(feature_std, dtype=np.float64)
        self.metadata = dict(metadata or {})

    @classmethod
    def zeros(cls, feature_names: list[str]) -> "LinearListwiseCEModel":
        dim = len(feature_names)
        return cls(
            feature_names=feature_names,
            weights=np.zeros(dim, dtype=np.float64),
            bias=0.0,
            feature_mean=np.zeros(dim, dtype=np.float64),
            feature_std=np.ones(dim, dtype=np.float64),
            metadata={"init": "zeros"},
        )

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        denom = np.where(self.feature_std <= 1e-12, 1.0, self.feature_std)
        return (x - self.feature_mean) / denom

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        xn = self._normalize(x)
        return xn @ self.weights + self.bias

    def predict_probs(self, x: np.ndarray) -> np.ndarray:
        logits = self.predict_logits(x)
        return _softmax(logits)

    def explain_row(self, x_row: np.ndarray, top_n: int = 3) -> str:
        x_row = np.asarray(x_row, dtype=np.float64).reshape(1, -1)
        xn = self._normalize(x_row)[0]
        contrib = xn * self.weights
        ranked = np.argsort(-contrib)
        parts: list[str] = []
        for idx in ranked[: max(1, top_n)]:
            value = float(contrib[idx])
            if abs(value) < 1e-9:
                continue
            sign = "+" if value >= 0 else "-"
            parts.append(f"{self.feature_names[int(idx)]}:{sign}{abs(value):.3f}")
        if not parts:
            return "Linear listwise CE rerank."
        return "Top linear signals: " + ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "linear_listwise_ce_v1",
            "feature_names": self.feature_names,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "LinearListwiseCEModel":
        feature_names = row.get("feature_names", FEATURE_NAMES)
        weights = np.asarray(row.get("weights", [0.0] * len(feature_names)), dtype=np.float64)
        bias = float(row.get("bias", 0.0))
        mean = np.asarray(row.get("feature_mean", [0.0] * len(feature_names)), dtype=np.float64)
        std = np.asarray(row.get("feature_std", [1.0] * len(feature_names)), dtype=np.float64)
        if len(weights) != len(feature_names):
            raise ValueError("Model weight dimension mismatch.")
        if len(mean) != len(feature_names) or len(std) != len(feature_names):
            raise ValueError("Model normalization vector dimension mismatch.")
        return cls(
            feature_names=list(feature_names),
            weights=weights,
            bias=bias,
            feature_mean=mean,
            feature_std=std,
            metadata=row.get("metadata", {}) if isinstance(row.get("metadata", {}), dict) else {},
        )

    @classmethod
    def load(cls, path: str | Path) -> "LinearListwiseCEModel":
        with Path(path).open("r", encoding="utf-8") as f:
            row = json.load(f)
        if not isinstance(row, dict):
            raise ValueError("Invalid model file format.")
        return cls.from_dict(row)


def train_linear_listwise_ce(
    groups: list[ListwiseGroup],
    feature_names: list[str],
    lr: float = 0.05,
    epochs: int = 200,
    l2: float = 1e-4,
    seed: int = 42,
) -> tuple[LinearListwiseCEModel, dict[str, Any]]:
    if not groups:
        raise ValueError("No training groups available.")
    dim = len(feature_names)
    if dim <= 0:
        raise ValueError("Feature dimension must be positive.")

    stacked = np.vstack([g.features for g in groups if g.features.size > 0])
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    std = np.where(std <= 1e-12, 1.0, std)

    norm_groups: list[tuple[np.ndarray, int]] = []
    for g in groups:
        if g.features.shape[1] != dim:
            raise ValueError(f"Feature dimension mismatch in query={g.query_id}.")
        if g.target_index < 0 or g.target_index >= g.features.shape[0]:
            raise ValueError(f"Target index out of range in query={g.query_id}.")
        x = (g.features - mean) / std
        norm_groups.append((x, g.target_index))

    rng = np.random.default_rng(seed)
    w = rng.normal(loc=0.0, scale=0.01, size=dim)
    b = 0.0

    history: list[dict[str, float]] = []
    n_groups = float(len(norm_groups))
    for epoch in range(1, epochs + 1):
        grad_w = np.zeros(dim, dtype=np.float64)
        grad_b = 0.0
        loss_sum = 0.0
        for x, target in norm_groups:
            logits = x @ w + b
            probs = _softmax(logits)
            loss_sum += -float(np.log(max(1e-12, probs[target])))
            diff = probs.copy()
            diff[target] -= 1.0
            grad_w += x.T @ diff
            grad_b += float(np.sum(diff))

        loss = (loss_sum / n_groups) + l2 * float(np.sum(w * w))
        grad_w = (grad_w / n_groups) + (2.0 * l2 * w)
        grad_b = grad_b / n_groups

        w -= lr * grad_w
        b -= lr * grad_b
        history.append({"epoch": float(epoch), "loss": float(loss)})

    model = LinearListwiseCEModel(
        feature_names=feature_names,
        weights=w,
        bias=b,
        feature_mean=mean,
        feature_std=std,
        metadata={
            "epochs": int(epochs),
            "lr": float(lr),
            "l2": float(l2),
            "seed": int(seed),
            "train_group_count": int(len(groups)),
            "train_loss": float(history[-1]["loss"]) if history else None,
        },
    )
    summary = {
        "epochs": int(epochs),
        "lr": float(lr),
        "l2": float(l2),
        "seed": int(seed),
        "train_group_count": int(len(groups)),
        "final_loss": float(history[-1]["loss"]) if history else None,
    }
    return model, summary

