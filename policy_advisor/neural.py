"""Deep-learning layer: controlled baseline-vs-regularised experiment and uncertainty tool.

Carries forward the deep-learning-systems project. There the question was whether
Batch Normalization + Dropout close the generalisation gap of a CNN on Fashion-MNIST
and how confident the network is when it is wrong. Here the same protocol (identical
seed, optimiser, epochs and split; only the regularisation differs) is run on the
tabular life-expectancy problem, and dropout is kept active at inference time
(Monte-Carlo dropout) so the network can report *how unsure it is* for a country.
That uncertainty, together with the disagreement between the neural and the
gradient-boosting predictions, becomes an agent tool.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from .evidence import (
    CATEGORICAL,
    LEVERS,
    LOG_COLS,
    MODEL_DIR,
    ROOT,
    TARGET,
    EvidenceBase,
    build_pipeline,
    grouped_split,
)

SEED = 42
EPOCHS = 60
BATCH = 128
LR = 1e-3


def _seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class BaselineMLP(nn.Module):
    def __init__(self, n_in: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RegularizedMLP(nn.Module):
    """Same width and depth as the baseline; only BatchNorm and Dropout are added."""

    def __init__(self, n_in: int, hidden: int = 128, p: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train(model: nn.Module, x_tr: torch.Tensor, y_tr: torch.Tensor, x_va: torch.Tensor, y_va: torch.Tensor,
           epochs: int = EPOCHS) -> pd.DataFrame:
    _seed()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    history = []
    n = len(x_tr)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(x_va), y_va).item()
        history.append({"epoch": epoch, "train_mse": total / n, "val_mse": val})
    return pd.DataFrame(history)


@dataclass
class NeuralComparison:
    preprocess: Any
    baseline: BaselineMLP
    regularized: RegularizedMLP
    baseline_history: pd.DataFrame
    regularized_history: pd.DataFrame
    summary: pd.DataFrame
    evidence: EvidenceBase
    mc_std_reference: dict[str, float]

    @classmethod
    def load_or_build(cls, evidence: EvidenceBase, cache: Path = MODEL_DIR / "neural_comparison.pkl",
                      verbose: bool = True) -> NeuralComparison:
        if cache.exists():
            with cache.open("rb") as fh:
                obj = pickle.load(fh)
            obj.evidence = evidence
            if verbose:
                print(f"Loaded cached neural comparison from {cache.relative_to(ROOT)}")
            return obj
        obj = cls.build(evidence, verbose=verbose)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as fh:
            pickle.dump(obj, fh)
        obj.summary.to_csv(MODEL_DIR / "neural_vs_boosting.csv")
        (MODEL_DIR / "uncertainty_reference.json").write_text(json.dumps(obj.mc_std_reference, indent=2))
        obj.baseline_history.to_csv(MODEL_DIR / "mlp_baseline_history.csv", index=False)
        obj.regularized_history.to_csv(MODEL_DIR / "mlp_regularized_history.csv", index=False)
        return obj

    @classmethod
    def build(cls, evidence: EvidenceBase, epochs: int = EPOCHS, verbose: bool = True) -> NeuralComparison:
        train, test = grouped_split(evidence.clean)
        prep = build_pipeline(nn.Identity(), evidence.numeric).named_steps["prep"]
        x_tr = torch.tensor(prep.fit_transform(train[evidence.features]).astype(np.float32))
        x_te = torch.tensor(prep.transform(test[evidence.features]).astype(np.float32))
        y_tr = torch.tensor(train[TARGET].to_numpy(np.float32))
        y_te = torch.tensor(test[TARGET].to_numpy(np.float32))

        _seed()
        base = BaselineMLP(x_tr.shape[1])
        base_hist = _train(base, x_tr, y_tr, x_te, y_te, epochs)
        _seed()
        reg = RegularizedMLP(x_tr.shape[1])
        reg_hist = _train(reg, x_tr, y_tr, x_te, y_te, epochs)

        def metrics(model: nn.Module) -> dict[str, float]:
            model.eval()
            with torch.no_grad():
                p_tr, p_te = model(x_tr).numpy(), model(x_te).numpy()
            mae_tr = float(np.mean(np.abs(p_tr - y_tr.numpy())))
            mae_te = float(np.mean(np.abs(p_te - y_te.numpy())))
            r2 = 1 - float(np.sum((p_te - y_te.numpy()) ** 2) / np.sum((y_te.numpy() - y_te.numpy().mean()) ** 2))
            return {"params": sum(p.numel() for p in model.parameters()), "train MAE": mae_tr, "test MAE": mae_te,
                    "test R2": r2, "generalisation gap (test-train MAE)": mae_te - mae_tr}

        gbr_pred = evidence.model.predict(test[evidence.features])
        gbr_tr = evidence.model.predict(train[evidence.features])
        gbr = {"params": float("nan"),
               "train MAE": float(np.mean(np.abs(gbr_tr - train[TARGET]))),
               "test MAE": float(np.mean(np.abs(gbr_pred - test[TARGET]))),
               "test R2": float(1 - np.sum((gbr_pred - test[TARGET]) ** 2) / np.sum((test[TARGET] - test[TARGET].mean()) ** 2))}
        gbr["generalisation gap (test-train MAE)"] = gbr["test MAE"] - gbr["train MAE"]
        summary = pd.DataFrame({"Baseline MLP": metrics(base), "Regularized MLP (BN+Dropout)": metrics(reg),
                                "Gradient boosting (evidence layer)": gbr})
        obj = cls(prep, base, reg, base_hist, reg_hist, summary, evidence, {})
        mean, std = obj.mc_dropout(test)
        err = np.abs(mean - y_te.numpy())
        gbr_err = np.abs(gbr_pred - y_te.numpy())
        disagreement = np.abs(gbr_pred - mean)
        obj.mc_std_reference = {
            "median": float(np.median(std)),
            "p75": float(np.percentile(std, 75)),
            "p90": float(np.percentile(std, 90)),
            "corr_std_vs_abs_error": float(np.corrcoef(std, err)[0, 1]),
            "mae_low_spread_half": float(err[std <= np.median(std)].mean()),
            "mae_high_spread_half": float(err[std > np.median(std)].mean()),
            "corr_disagreement_vs_gbr_error": float(np.corrcoef(disagreement, gbr_err)[0, 1]),
            "gbr_mae_when_models_agree": float(gbr_err[disagreement <= evidence.cv_metrics["mae"]].mean()),
            "gbr_mae_when_models_disagree": float(gbr_err[disagreement > evidence.cv_metrics["mae"]].mean()),
            "share_disagree": float((disagreement > evidence.cv_metrics["mae"]).mean()),
        }
        if verbose:
            print(summary.round(3).to_string())
            print("MC-dropout reference:", {k: round(v, 3) for k, v in obj.mc_std_reference.items()})
        return obj

    # ------------------------------------------------------------------ tool
    def mc_dropout(self, frame: pd.DataFrame, samples: int = 100) -> tuple[np.ndarray, np.ndarray]:
        x = torch.tensor(self.preprocess.transform(frame[self.evidence.features]).astype(np.float32))
        self.regularized.eval()
        for module in self.regularized.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        _seed()
        with torch.no_grad():
            draws = torch.stack([self.regularized(x) for _ in range(samples)]).numpy()
        self.regularized.eval()
        return draws.mean(axis=0), draws.std(axis=0)

    def uncertainty_estimate(self, country: str, interventions: dict[str, float] | None = None) -> dict[str, Any]:
        """Second opinion on a prediction: MC-dropout spread and cross-model disagreement."""
        ev = self.evidence
        iso3 = ev.resolve_country(country)
        if iso3 is None:
            return {"error": f"Unknown country: {country!r}"}
        row = ev._latest_row(iso3)
        assert row is not None
        profile = row[ev.features].copy()
        for lever, value in (interventions or {}).items():
            if lever not in LEVERS:
                return {"error": f"Unknown lever {lever!r}"}
            col = f"log_{lever}" if lever in LOG_COLS else lever
            profile[col] = np.log1p(float(value)) if lever in LOG_COLS else float(value)
        frame = pd.DataFrame([profile])
        frame[ev.numeric] = frame[ev.numeric].astype(float)
        frame[CATEGORICAL] = frame[CATEGORICAL].astype(str)
        nn_mean, nn_std = self.mc_dropout(frame)
        gbr_pred = float(ev.model.predict(frame)[0])
        disagreement = abs(gbr_pred - float(nn_mean[0]))
        flags = []
        if float(nn_std[0]) > self.mc_std_reference["p75"]:
            flags.append(f"MC-dropout spread {nn_std[0]:.1f}y is above the 75th percentile of hold-out countries "
                         f"({self.mc_std_reference['p75']:.1f}y)")
        if disagreement > ev.cv_metrics["mae"]:
            flags.append(f"neural and boosting models disagree by {disagreement:.1f}y (> model MAE)")
        return {
            "iso3": iso3,
            "gradient_boosting_prediction": round(gbr_pred, 1),
            "neural_prediction_mean": round(float(nn_mean[0]), 1),
            "neural_mc_dropout_std": round(float(nn_std[0]), 2),
            "hold_out_median_std": round(self.mc_std_reference["median"], 2),
            "cross_model_disagreement_years": round(disagreement, 1),
            "flags": flags,
            "confidence": "low" if flags else "normal",
        }
