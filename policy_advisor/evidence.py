"""Evidence layer: World Bank data, life-expectancy model and intervention simulator.

This module carries forward the data-science project ("What Makes a Nation Live
Longer?"): the same cleaning rules (sparse/leaky column removal, log transforms,
within-country interpolation), the same preprocessing pipeline and the same
country-grouped cross-validation, so that no test country is ever seen during
training. It exposes the results as pure functions that the agent can call as
typed tools.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "world_bank_indicators.csv"
MODEL_DIR = ROOT / "outputs" / "models"
RANDOM_STATE = 42

SPARSE_COLS = ["smoking_pct", "adult_literacy_pct", "gini"]
LEAKY_COLS = ["under5_mortality"]
LOG_COLS = ["gdp_per_capita", "health_exp_per_capita", "population", "co2_per_capita"]
CATEGORICAL = ["region", "income_group"]
TARGET = "life_expectancy"
LATEST_YEAR = 2022

# Levers a ministry can realistically move, with the bounds the simulator accepts.
LEVERS: dict[str, tuple[float, float]] = {
    "basic_water_pct": (0, 100),
    "basic_sanitation_pct": (0, 100),
    "electricity_access_pct": (0, 100),
    "measles_immunization_pct": (0, 100),
    "secondary_enrollment_pct": (0, 100),
    "internet_users_pct": (0, 100),
    "fertility_rate": (0.8, 8.0),
    "health_exp_per_capita": (1, 20000),
}

# Countries whose grouped-CV residual exceeded 2x the MAE in the prior project;
# used to warn the agent that the evidence base is weak there.
KNOWN_HARD_CASES = {"NGA", "ZAF", "PLW", "TCD", "SWZ", "GNQ", "NRU", "LSO", "CAF"}


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame[CATEGORICAL] = frame[CATEGORICAL].apply(lambda s: s.str.strip())
    return frame


def target_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    """Implausible targets, replicating the prior project's rule (below 30 years)."""
    return frame[frame[TARGET] < 30]


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.drop(columns=SPARSE_COLS + LEAKY_COLS).dropna(subset=[TARGET])
    out = out.drop(index=target_outliers(out).index).copy()
    for col in LOG_COLS:
        out[f"log_{col}"] = np.log1p(out[col])
    out = out.drop(columns=LOG_COLS)
    feature_cols = [c for c in out.select_dtypes("number").columns if c not in (TARGET, "year")]
    out[feature_cols] = out.groupby("iso3")[feature_cols].transform(lambda s: s.interpolate(limit_direction="both"))
    return out.reset_index(drop=True)


def numeric_features(clean: pd.DataFrame) -> list[str]:
    return [c for c in clean.select_dtypes("number").columns if c not in (TARGET, "year")]


def build_pipeline(regressor: Any, numeric: list[str]) -> Pipeline:
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    preprocess = ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)]
    )
    return Pipeline([("prep", preprocess), ("model", regressor)])


def grouped_split(clean: pd.DataFrame, test_size: float = 0.2):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(clean, groups=clean["iso3"]))
    return clean.iloc[train_idx], clean.iloc[test_idx]


@dataclass
class EvidenceBase:
    """Trained life-expectancy model plus the data needed to answer policy questions."""

    clean: pd.DataFrame
    numeric: list[str]
    features: list[str]
    model: Pipeline
    metrics: dict[str, float]
    cv_metrics: dict[str, float]
    importance: pd.Series
    test_countries: set[str]
    country_residuals: pd.Series
    lever_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    # ------------------------------------------------------------------ build
    @classmethod
    def load_or_build(cls, cache: Path = MODEL_DIR / "evidence_base.pkl", verbose: bool = True) -> EvidenceBase:
        if cache.exists():
            with cache.open("rb") as fh:
                base = pickle.load(fh)
            if verbose:
                print(f"Loaded cached evidence base from {cache.relative_to(ROOT)}")
            return base
        base = cls.build(verbose=verbose)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as fh:
            pickle.dump(base, fh)
        base.save_card()
        return base

    @classmethod
    def build(cls, data_path: Path = DATA_PATH, verbose: bool = True) -> EvidenceBase:
        clean = clean_data(load_raw(data_path))
        numeric = numeric_features(clean)
        features = numeric + CATEGORICAL
        gbr = GradientBoostingRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=RANDOM_STATE
        )

        # Country-grouped 5-fold CV: the honest estimate for a never-seen country.
        cv_scores = {"r2": [], "mae": []}
        for train_idx, test_idx in GroupKFold(n_splits=5).split(clean, groups=clean["iso3"]):
            pipe = build_pipeline(gbr, numeric).fit(clean.iloc[train_idx][features], clean.iloc[train_idx][TARGET])
            pred = pipe.predict(clean.iloc[test_idx][features])
            cv_scores["r2"].append(r2_score(clean.iloc[test_idx][TARGET], pred))
            cv_scores["mae"].append(mean_absolute_error(clean.iloc[test_idx][TARGET], pred))
        cv_metrics = {k: float(np.mean(v)) for k, v in cv_scores.items()}

        train, test = grouped_split(clean)
        model = build_pipeline(gbr, numeric).fit(train[features], train[TARGET])
        pred = model.predict(test[features])
        metrics = {
            "r2": float(r2_score(test[TARGET], pred)),
            "mae": float(mean_absolute_error(test[TARGET], pred)),
            "rmse": float(np.sqrt(mean_squared_error(test[TARGET], pred))),
            "n_train_countries": int(train["iso3"].nunique()),
            "n_test_countries": int(test["iso3"].nunique()),
        }
        residuals = (test[TARGET] - pred).groupby(test["iso3"]).apply(lambda s: float(np.abs(s).mean()))
        perm = permutation_importance(model, test[features], test[TARGET], n_repeats=5, random_state=RANDOM_STATE)
        importance = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)

        lever_bounds = {}
        for lever, (lo, hi) in LEVERS.items():
            col = f"log_{lever}" if lever in LOG_COLS else lever
            observed = np.expm1(clean[col]) if lever in LOG_COLS else clean[col]
            lever_bounds[lever] = (lo, min(hi, float(observed.quantile(0.995))))

        base = cls(clean, numeric, features, model, metrics, cv_metrics, importance,
                   set(test["iso3"]), residuals, lever_bounds)
        if verbose:
            print(f"Evidence base ready: {clean.shape[0]} country-years, {clean['iso3'].nunique()} countries. "
                  f"Grouped-CV R2={cv_metrics['r2']:.3f} MAE={cv_metrics['mae']:.2f}y; "
                  f"hold-out R2={metrics['r2']:.3f} MAE={metrics['mae']:.2f}y")
        return base

    # ------------------------------------------------------------- utilities
    def _latest_row(self, iso3: str) -> pd.Series | None:
        rows = self.clean[self.clean["iso3"] == iso3.upper()]
        if rows.empty:
            return None
        return rows.sort_values("year").iloc[-1]

    def resolve_country(self, query: str) -> str | None:
        query = query.strip()
        if len(query) == 3 and query.upper() in set(self.clean["iso3"]):
            return query.upper()
        names = self.clean.drop_duplicates("iso3").set_index("country")["iso3"]
        exact = names[names.index.str.lower() == query.lower()]
        if not exact.empty:
            return str(exact.iloc[0])
        partial = names[names.index.str.lower().str.contains(query.lower(), regex=False)]
        return str(partial.iloc[0]) if len(partial) == 1 else None

    def evidence_quality(self, iso3: str) -> dict[str, Any]:
        """How much should the agent trust the model for this country?"""
        raw = load_raw()
        raw = raw[raw["iso3"] == iso3]
        latest = raw[raw["year"] == raw["year"].max()]
        missing_latest = latest[[c for c in LEVERS if c in latest.columns]].isna().mean(axis=1).iloc[0]
        flags = []
        if iso3 in KNOWN_HARD_CASES:
            flags.append("country was a large-residual case in grouped cross-validation")
        if missing_latest > 0.3:
            flags.append(f"{missing_latest:.0%} of lever indicators missing in the latest year (interpolated)")
        if iso3 in self.test_countries and float(self.country_residuals.get(iso3, 0)) > 2 * self.metrics["mae"]:
            flags.append(f"hold-out error {self.country_residuals[iso3]:.1f}y exceeds twice the model MAE")
        return {
            "iso3": iso3,
            "hold_out_country": iso3 in self.test_countries,
            "hold_out_abs_error_years": round(float(self.country_residuals.get(iso3, float("nan"))), 2)
            if iso3 in self.test_countries else None,
            "missing_share_latest_year": round(float(missing_latest), 3),
            "flags": flags,
            "trust": "low" if flags else "normal",
        }

    # ------------------------------------------------------------------ tools
    def country_profile(self, country: str) -> dict[str, Any]:
        iso3 = self.resolve_country(country)
        if iso3 is None:
            return {"error": f"Unknown or ambiguous country: {country!r}. Use an ISO3 code or the exact World Bank name."}
        row = self._latest_row(iso3)
        assert row is not None
        profile = {
            "iso3": iso3,
            "country": row["country"],
            "year": int(row["year"]),
            "region": row["region"],
            "income_group": row["income_group"],
            "life_expectancy": round(float(row[TARGET]), 1),
            "model_prediction": round(float(self.model.predict(row[self.features].to_frame().T)[0]), 1),
        }
        for lever in LEVERS:
            col = f"log_{lever}" if lever in LOG_COLS else lever
            value = row[col]
            if pd.notna(value):
                profile[lever] = round(float(np.expm1(value) if lever in LOG_COLS else value), 1)
        profile["evidence_quality"] = self.evidence_quality(iso3)
        return profile

    def peer_countries(self, country: str, n: int = 5) -> dict[str, Any]:
        """Same region and income group in the latest year, with the wealth-adjusted over/under-performance."""
        iso3 = self.resolve_country(country)
        if iso3 is None:
            return {"error": f"Unknown country: {country!r}"}
        row = self._latest_row(iso3)
        assert row is not None
        latest = self.clean[self.clean["year"] == self.clean["year"].max()]
        peers = latest[(latest["region"] == row["region"]) & (latest["income_group"] == row["income_group"])
                       & (latest["iso3"] != iso3)].copy()
        if peers.empty:
            peers = latest[(latest["income_group"] == row["income_group"]) & (latest["iso3"] != iso3)].copy()
        peers["predicted"] = self.model.predict(peers[self.features])
        peers["residual_vs_model"] = peers[TARGET] - peers["predicted"]
        peers = peers.sort_values("residual_vs_model", ascending=False)
        levers_present = [lever if lever not in LOG_COLS else f"log_{lever}" for lever in LEVERS]

        def compact(r: pd.Series) -> dict[str, Any]:
            item = {"iso3": r["iso3"], "country": r["country"], "life_expectancy": round(float(r[TARGET]), 1),
                    "residual_vs_model": round(float(r["residual_vs_model"]), 1)}
            for lever, col in zip(LEVERS, levers_present, strict=True):
                if pd.notna(r[col]):
                    item[lever] = round(float(np.expm1(r[col]) if lever in LOG_COLS else r[col]), 1)
            return item

        return {
            "target": iso3,
            "peer_group": f"{row['region']} / {row['income_group']}",
            "n_peers": int(len(peers)),
            "over_performers": [compact(r) for _, r in peers.head(n).iterrows()],
            "under_performers": [compact(r) for _, r in peers.tail(n).iterrows()],
            "peer_median_life_expectancy": round(float(peers[TARGET].median()), 1) if len(peers) else None,
        }

    def simulate_intervention(self, country: str, interventions: dict[str, float]) -> dict[str, Any]:
        """Predict life expectancy for the latest profile with the given lever values applied."""
        iso3 = self.resolve_country(country)
        if iso3 is None:
            return {"error": f"Unknown country: {country!r}"}
        row = self._latest_row(iso3)
        assert row is not None
        baseline = row[self.features].copy()
        modified = baseline.copy()
        applied: dict[str, dict[str, float]] = {}
        caveats: list[str] = []
        for lever, value in interventions.items():
            if lever not in LEVERS:
                return {"error": f"Unknown lever {lever!r}. Allowed: {sorted(LEVERS)}"}
            lo, hi = LEVERS[lever]
            if not (lo <= float(value) <= hi):
                return {"error": f"{lever}={value} is outside the accepted range [{lo}, {hi}]"}
            col = f"log_{lever}" if lever in LOG_COLS else lever
            before = float(np.expm1(baseline[col]) if lever in LOG_COLS else baseline[col])
            if np.isnan(before):
                caveats.append(f"{lever} is missing for {iso3}; baseline imputed by the model.")
            modified[col] = np.log1p(float(value)) if lever in LOG_COLS else float(value)
            applied[lever] = {"before": round(before, 2) if not np.isnan(before) else None, "after": float(value)}
            _, observed_hi = self.lever_bounds[lever]
            if float(value) > observed_hi:
                caveats.append(f"{lever}={value} exceeds the 99.5th percentile observed in the data ({observed_hi:.1f}); "
                               "prediction is an extrapolation.")
            if before is not None and not np.isnan(before) and float(value) < before and lever != "fertility_rate" \
                    and lever != "health_exp_per_capita":
                caveats.append(f"{lever} is being reduced below its current level.")
        frame = pd.DataFrame([baseline, modified])
        preds = self.model.predict(frame[self.features])
        gain = float(preds[1] - preds[0])
        if abs(gain) < self.cv_metrics["mae"] / 2:
            caveats.append(f"Predicted change ({gain:+.1f}y) is small relative to model MAE ({self.cv_metrics['mae']:.1f}y).")
        return {
            "iso3": iso3,
            "country": row["country"],
            "year": int(row["year"]),
            "observed_life_expectancy": round(float(row[TARGET]), 1),
            "baseline_prediction": round(float(preds[0]), 1),
            "scenario_prediction": round(float(preds[1]), 1),
            "predicted_gain_years": round(gain, 1),
            "model_mae_years": round(self.cv_metrics["mae"], 1),
            "applied": applied,
            "caveats": caveats,
            "evidence_quality": self.evidence_quality(iso3),
            "note": "Associational model: predicted gains describe countries that already have these indicator levels, "
                    "not the causal effect of a programme.",
        }

    def model_card(self) -> dict[str, Any]:
        return {
            "model": "GradientBoostingRegressor(n_estimators=500, lr=0.05, depth=3, subsample=0.8)",
            "training_data": f"World Bank WDI 2000-{int(self.clean['year'].max())}, {self.clean['iso3'].nunique()} countries",
            "validation": "5-fold country-grouped CV + 20% country-grouped hold-out",
            "grouped_cv": {k: round(v, 3) for k, v in self.cv_metrics.items()},
            "hold_out": {k: round(v, 3) if isinstance(v, float) else v for k, v in self.metrics.items()},
            "top_features": {k: round(float(v), 3) for k, v in self.importance.head(8).items()},
            "known_hard_cases": sorted(KNOWN_HARD_CASES),
        }

    def save_card(self, path: Path = MODEL_DIR / "model_card.json") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_card(), indent=2), encoding="utf-8")
        return path
