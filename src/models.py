"""Training XGBoost (Poisson skor + classifier W/D/L), blending, dan backtest."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier, XGBRegressor

from data_prep import BASE_DIR
from features import FEATURE_COLS, build_match_features, time_decay_weights

MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

MAX_GOALS = 10  # grid skor 0..10
GOAL_CAP = 10  # redam outlier (31-0 dsb.) saat training Poisson
ELO_DIFF_IDX = FEATURE_COLS.index("elo_diff")
# (classifier, poisson, elo-logistic, prior) — prior = shrinkage ke base-rate
DEFAULT_WEIGHTS = (0.2, 0.2, 0.5, 0.1)

XGB_REG_PARAMS = dict(
    objective="count:poisson",
    n_estimators=600,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    n_jobs=-1,
    random_state=42,
)

XGB_CLF_PARAMS = dict(
    objective="multi:softprob",
    num_class=3,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    n_jobs=-1,
    random_state=42,
)


def train_models(train_df: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """Latih ensemble: model gol Poisson, classifier W/D/L, dan Elo-logistik."""
    X = train_df[FEATURE_COLS].to_numpy()
    w = time_decay_weights(train_df["date"], cutoff)
    y_home = train_df["home_score"].clip(upper=GOAL_CAP).to_numpy()
    y_away = train_df["away_score"].clip(upper=GOAL_CAP).to_numpy()
    y_out = train_df["outcome"].to_numpy()

    goals_home = XGBRegressor(**XGB_REG_PARAMS)
    goals_home.fit(X, y_home, sample_weight=w)
    goals_away = XGBRegressor(**XGB_REG_PARAMS)
    goals_away.fit(X, y_away, sample_weight=w)

    clf = CalibratedClassifierCV(
        XGBClassifier(**XGB_CLF_PARAMS), method="isotonic", cv=3
    )
    clf.fit(X, y_out, sample_weight=w)

    # tanpa decay: regularisasi maksimal — komponen paling stabil ensemble
    elo_lr = LogisticRegression(max_iter=1000)
    elo_lr.fit(X[:, [ELO_DIFF_IDX]], y_out)

    prior = np.array([np.sum(w[y_out == c]) for c in (0, 1, 2)])
    prior /= prior.sum()

    return {
        "goals_home": goals_home,
        "goals_away": goals_away,
        "outcome": clf,
        "elo_logistic": elo_lr,
        "prior": prior,
        "weights": DEFAULT_WEIGHTS,
    }


def predict_lambdas(models: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Expected goals (lambda) kedua tim; dibatasi agar tetap waras."""
    lh = np.clip(models["goals_home"].predict(X), 0.05, 6.0)
    la = np.clip(models["goals_away"].predict(X), 0.05, 6.0)
    return lh, la


def poisson_pmf_vector(lam: np.ndarray, max_goals: int = MAX_GOALS) -> np.ndarray:
    """PMF Poisson per baris: shape (n, max_goals+1), dinormalisasi."""
    k = np.arange(max_goals + 1)
    log_fact = np.cumsum(np.log(np.maximum(k, 1)))
    lam = np.asarray(lam, dtype=np.float64).reshape(-1, 1)
    logp = -lam + k * np.log(lam) - log_fact
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def score_matrix(lam_home: float, lam_away: float) -> np.ndarray:
    """Matriks probabilitas skor (home_gol x away_gol)."""
    ph = poisson_pmf_vector(np.array([lam_home]))[0]
    pa = poisson_pmf_vector(np.array([lam_away]))[0]
    return np.outer(ph, pa)


def outcome_probs_from_lambdas(lh: np.ndarray, la: np.ndarray) -> np.ndarray:
    """Probabilitas (menang, seri, kalah) dari grid Poisson — vektorized."""
    ph = poisson_pmf_vector(lh)  # (n, K)
    pa = poisson_pmf_vector(la)
    grid = ph[:, :, None] * pa[:, None, :]  # (n, K, K)
    iu = np.triu_indices(grid.shape[1], k=1)
    il = np.tril_indices(grid.shape[1], k=-1)
    p_draw = grid[:, np.arange(grid.shape[1]), np.arange(grid.shape[1])].sum(axis=1)
    p_home = grid[:, il[0], il[1]].sum(axis=1)  # home_gol > away_gol
    p_away = grid[:, iu[0], iu[1]].sum(axis=1)
    return np.column_stack([p_home, p_draw, p_away])


def component_probs(models: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Probabilitas W/D/L ketiga komponen ensemble: (classifier, poisson, elo)."""
    p_clf = models["outcome"].predict_proba(X)
    lh, la = predict_lambdas(models, X)
    p_poi = outcome_probs_from_lambdas(lh, la)
    p_elo = models["elo_logistic"].predict_proba(X[:, [ELO_DIFF_IDX]])
    return p_clf, p_poi, p_elo


def ensemble_probs(
    models: dict, X: np.ndarray, weights: tuple[float, ...] | None = None
) -> np.ndarray:
    """Probabilitas akhir = kombinasi berbobot 4 komponen (+ prior)."""
    w1, w2, w3, w4 = weights or models.get("weights", DEFAULT_WEIGHTS)
    p_clf, p_poi, p_elo = component_probs(models, X)
    p = w1 * p_clf + w2 * p_poi + w3 * p_elo + w4 * models["prior"][None, :]
    return p / p.sum(axis=1, keepdims=True)


def tune_blend_weights(
    featured: pd.DataFrame, cutoff: pd.Timestamp, n_editions: int = 2, step: float = 0.1
) -> tuple[float, ...]:
    """Cari bobot ensemble via grid search di simplex 4 komponen.

    Validasi = laga DUA EDISI PIALA DUNIA terakhir sebelum cutoff
    (walk-forward bersarang): model dasar dilatih hanya dengan data
    sebelum edisi tertua validasi, lalu bobot dipilih yang meminimalkan
    log-loss pada laga-laga PD tersebut. Distribusi validasi = distribusi
    target (pertandingan Piala Dunia), tanpa menyentuh data tes.
    """
    wc = featured[
        (featured["tournament"] == "FIFA World Cup") & (featured["date"] < cutoff)
    ]
    years = sorted(wc["date"].dt.year.unique())[-n_editions:]
    val = wc[wc["date"].dt.year.isin(years)]
    inner_cutoff = val["date"].min()
    train = featured[featured["date"] < inner_cutoff]
    base = train_models(train, inner_cutoff)
    p_clf, p_poi, p_elo = component_probs(base, val[FEATURE_COLS].to_numpy())
    p_pri = np.tile(base["prior"], (len(val), 1))
    y = val["outcome"].to_numpy()

    best, best_ll = DEFAULT_WEIGHTS, np.inf
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1.0 + 1e-9:
                continue
            for w3 in grid:
                w4 = 1.0 - w1 - w2 - w3
                if w4 < -1e-9:
                    continue
                w4 = max(w4, 0.0)
                p = w1 * p_clf + w2 * p_poi + w3 * p_elo + w4 * p_pri
                p /= p.sum(axis=1, keepdims=True)
                ll = log_loss(y, p, labels=[0, 1, 2])
                if ll < best_ll:
                    best_ll = ll
                    best = (round(w1, 3), round(w2, 3), round(w3, 3), round(w4, 3))
    return best


def most_likely_score(lam_home: float, lam_away: float) -> tuple[int, int, float]:
    m = score_matrix(lam_home, lam_away)
    h, a = np.unravel_index(np.argmax(m), m.shape)
    return int(h), int(a), float(m[h, a])


# ---------------------------------------------------------------------------
# Backtest walk-forward pada Piala Dunia 2018 & 2022
# ---------------------------------------------------------------------------

def _brier(y_true: np.ndarray, p: np.ndarray) -> float:
    onehot = np.eye(3)[y_true]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "log_loss": float(log_loss(y, p, labels=[0, 1, 2])),
        "brier": _brier(y, p),
        "accuracy": float(accuracy_score(y, p.argmax(axis=1))),
    }


def backtest_world_cup(featured: pd.DataFrame, wc_year: int) -> dict:
    """Train pada data sebelum turnamen, evaluasi pada laga PD tahun itu."""
    test_mask = (featured["tournament"] == "FIFA World Cup") & (
        featured["date"].dt.year == wc_year
    )
    test = featured[test_mask]
    cutoff = test["date"].min()
    train = featured[featured["date"] < cutoff]

    weights = tune_blend_weights(train, cutoff)
    models = train_models(train, cutoff)
    models["weights"] = weights
    Xte = test[FEATURE_COLS].to_numpy()
    yte = test["outcome"].to_numpy()

    lh, la = predict_lambdas(models, Xte)
    results = {
        "n_matches": int(len(test)),
        "blend_weights": list(weights),
        "model_blend": _metrics(yte, ensemble_probs(models, Xte)),
        "model_classifier": _metrics(yte, models["outcome"].predict_proba(Xte)),
        "model_poisson": _metrics(yte, outcome_probs_from_lambdas(lh, la)),
    }

    # Baseline 1: regresi logistik hanya dengan elo_diff
    base = LogisticRegression(max_iter=1000)
    base.fit(train[["elo_diff"]].to_numpy(), train["outcome"].to_numpy())
    results["baseline_elo_logistic"] = _metrics(
        yte, base.predict_proba(test[["elo_diff"]].to_numpy())
    )

    # Baseline 2: prior frekuensi global hasil pertandingan
    prior = np.bincount(train["outcome"], minlength=3) / len(train)
    results["baseline_prior"] = _metrics(yte, np.tile(prior, (len(test), 1)))

    # MAE gol sebagai ukuran kualitas prediksi skor
    results["goals_mae"] = {
        "home": float(np.mean(np.abs(lh - test["home_score"].to_numpy()))),
        "away": float(np.mean(np.abs(la - test["away_score"].to_numpy()))),
    }
    return results


def run_full_training(featured: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    """Latih model final pada seluruh data < cutoff, tuning bobot, simpan."""
    MODELS_DIR.mkdir(exist_ok=True)
    train = featured[featured["date"] < cutoff]
    weights = tune_blend_weights(train, cutoff)
    models = train_models(train, cutoff)
    models["weights"] = weights
    joblib.dump(models, MODELS_DIR / "wc2026_models.joblib")
    return models


def save_metrics(metrics: dict) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def load_models() -> dict:
    return joblib.load(MODELS_DIR / "wc2026_models.joblib")
