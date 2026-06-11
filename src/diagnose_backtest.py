"""Diagnostik cepat: bandingkan metrik backtest semua varian model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_prep import load_results
from features import build_match_features
import models as M

res = load_results()
f = build_match_features(res)
for year in (2018, 2022):
    bt = M.backtest_world_cup(f, year)
    print(f"=== PD {year} ({bt['n_matches']} laga) ===")
    print(f"  bobot (clf, poisson, elo, prior): {bt['blend_weights']}")
    for k in [
        "model_blend",
        "model_classifier",
        "model_poisson",
        "baseline_elo_logistic",
        "baseline_prior",
    ]:
        m = bt[k]
        print(
            f"  {k:25s} logloss={m['log_loss']:.4f} "
            f"brier={m['brier']:.4f} acc={m['accuracy']:.3f}"
        )
    print("  MAE gol:", {k: round(v, 2) for k, v in bt["goals_mae"].items()})
