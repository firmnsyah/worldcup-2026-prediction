"""Feature engineering per pertandingan, bebas leakage.

Semua fitur dihitung HANYA dari pertandingan sebelum tanggal laga
(single pass kronologis): Elo pra-laga, form rolling, rata-rata gol,
dan rekor head-to-head.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from data_prep import compute_elo, tournament_k

FEATURE_COLS = [
    "elo_home",
    "elo_away",
    "elo_diff",
    "form5_home",
    "form5_away",
    "form10_home",
    "form10_away",
    "gf10_home",
    "ga10_home",
    "gf10_away",
    "ga10_away",
    "h2h_winrate_home",
    "h2h_matches",
    "neutral",
    "importance",
]


class TeamHistory:
    """Ringkasan rolling per tim (form & gol 10 laga terakhir)."""

    __slots__ = ("recent",)

    def __init__(self) -> None:
        self.recent: deque[tuple[int, int, int]] = deque(maxlen=10)  # (pts, gf, ga)

    def snapshot(self) -> tuple[float, float, float, float]:
        if not self.recent:
            return 1.0, 1.0, 1.25, 1.25  # prior netral utk tim tanpa riwayat
        pts = [r[0] for r in self.recent]
        gf = [r[1] for r in self.recent]
        ga = [r[2] for r in self.recent]
        form5 = float(np.mean(pts[-5:]))
        form10 = float(np.mean(pts))
        return form5, form10, float(np.mean(gf)), float(np.mean(ga))

    def add(self, gf: int, ga: int) -> None:
        pts = 3 if gf > ga else (1 if gf == ga else 0)
        self.recent.append((pts, gf, ga))


def build_match_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan kolom FEATURE_COLS + target ke DataFrame pertandingan."""
    matches, _ = compute_elo(matches)

    teams: dict[str, TeamHistory] = defaultdict(TeamHistory)
    h2h: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])  # (menang_a+0.5*seri, total)

    rows = np.empty((len(matches), 13), dtype=np.float64)
    for i, row in enumerate(matches.itertuples(index=False)):
        th, ta = teams[row.home_team], teams[row.away_team]
        f5h, f10h, gf10h, ga10h = th.snapshot()
        f5a, f10a, gf10a, ga10a = ta.snapshot()

        key = (row.home_team, row.away_team)
        wins_h, n = h2h[key]
        winrate = wins_h / n if n > 0 else 0.5

        rows[i] = (
            f5h, f5a, f10h, f10a,
            gf10h, ga10h, gf10a, ga10a,
            winrate, min(n, 10.0),
            float(row.neutral),
            tournament_k(row.tournament),
            0.0,  # placeholder elo_diff, diisi vektor di bawah
        )

        # update state SETELAH fitur dicatat (anti-leakage)
        th.add(row.home_score, row.away_score)
        ta.add(row.away_score, row.home_score)
        if row.home_score > row.away_score:
            res_h = 1.0
        elif row.home_score == row.away_score:
            res_h = 0.5
        else:
            res_h = 0.0
        h2h[key][0] += res_h
        h2h[key][1] += 1.0
        rev = (row.away_team, row.home_team)
        h2h[rev][0] += 1.0 - res_h
        h2h[rev][1] += 1.0

    out = matches.copy()
    cols = [
        "form5_home", "form5_away", "form10_home", "form10_away",
        "gf10_home", "ga10_home", "gf10_away", "ga10_away",
        "h2h_winrate_home", "h2h_matches", "neutral", "importance", "_pad",
    ]
    for j, c in enumerate(cols):
        out[c] = rows[:, j]
    out = out.drop(columns="_pad")
    out["elo_diff"] = (
        out["elo_home"] + 100.0 * (1.0 - out["neutral"]) - out["elo_away"]
    )

    # target
    out["outcome"] = np.select(
        [out["home_score"] > out["away_score"], out["home_score"] == out["away_score"]],
        [0, 1],
        default=2,
    )  # 0=menang kandang, 1=seri, 2=menang tandang
    return out


def feature_row(
    elo_home: float,
    elo_away: float,
    snap_home: tuple[float, float, float, float],
    snap_away: tuple[float, float, float, float],
    h2h_winrate: float,
    h2h_n: float,
    neutral: bool,
    importance: float = 60.0,
) -> np.ndarray:
    """Satu baris fitur (urutan = FEATURE_COLS) untuk prediksi laga baru."""
    f5h, f10h, gf10h, ga10h = snap_home
    f5a, f10a, gf10a, ga10a = snap_away
    elo_diff = elo_home + (0.0 if neutral else 100.0) - elo_away
    return np.array(
        [
            elo_home, elo_away, elo_diff,
            f5h, f5a, f10h, f10a,
            gf10h, ga10h, gf10a, ga10a,
            h2h_winrate, min(h2h_n, 10.0),
            float(neutral), importance,
        ],
        dtype=np.float64,
    )


def time_decay_weights(dates: pd.Series, cutoff: pd.Timestamp, half_life_years: float = 10.0) -> np.ndarray:
    """Bobot sampel meluruh eksponensial: laga lama berpengaruh lebih kecil."""
    years_ago = (cutoff - dates).dt.days / 365.25
    return np.power(0.5, years_ago.to_numpy() / half_life_years)


def final_team_states(matches: pd.DataFrame):
    """State terkini semua tim setelah seluruh pertandingan dimainkan.

    Mengembalikan (elo: dict, snapshots: dict team->tuple form,
    h2h: dict (a,b)->[wins_a_plus_half_draw, total]).
    Dipakai untuk membangun fitur prediksi laga PD 2026.
    """
    _, elo = compute_elo(matches)
    teams: dict[str, TeamHistory] = defaultdict(TeamHistory)
    h2h: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for row in matches.itertuples(index=False):
        teams[row.home_team].add(row.home_score, row.away_score)
        teams[row.away_team].add(row.away_score, row.home_score)
        if row.home_score > row.away_score:
            res_h = 1.0
        elif row.home_score == row.away_score:
            res_h = 0.5
        else:
            res_h = 0.0
        h2h[(row.home_team, row.away_team)][0] += res_h
        h2h[(row.home_team, row.away_team)][1] += 1.0
        h2h[(row.away_team, row.home_team)][0] += 1.0 - res_h
        h2h[(row.away_team, row.home_team)][1] += 1.0
    snapshots = {t: th.snapshot() for t, th in teams.items()}
    return elo, snapshots, h2h
