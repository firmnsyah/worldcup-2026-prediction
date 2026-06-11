"""Data loading, normalisasi nama tim, dan Elo rating engine.

Sumber data:
- results.csv  : 49 ribu+ pertandingan internasional 1872-2026 (training utama).
  Baris dengan skor kosong = 72 fixture fase grup Piala Dunia 2026.
- wc_2026_groups.csv : 48 tim peserta dan pembagian grup A-L.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_CSV = (
    BASE_DIR / "International football results from 1872 to 2026" / "results.csv"
)
SHOOTOUTS_CSV = (
    BASE_DIR / "International football results from 1872 to 2026" / "shootouts.csv"
)
GROUPS_CSV = (
    BASE_DIR / "FIFA World Cup 1930-2026 Ultimate ML Dataset" / "wc_2026_groups.csv"
)
RANKING_CSV = (
    BASE_DIR / "Football - FIFA World Cup, 1930 - 2026" / "fifa_ranking_2026-06-08.csv"
)
ULTIMATE_DIR = BASE_DIR / "FIFA World Cup 1930-2026 Ultimate ML Dataset"
SNAPSHOT_CSV = ULTIMATE_DIR / "wc_2026_teams_snapshot.csv"
COACHES_CSV = ULTIMATE_DIR / "wc_coaches_2026.csv"
ALLTIME_CSV = ULTIMATE_DIR / "wc_team_alltime_stats.csv"
GOALSCORERS_CSV = (
    BASE_DIR / "International football results from 1872 to 2026" / "goalscorers.csv"
)
WC_MATCHES_CSV = (
    BASE_DIR / "Football - FIFA World Cup, 1930 - 2026" / "matches_1930_2022.csv"
)

# Nama kanonik mengikuti konvensi results.csv.
NAME_MAP = {
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Congo DR": "DR Congo",
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "USA": "United States",
    "China PR": "China",
    "Cabo Verde": "Cape Verde",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Brunei Darussalam": "Brunei",
    "Hong Kong, China": "Hong Kong",
    "Chinese Taipei": "Taiwan",
    "The Gambia": "Gambia",
    "St Kitts and Nevis": "Saint Kitts and Nevis",
    "St Lucia": "Saint Lucia",
    "St Vincent and the Grenadines": "Saint Vincent and the Grenadines",
    "US Virgin Islands": "United States Virgin Islands",
}

HOST_NATIONS = {"United States", "Canada", "Mexico"}

TOURNAMENT_START = pd.Timestamp("2026-06-11")


def canon(name: str) -> str:
    return NAME_MAP.get(name, name)


def load_results() -> pd.DataFrame:
    """Seluruh pertandingan yang sudah dimainkan, urut kronologis."""
    df = pd.read_csv(RESULTS_CSV, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["home_score"].notna()].copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_fixtures_2026() -> pd.DataFrame:
    """72 fixture fase grup PD 2026 (baris results.csv tanpa skor) + info grup."""
    df = pd.read_csv(RESULTS_CSV, encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"])
    fx = df[df["home_score"].isna()].copy()
    assert len(fx) == 72, f"Diharapkan 72 fixture, dapat {len(fx)}"

    groups = load_groups()
    team_group = dict(zip(groups["team"], groups["group"]))
    fx["group"] = fx["home_team"].map(team_group)
    missing = fx[fx["group"].isna()]
    assert missing.empty, f"Tim tanpa grup: {missing.home_team.unique()}"
    assert (
        fx["home_team"].map(team_group) == fx["away_team"].map(team_group)
    ).all(), "Pasangan laga grup tidak konsisten dengan pembagian grup"
    return fx.sort_values("date").reset_index(drop=True)


def load_groups() -> pd.DataFrame:
    """48 tim peserta dengan grup, dinormalisasi ke nama kanonik."""
    g = pd.read_csv(GROUPS_CSV, encoding="utf-8")
    g["team"] = g["team"].map(canon)
    assert len(g) == 48 and g["team"].nunique() == 48
    return g


def load_fifa_ranking() -> pd.DataFrame:
    r = pd.read_csv(RANKING_CSV, encoding="utf-8")
    r["team"] = r["team"].map(canon)
    return r


def load_team_snapshot_2026() -> pd.DataFrame:
    """Snapshot 48 peserta 2026: ranking FIFA, sejarah PD, metode kualifikasi."""
    s = pd.read_csv(SNAPSHOT_CSV, encoding="utf-8")
    s["team"] = s["team"].map(canon)
    return s


def load_coaches_2026() -> pd.DataFrame:
    """Profil pelatih 48 tim peserta 2026."""
    c = pd.read_csv(COACHES_CSV, encoding="utf-8")
    c["team"] = c["team"].map(canon)
    return c


def load_alltime_stats() -> pd.DataFrame:
    """Statistik all-time Piala Dunia per tim."""
    a = pd.read_csv(ALLTIME_CSV, encoding="utf-8")
    a["team"] = a["team"].map(canon)
    return a


def load_goalscorers() -> pd.DataFrame:
    """Data pencetak gol per pertandingan internasional (1916-2026)."""
    g = pd.read_csv(GOALSCORERS_CSV, encoding="utf-8")
    g["team"] = g["team"].map(canon)
    return g


# ---------------------------------------------------------------------------
# Elo rating engine (mengikuti metodologi eloratings.net)
# ---------------------------------------------------------------------------

CONTINENTAL_FINALS = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "CONCACAF Championship",
    "Oceania Nations Cup",
    "Confederations Cup",
}

HOME_ADVANTAGE_ELO = 100.0
INITIAL_ELO = 1500.0


def tournament_k(tournament: str) -> float:
    """K-factor sesuai tingkat kepentingan turnamen."""
    if tournament == "FIFA World Cup":
        return 60.0
    if tournament in CONTINENTAL_FINALS:
        return 50.0
    if "qualification" in tournament or "Nations League" in tournament:
        return 40.0
    if tournament == "Friendly":
        return 20.0
    return 30.0


def goal_margin_multiplier(diff: int) -> float:
    d = abs(diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11.0 + d) / 8.0


def elo_expected(elo_home: float, elo_away: float, neutral: bool) -> float:
    adv = 0.0 if neutral else HOME_ADVANTAGE_ELO
    return 1.0 / (1.0 + 10.0 ** (-(elo_home + adv - elo_away) / 400.0))


def elo_update(
    elo_home: float,
    elo_away: float,
    home_score: int,
    away_score: int,
    tournament: str,
    neutral: bool,
) -> tuple[float, float]:
    """Kembalikan (elo_home_baru, elo_away_baru) setelah satu pertandingan."""
    exp_home = elo_expected(elo_home, elo_away, neutral)
    if home_score > away_score:
        actual = 1.0
    elif home_score == away_score:
        actual = 0.5
    else:
        actual = 0.0
    k = tournament_k(tournament) * goal_margin_multiplier(home_score - away_score)
    delta = k * (actual - exp_home)
    return elo_home + delta, elo_away - delta


def compute_elo(matches: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Hitung Elo kronologis.

    Mengembalikan (DataFrame dengan kolom elo_home/elo_away PRA-pertandingan,
    dict rating akhir per tim). Kolom pra-pertandingan bebas leakage karena
    hanya memakai hasil sebelum laga tersebut.
    """
    ratings: dict[str, float] = {}
    elo_home_col = []
    elo_away_col = []
    for row in matches.itertuples(index=False):
        eh = ratings.get(row.home_team, INITIAL_ELO)
        ea = ratings.get(row.away_team, INITIAL_ELO)
        elo_home_col.append(eh)
        elo_away_col.append(ea)
        new_eh, new_ea = elo_update(
            eh, ea, row.home_score, row.away_score, row.tournament, bool(row.neutral)
        )
        ratings[row.home_team] = new_eh
        ratings[row.away_team] = new_ea
    out = matches.copy()
    out["elo_home"] = elo_home_col
    out["elo_away"] = elo_away_col
    return out, ratings


def shootout_win_rates() -> dict[str, float]:
    """Win-rate adu penalti historis per tim (smoothing Bayesian ke 0.5)."""
    s = pd.read_csv(SHOOTOUTS_CSV, encoding="utf-8")
    stats: dict[str, list[int]] = {}
    for row in s.itertuples(index=False):
        for team in (row.home_team, row.away_team):
            stats.setdefault(team, [0, 0])
            stats[team][1] += 1
            if row.winner == team:
                stats[team][0] += 1
    # prior 4 shootout @50% agar tim minim data tidak ekstrem
    return {t: (w + 2.0) / (n + 4.0) for t, (w, n) in stats.items()}
