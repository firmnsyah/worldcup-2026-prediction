"""Simulasi Monte Carlo Piala Dunia 2026 dengan bracket resmi FIFA.

Format 2026: 12 grup (A-L) berisi 4 tim. Juara & runner-up grup plus
8 peringkat-3 terbaik lolos ke Babak 32 Besar. Template bracket dan
slot peringkat-3 mengikuti jadwal resmi FIFA (laga 73-104).

Sampling skor memakai grid Poisson yang di-reweight agar massa
probabilitas Menang/Seri/Kalah sesuai probabilitas ensemble
(XGB classifier terkalibrasi + Poisson + Elo-logistik, bobot tuned).
Rating dibekukan pada awal turnamen (praktik umum simulator publik).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from data_prep import (
    HOST_NATIONS,
    TOURNAMENT_START,
    load_fixtures_2026,
    load_groups,
    load_results,
    shootout_win_rates,
)
from features import feature_row, final_team_states
from models import (
    MAX_GOALS,
    OUTPUT_DIR,
    ensemble_probs,
    load_models,
    predict_lambdas,
)

GRID = MAX_GOALS + 1  # 11 -> grid skor 11x11 = 121 sel

# Babak 32 Besar resmi FIFA. W=juara grup, R=runner-up, T=peringkat-3
# (himpunan grup asal yang diizinkan per slot, sesuai Annex C FIFA).
R32_TEMPLATE: dict[int, tuple] = {
    73: (("R", "A"), ("R", "B")),
    74: (("W", "E"), ("T", frozenset("ABCDF"))),
    75: (("W", "F"), ("R", "C")),
    76: (("W", "C"), ("R", "F")),
    77: (("W", "I"), ("T", frozenset("CDFGH"))),
    78: (("R", "E"), ("R", "I")),
    79: (("W", "A"), ("T", frozenset("CEFHI"))),
    80: (("W", "L"), ("T", frozenset("EHIJK"))),
    81: (("W", "D"), ("T", frozenset("BEFIJ"))),
    82: (("W", "G"), ("T", frozenset("AEHIJ"))),
    83: (("R", "K"), ("R", "L")),
    84: (("W", "H"), ("R", "J")),
    85: (("W", "B"), ("T", frozenset("EFGIJ"))),
    86: (("W", "J"), ("R", "H")),
    87: (("W", "K"), ("T", frozenset("DEIJL"))),
    88: (("R", "D"), ("R", "G")),
}
R16_TEMPLATE = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
                93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
QF_TEMPLATE = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
SF_TEMPLATE = {101: (97, 98), 102: (99, 100)}
THIRD_SLOTS = {m: spec[1][1] for m, spec in R32_TEMPLATE.items() if spec[1][0] == "T"}

ROUNDS = ["32 Besar", "16 Besar", "Perempat Final", "Semifinal", "Final", "Juara"]


@dataclass
class SimInputs:
    teams: list[str]
    idx: dict[str, int]
    group_of: list[str]                      # huruf grup per indeks tim
    groups: dict[str, list[int]]             # grup -> 4 indeks tim
    fixtures: list[tuple[int, int]]          # 72 laga grup (indeks home, away)
    fixture_meta: pd.DataFrame
    group_cdf: np.ndarray                    # (72, 121) CDF skor per laga grup
    group_grid: np.ndarray                   # (72, 121) PMF skor (reweighted)
    group_probs: np.ndarray                  # (72, 3) blended W/D/L
    group_lambdas: np.ndarray                # (72, 2)
    ko_cdf: dict[tuple[int, int], np.ndarray]
    ko_lambdas: dict[tuple[int, int], tuple[float, float]]
    ko_probs: dict[tuple[int, int], np.ndarray]
    pens_home: dict[tuple[int, int], float]  # prob tim "home" menang adu penalti
    elo: dict[str, float] = field(default_factory=dict)


def _reweight_grid(grid_flat: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Skala massa tiap region (menang/seri/kalah) grid agar = probabilitas target."""
    g = grid_flat.reshape(GRID, GRID).copy()
    diag = np.eye(GRID, dtype=bool)
    home_win = np.tril(np.ones((GRID, GRID), dtype=bool), -1)  # gol home > away
    away_win = np.triu(np.ones((GRID, GRID), dtype=bool), 1)
    for mask, t in ((home_win, target[0]), (diag, target[1]), (away_win, target[2])):
        mass = g[mask].sum()
        if mass > 1e-12:
            g[mask] *= t / mass
    g /= g.sum()
    return g.ravel()


def _batch_grids(models: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PMF skor reweighted + blended probs + lambda untuk batch fitur."""
    lh, la = predict_lambdas(models, X)
    p_blend = ensemble_probs(models, X)

    k = np.arange(GRID)
    log_fact = np.cumsum(np.log(np.maximum(k, 1)))
    ph = np.exp(-lh[:, None] + k * np.log(lh[:, None]) - log_fact)
    pa = np.exp(-la[:, None] + k * np.log(la[:, None]) - log_fact)
    ph /= ph.sum(axis=1, keepdims=True)
    pa /= pa.sum(axis=1, keepdims=True)
    grids = (ph[:, :, None] * pa[:, None, :]).reshape(len(X), -1)
    grids = np.stack([_reweight_grid(grids[i], p_blend[i]) for i in range(len(X))])
    return grids, p_blend, np.column_stack([lh, la])


def build_sim_inputs(models: dict | None = None) -> SimInputs:
    """Siapkan semua tabel prediksi (sekali saja, lalu simulasi murni numpy)."""
    if models is None:
        models = load_models()
    results = load_results()
    results = results[results["date"] < TOURNAMENT_START]
    elo, snaps, h2h = final_team_states(results)
    sw = shootout_win_rates()

    groups_df = load_groups()
    teams = sorted(groups_df["team"])
    idx = {t: i for i, t in enumerate(teams)}
    group_of = [""] * 48
    groups: dict[str, list[int]] = {}
    for row in groups_df.itertuples(index=False):
        groups.setdefault(row.group, []).append(idx[row.team])
        group_of[idx[row.team]] = row.group

    def default_snap(t: str):
        return snaps.get(t, (1.0, 1.0, 1.25, 1.25))

    def fila(home: str, away: str, neutral: bool) -> np.ndarray:
        wins, n = h2h.get((home, away), (0.0, 0.0))
        wr = wins / n if n > 0 else 0.5
        return feature_row(
            elo.get(home, 1500.0), elo.get(away, 1500.0),
            default_snap(home), default_snap(away), wr, n, neutral,
        )

    # --- 72 laga fase grup (venue/neutral sesuai data fixture) ---
    fx = load_fixtures_2026()
    fixtures = [(idx[r.home_team], idx[r.away_team]) for r in fx.itertuples(index=False)]
    Xg = np.stack([
        fila(r.home_team, r.away_team, bool(r.neutral))
        for r in fx.itertuples(index=False)
    ])
    g_grid, g_probs, g_lam = _batch_grids(models, Xg)
    g_cdf = np.cumsum(g_grid, axis=1)

    # --- tabel knockout utk semua pasangan yang mungkin ---
    # Orientasi: jika tepat satu tim adalah tuan rumah -> tuan rumah jadi
    # "home" (neutral=False); selain itu pasangan (i<j) netral.
    pair_keys: list[tuple[int, int, bool]] = []
    for i, j in itertools.combinations(range(48), 2):
        hi, hj = teams[i] in HOST_NATIONS, teams[j] in HOST_NATIONS
        if hi and not hj:
            pair_keys.append((i, j, False))
        elif hj and not hi:
            pair_keys.append((j, i, False))
        else:
            pair_keys.append((i, j, True))
    Xk = np.stack([fila(teams[a], teams[b], neu) for a, b, neu in pair_keys])
    k_grid, k_probs, k_lam = _batch_grids(models, Xk)
    k_cdf_all = np.cumsum(k_grid, axis=1)

    ko_cdf, ko_lambdas, ko_probs, pens = {}, {}, {}, {}
    for r, (a, b, _neu) in enumerate(pair_keys):
        ko_cdf[(a, b)] = k_cdf_all[r]
        ko_lambdas[(a, b)] = (float(k_lam[r, 0]), float(k_lam[r, 1]))
        ko_probs[(a, b)] = k_probs[r]
        sa = sw.get(teams[a], 0.5)
        sb = sw.get(teams[b], 0.5)
        tilt = np.clip((elo.get(teams[a], 1500) - elo.get(teams[b], 1500)) / 4000, -0.08, 0.08)
        pens[(a, b)] = float(np.clip(0.5 + 0.3 * (sa - sb) + tilt, 0.2, 0.8))

    return SimInputs(
        teams=teams, idx=idx, group_of=group_of, groups=groups,
        fixtures=fixtures, fixture_meta=fx,
        group_cdf=g_cdf, group_grid=g_grid, group_probs=g_probs, group_lambdas=g_lam,
        ko_cdf=ko_cdf, ko_lambdas=ko_lambdas, ko_probs=ko_probs, pens_home=pens,
        elo={t: elo.get(t, 1500.0) for t in teams},
    )


# ---------------------------------------------------------------------------
# Alokasi peringkat-3 ke slot bracket (perfect matching, di-cache per kombinasi)
# ---------------------------------------------------------------------------

_third_cache: dict[frozenset, dict[int, str] | None] = {}


def assign_thirds(qualified_groups: list[str]) -> dict[int, str]:
    """Petakan 8 grup peringkat-3 lolos -> slot laga, hormati himpunan izin."""
    key = frozenset(qualified_groups)
    if key in _third_cache:
        cached = _third_cache[key]
        if cached is not None:
            return cached
    slots = sorted(THIRD_SLOTS, key=lambda m: len(THIRD_SLOTS[m] & key))
    assignment: dict[int, str] = {}

    def backtrack(si: int, remaining: set[str]) -> bool:
        if si == len(slots):
            return True
        m = slots[si]
        for g in sorted(THIRD_SLOTS[m] & remaining):
            assignment[m] = g
            if backtrack(si + 1, remaining - {g}):
                return True
            del assignment[m]
        return False

    if not backtrack(0, set(key)):
        # fallback (kombinasi tanpa matching sempurna): isi greedy
        remaining = list(qualified_groups)
        for m in slots:
            pick = next((g for g in remaining if g in THIRD_SLOTS[m]), remaining[0])
            assignment[m] = pick
            remaining.remove(pick)
    _third_cache[key] = dict(assignment)
    return assignment


# ---------------------------------------------------------------------------
# Mesin simulasi
# ---------------------------------------------------------------------------

def _play_knockout(si: SimInputs, a: int, b: int, rng: np.random.Generator) -> int:
    """Mainkan satu laga knockout, kembalikan indeks pemenang."""
    key = (a, b) if (a, b) in si.ko_cdf else (b, a)
    h, w = key[0], key[1]
    cell = int(np.searchsorted(si.ko_cdf[key], rng.random()))
    gh, ga = cell // GRID, cell % GRID
    if gh != ga:
        return h if gh > ga else w
    lh, la = si.ko_lambdas[key]
    eh, ea = rng.poisson(lh / 3.0), rng.poisson(la / 3.0)
    if eh != ea:
        return h if eh > ea else w
    return h if rng.random() < si.pens_home[key] else w


def run_simulation(si: SimInputs, n_sims: int = 10000, seed: int = 42) -> dict:
    """Jalankan Monte Carlo; kembalikan agregat probabilitas."""
    rng = np.random.default_rng(seed)
    n_teams = 48
    champion = np.zeros(n_teams)
    reach = np.zeros((n_teams, len(ROUNDS)))
    group_pts = np.zeros(n_teams)
    group_pos = np.zeros((n_teams, 4))
    qualify = np.zeros(n_teams)

    # sampel semua skor fase grup sekaligus: (n_sims, 72)
    u = rng.random((n_sims, 72))
    cells = np.empty((n_sims, 72), dtype=np.int64)
    for m in range(72):
        cells[:, m] = np.searchsorted(si.group_cdf[m], u[:, m])
    goals_h = cells // GRID
    goals_a = cells % GRID

    group_letters = sorted(si.groups)
    for s in range(n_sims):
        pts = np.zeros(n_teams, dtype=np.int64)
        gd = np.zeros(n_teams, dtype=np.int64)
        gf = np.zeros(n_teams, dtype=np.int64)
        for m, (hm, aw) in enumerate(si.fixtures):
            gh, ga = goals_h[s, m], goals_a[s, m]
            gf[hm] += gh
            gf[aw] += ga
            gd[hm] += gh - ga
            gd[aw] += ga - gh
            if gh > ga:
                pts[hm] += 3
            elif gh < ga:
                pts[aw] += 3
            else:
                pts[hm] += 1
                pts[aw] += 1

        winners: dict[str, int] = {}
        runners: dict[str, int] = {}
        thirds: list[tuple] = []
        for g in group_letters:
            members = si.groups[g]
            order = sorted(
                members,
                key=lambda t: (pts[t], gd[t], gf[t], rng.random()),
                reverse=True,
            )
            winners[g], runners[g] = order[0], order[1]
            thirds.append((pts[order[2]], gd[order[2]], gf[order[2]], rng.random(), order[2], g))
            for pos, t in enumerate(order):
                group_pos[t, pos] += 1
        group_pts += pts

        thirds.sort(reverse=True)
        best8 = thirds[:8]
        third_team = {g: t for *_x, t, g in best8}
        slot_assign = assign_thirds([g for *_x, _t, g in best8])

        advancing = list(winners.values()) + list(runners.values()) + list(third_team.values())
        qualify[advancing] += 1
        reach[advancing, 0] += 1

        # Babak 32 Besar
        match_winner: dict[int, int] = {}
        for m, (sh, sa) in R32_TEMPLATE.items():
            def resolve(spec):
                kind, val = spec
                if kind == "W":
                    return winners[val]
                if kind == "R":
                    return runners[val]
                return third_team[slot_assign[m]]
            a, b = resolve(sh), resolve(sa)
            match_winner[m] = _play_knockout(si, a, b, rng)

        for tmpl, round_i in ((R16_TEMPLATE, 1), (QF_TEMPLATE, 2), (SF_TEMPLATE, 3)):
            nxt: dict[int, int] = {}
            for m, (m1, m2) in tmpl.items():
                a, b = match_winner[m1], match_winner[m2]
                reach[[a, b], round_i] += 1
                nxt[m] = _play_knockout(si, a, b, rng)
            match_winner = {**match_winner, **nxt}

        fa, fb = match_winner[101], match_winner[102]
        reach[[fa, fb], 4] += 1
        champ = _play_knockout(si, fa, fb, rng)
        reach[champ, 5] += 1
        champion[champ] += 1

    return {
        "n_sims": n_sims,
        "champion": champion / n_sims,
        "reach": reach / n_sims,
        "group_pts": group_pts / n_sims,
        "group_pos": group_pos / n_sims,
        "qualify": qualify / n_sims,
    }


# ---------------------------------------------------------------------------
# Bracket modal (jalur paling mungkin) — playthrough deterministik
# ---------------------------------------------------------------------------

def modal_bracket(si: SimInputs) -> dict:
    exp_pts = np.zeros(48)
    exp_gd = np.zeros(48)
    for m, (hm, aw) in enumerate(si.fixtures):
        ph, pd_, pa = si.group_probs[m]
        exp_pts[hm] += 3 * ph + pd_
        exp_pts[aw] += 3 * pa + pd_
        lam_h, lam_a = si.group_lambdas[m]
        exp_gd[hm] += lam_h - lam_a
        exp_gd[aw] += lam_a - lam_h

    winners, runners, thirds = {}, {}, []
    for g in sorted(si.groups):
        order = sorted(si.groups[g], key=lambda t: (exp_pts[t], exp_gd[t]), reverse=True)
        winners[g], runners[g] = order[0], order[1]
        thirds.append((exp_pts[order[2]], exp_gd[order[2]], order[2], g))
    thirds.sort(reverse=True)
    best8 = thirds[:8]
    third_team = {g: t for *_x, t, g in best8}
    slot_assign = assign_thirds([g for *_x, _t, g in best8])

    def advance_prob(a: int, b: int) -> float:
        key = (a, b) if (a, b) in si.ko_probs else (b, a)
        ph, pd_, pa = si.ko_probs[key]
        p_first = ph + pd_ * si.pens_home[key]  # aproksimasi: seri -> ET/pen
        return p_first if key[0] == a else 1.0 - p_first

    bracket: dict[str, list] = {r: [] for r in ["32 Besar", "16 Besar", "Perempat Final", "Semifinal", "Final"]}
    match_winner: dict[int, int] = {}

    for m, (sh, sa) in R32_TEMPLATE.items():
        def resolve(spec):
            kind, val = spec
            if kind == "W":
                return winners[val]
            if kind == "R":
                return runners[val]
            return third_team[slot_assign[m]]
        a, b = resolve(sh), resolve(sa)
        p = advance_prob(a, b)
        w = a if p >= 0.5 else b
        match_winner[m] = w
        bracket["32 Besar"].append(
            {"match": m, "home": si.teams[a], "away": si.teams[b],
             "winner": si.teams[w], "p_winner": round(max(p, 1 - p), 3)}
        )

    for tmpl, rname in ((R16_TEMPLATE, "16 Besar"), (QF_TEMPLATE, "Perempat Final"),
                        (SF_TEMPLATE, "Semifinal"), ({104: (101, 102)}, "Final")):
        for m, (m1, m2) in tmpl.items():
            a, b = match_winner[m1], match_winner[m2]
            p = advance_prob(a, b)
            w = a if p >= 0.5 else b
            match_winner[m] = w
            bracket[rname].append(
                {"match": m, "home": si.teams[a], "away": si.teams[b],
                 "winner": si.teams[w], "p_winner": round(max(p, 1 - p), 3)}
            )
    return bracket


def export_predictions(si: SimInputs, agg: dict, path=None) -> dict:
    """Susun seluruh hasil menjadi predictions.json untuk dashboard."""
    teams = si.teams
    rounds_dict = {
        t: {ROUNDS[r]: round(float(agg["reach"][i, r]) * 100, 2) for r in range(len(ROUNDS))}
        for i, t in enumerate(teams)
    }

    groups_out = {}
    for g in sorted(si.groups):
        rows = []
        for t in sorted(si.groups[g], key=lambda t: agg["group_pts"][t], reverse=True):
            rows.append({
                "tim": teams[t],
                "exp_poin": round(float(agg["group_pts"][t]), 2),
                "p_juara_grup": round(float(agg["group_pos"][t, 0]) * 100, 1),
                "p_runner_up": round(float(agg["group_pos"][t, 1]) * 100, 1),
                "p_lolos": round(float(agg["qualify"][t]) * 100, 1),
                "elo": round(si.elo[teams[t]]),
            })
        groups_out[g] = rows

    matches_out = []
    for m, meta in enumerate(si.fixture_meta.itertuples(index=False)):
        grid = si.group_grid[m].reshape(GRID, GRID)
        h, a = np.unravel_index(np.argmax(grid), grid.shape)
        ph, pd_, pa = si.group_probs[m]
        matches_out.append({
            "tanggal": str(pd.Timestamp(meta.date).date()),
            "grup": meta.group,
            "home": meta.home_team,
            "away": meta.away_team,
            "p_home": round(float(ph) * 100, 1),
            "p_seri": round(float(pd_) * 100, 1),
            "p_away": round(float(pa) * 100, 1),
            "skor": f"{int(h)}-{int(a)}",
            "p_skor": round(float(grid[h, a]) * 100, 1),
            "xg_home": round(float(si.group_lambdas[m, 0]), 2),
            "xg_away": round(float(si.group_lambdas[m, 1]), 2),
        })

    out = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_sims": agg["n_sims"],
        "champion": {
            teams[i]: round(float(agg["champion"][i]) * 100, 2)
            for i in np.argsort(-agg["champion"])
        },
        "rounds": rounds_dict,
        "groups": groups_out,
        "matches": matches_out,
        "modal_bracket": modal_bracket(si),
        "team_group": {teams[i]: si.group_of[i] for i in range(48)},
        "elo": {t: round(e) for t, e in si.elo.items()},
    }
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = path or OUTPUT_DIR / "predictions.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    si = build_sim_inputs()
    agg = run_simulation(si, n_sims=10000)
    out = export_predictions(si, agg)
    top = list(out["champion"].items())[:10]
    print("Top 10 kandidat juara:")
    for t, p in top:
        print(f"  {t:20s} {p:5.2f}%")
