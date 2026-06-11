# FIFA World Cup Dataset 1930–2026
## A Comprehensive Historical + Predictive Dataset for the 2026 FIFA World Cup

**Created:** June 2026 | **Coverage:** All 22 World Cups (1930–2022) + Full 2026 Snapshot  
**Kaggle Use Cases:** EDA, Time-Series Analysis, Match Prediction, Tournament Simulation, ML Modeling

---

## Files Overview

### 1. `wc_tournaments.csv`
One row per World Cup edition (1930–2022). Contains host, champion, runner-up, top scorer, total goals, attendance, and format era.  
**Rows:** 22 | **Key columns:** `wc_year`, `champion`, `total_goals`, `format_era`

### 2. `wc_team_appearances.csv`
Panel dataset — one row per (team × WC year). Covers qualified teams only (non-qualifiers excluded for cleanliness). Includes participation status, stage reached, W/D/L/GF/GA, ELO rating, FIFA ranking (where available).  
**Rows:** ~350 | **Key columns:** `team`, `wc_year`, `participation_status`, `final_stage_reached`, `elo_rating_approx`

> **Note on ELO:** FIFA rankings started in 1993. For pre-1993 tournaments, `elo_rating_approx` (World Football Elo Ratings) is used — a widely accepted proxy that goes back to the 1870s.

### 3. `wc_matches_historical.csv`
Key matches from every World Cup: all finals, semi-finals, quarter-finals, and notable group stage matches.  
**Rows:** 100+ | **Key columns:** `match_id`, `wc_year`, `stage`, `home_team`, `away_team`, `home_goals`, `away_goals`, `aet`, `penalties`, `winning_team`, `home_pre_match_elo`, `away_pre_match_elo`

### 4. `wc_team_alltime_stats.csv`
Aggregated career statistics for every nation that has appeared at a World Cup.  
**Rows:** 80+ nations | **Key columns:** `total_wc_appearances`, `win_rate`, `best_finish`, `titles`, `elo_peak_approx`

### 5. `wc_2026_groups.csv`
Complete 2026 tournament group stage draw with all 48 teams, their groups (A–L), FIFA rankings, squad market values, key players, and contextual notes.  
**Rows:** 48 | **Key columns:** `group`, `team`, `fifa_rank_apr2026`, `squad_market_value_eur_millions`, `key_player`

### 6. `wc_2026_teams_snapshot.csv`
2026 team snapshot combining tournament context: qualification method, debut status, host flags, FIFA rank, best WC finish, and analytical notes per team.  
**Rows:** 48 | **Key columns:** `team`, `confederation`, `fifa_rank_apr2026`, `best_wc_finish`, `is_debut`, `notes`

### 7. `wc_prediction_features_2026.csv`
ML-ready feature matrix for all 48 teams. Designed for tournament simulation models. Includes FIFA rank, ELO, squad age, market value, recent form, qualifying stats, host flag, and a pre-computed win probability baseline from Poisson/ELO simulation.  
**Rows:** 48 | **Key columns:** `elo_rating_2026`, `squad_market_value_eur_m`, `recent_form_pts_last10`, `prediction_win_probability_pct`

---

## Suggested Notebook Projects

### Beginner
- EDA: Historical goals per WC edition — did football become more defensive?
- Which confederation dominates? Win rate by confederation over time
- Home advantage: do hosts outperform their ELO expectation?

### Intermediate
- Predict match outcomes using ELO difference + logistic regression
- Feature importance: what matters most — ranking, market value, or experience?
- Simulate 2026 group stage based on pre-tournament features

### Advanced
- Poisson regression for scoreline prediction (home/away goals as separate Poisson distributions)
- Full bracket simulation: 10,000 Monte Carlo runs → win probability per team
- Time-series ELO evolution: track rise/fall of nations 1930–2026

---

## Data Sources & Methodology

| Source | Used For |
|--------|----------|
| Wikipedia (FIFA WC pages) | Match results, squad lists, tournament stats |
| worldcupwiki.com | 2026 qualified teams, groups, rankings |
| ESPN/Yahoo Sports | 2026 group draw confirmation |
| World Football Elo Ratings (eloratings.net methodology) | Pre-1993 team strength proxy |
| FIFA official rankings (April 1, 2026) | Current team rankings |
| Transfermarkt methodology | Squad market values (estimates for pre-2002) |

---

## Known Limitations

- Match-level data focuses on knockouts + selected group stage; not every group stage match from 1930–1990 is included (future work)
- Squad market values pre-2002 are estimates based on historical equivalence
- FIFA rankings are unavailable before 1993; ELO used as substitute
- 2026 prediction probabilities are pre-tournament estimates based on ELO + recent form; will diverge as tournament progresses

---

## 2026 Quick Reference

| Detail | Info |
|--------|------|
| Dates | June 11 – July 19, 2026 |
| Hosts | USA, Canada, Mexico |
| Teams | 48 (first ever expanded format) |
| Groups | 12 groups of 4 |
| Total matches | 104 |
| Final venue | MetLife Stadium, New Jersey |
| Opening match | Mexico vs South Africa, Estadio Azteca |
| Defending champion | Argentina |
| Top ELO-rated | France (1990), Spain (1990), Netherlands (1940) |
| Prediction favourite | France / Spain |

---

## License
Public domain data compiled for educational and research use.  
Please cite sources above if publishing work based on this dataset.
