"""Dashboard Streamlit — Prediksi Piala Dunia 2026.

Jalankan:  streamlit run app.py
Membaca artefak output/predictions.json + output/metrics.json.
Tombol "Jalankan Ulang Simulasi" memakai model tersimpan di models/.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))

OUTPUT = BASE / "output"

st.set_page_config(
    page_title="Prediksi Piala Dunia 2026", page_icon="🏆", layout="wide"
)

ROUNDS = ["32 Besar", "16 Besar", "Perempat Final", "Semifinal", "Final", "Juara"]

# Warna konsisten di seluruh app
C_WIN, C_DRAW, C_LOSE = "#22c55e", "#64748b", "#3b82f6"
COLOR_SEQ = ["#22c55e", "#facc15", "#3b82f6", "#f472b6", "#a78bfa", "#fb923c"]

# Kode ISO-2 flagcdn.com untuk 48 peserta (emoji bendera tidak dirender Windows)
FLAG_ISO = {
    "Czech Republic": "cz", "Mexico": "mx", "South Africa": "za", "South Korea": "kr",
    "Bosnia and Herzegovina": "ba", "Canada": "ca", "Qatar": "qa", "Switzerland": "ch",
    "Brazil": "br", "Haiti": "ht", "Morocco": "ma", "Scotland": "gb-sct",
    "Australia": "au", "Paraguay": "py", "Turkey": "tr", "United States": "us",
    "Curaçao": "cw", "Ecuador": "ec", "Germany": "de", "Ivory Coast": "ci",
    "Japan": "jp", "Netherlands": "nl", "Sweden": "se", "Tunisia": "tn",
    "Belgium": "be", "Egypt": "eg", "Iran": "ir", "New Zealand": "nz",
    "Cape Verde": "cv", "Saudi Arabia": "sa", "Spain": "es", "Uruguay": "uy",
    "France": "fr", "Iraq": "iq", "Norway": "no", "Senegal": "sn",
    "Algeria": "dz", "Argentina": "ar", "Austria": "at", "Jordan": "jo",
    "Colombia": "co", "DR Congo": "cd", "Portugal": "pt", "Uzbekistan": "uz",
    "Croatia": "hr", "England": "gb-eng", "Ghana": "gh", "Panama": "pa",
}


def flag_url(team, w: int = 40) -> str:
    iso = FLAG_ISO.get(team)
    return f"https://flagcdn.com/w{w}/{iso}.png" if iso else ""


def flag_img(team, h: int = 16) -> str:
    url = flag_url(team)
    if not url:
        return ""
    return (
        f"<img src='{url}' height='{h}' "
        f"style='border-radius:3px;vertical-align:-2px;margin:0 6px 0 0'/>"
    )


st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem;}

.hero {
  background: radial-gradient(120% 160% at 0% 0%, #14532d 0%, #0f3460 55%, #0b1120 100%);
  border: 1px solid #1e293b; border-radius: 18px;
  padding: 26px 30px 22px; margin-bottom: 14px;
}
.hero h1 {margin: 0 0 4px; font-size: 2rem; color: #f8fafc;}
.hero .sub {color: #94a3b8; font-size: 0.92rem; margin-bottom: 12px;}
.badge {
  display: inline-block; padding: 4px 12px; border-radius: 999px;
  background: rgba(34,197,94,.12); border: 1px solid rgba(34,197,94,.35);
  color: #4ade80; font-size: 0.78rem; margin-right: 8px;
}

[data-testid="stMetric"] {
  background: #151e31; border: 1px solid #1e293b;
  border-radius: 14px; padding: 14px 18px;
}

.stTabs [data-baseweb="tab-list"] {gap: 6px;}
.stTabs [data-baseweb="tab"] {
  background: #151e31; border-radius: 10px; padding: 6px 16px;
}
.stTabs [aria-selected="true"] {background: #14532d !important;}

.pod {display: flex; gap: 14px; margin-bottom: 14px;}
.pod-card {
  flex: 1; text-align: center; background: #151e31;
  border: 1px solid #1e293b; border-radius: 16px; padding: 18px 12px 14px;
}
.pod-1 {border-top: 4px solid #facc15;}
.pod-2 {border-top: 4px solid #cbd5e1;}
.pod-3 {border-top: 4px solid #d97706;}
.pod-rank {font-size: 0.75rem; color: #94a3b8; letter-spacing: 2px;}
.pod-team {font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin: 6px 0;}
.pod-pct {font-size: 1.7rem; font-weight: 800; color: #4ade80;}

.match-card {
  background: #151e31; border: 1px solid #1e293b;
  border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;
}
.mc-row {display: flex; align-items: center; justify-content: space-between;}
.mc-team {font-weight: 600; font-size: 0.92rem; color: #e2e8f0; width: 40%;}
.mc-team.away {text-align: right;}
.mc-score {
  font-size: 1.05rem; font-weight: 800; color: #facc15;
  background: #0b1120; border-radius: 8px; padding: 2px 12px;
}
.prob-bar {display: flex; height: 7px; border-radius: 4px; overflow: hidden; margin: 9px 0 6px;}
.mc-meta {font-size: 0.74rem; color: #94a3b8;}

.bk-card {
  background: #151e31; border: 1px solid #1e293b; border-left: 3px solid #334155;
  border-radius: 9px; padding: 6px 8px; margin-bottom: 7px; font-size: 0.78rem;
}
.bk-card.final {border-left-color: #facc15;}
.bk-row {display: flex; justify-content: space-between; color: #cbd5e1; padding: 1px 0;}
.bk-row.win {color: #4ade80; font-weight: 700;}
.bk-prob {color: #94a3b8; font-size: 0.72rem;}

.champ-banner {
  background: linear-gradient(90deg, rgba(250,204,21,.14), rgba(34,197,94,.10));
  border: 1px solid rgba(250,204,21,.4); border-radius: 14px;
  padding: 16px 22px; font-size: 1.05rem; color: #fde68a; margin-top: 10px;
}

.info-card {
  background: #151e31; border: 1px solid #1e293b; border-radius: 12px;
  padding: 14px 16px; margin-bottom: 10px; font-size: 0.9rem; color: #cbd5e1;
}
.info-card b {color: #f1f5f9;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Memuat model & menyiapkan tabel prediksi...")
def get_sim_inputs():
    from simulate import build_sim_inputs

    return build_sim_inputs()


@st.cache_data(show_spinner="Menjalankan simulasi Monte Carlo...")
def simulate(n_sims: int, seed: int):
    from simulate import export_predictions, run_simulation

    si = get_sim_inputs()
    agg = run_simulation(si, n_sims=n_sims, seed=seed)
    return export_predictions(si, agg)


def load_predictions() -> dict | None:
    p = OUTPUT / "predictions.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def load_metrics() -> dict | None:
    p = OUTPUT / "metrics.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


@st.cache_data(show_spinner=False)
def load_profiles():
    """Data profil tim dari dataset snapshot 2026 (bukan fitur model)."""
    from data_prep import (
        load_alltime_stats,
        load_coaches_2026,
        load_fifa_ranking,
        load_goalscorers,
        load_groups,
        load_team_snapshot_2026,
    )

    return {
        "groups": load_groups(),
        "snapshot": load_team_snapshot_2026(),
        "coaches": load_coaches_2026(),
        "alltime": load_alltime_stats(),
        "ranking": load_fifa_ranking(),
        "goalscorers": load_goalscorers(),
    }


def match_card(m: dict, show_group: bool = True) -> str:
    """Kartu HTML satu pertandingan dengan bar probabilitas W/D/L."""
    grp = f" · Grup {m['grup']}" if show_group else ""
    return f"""
<div class="match-card">
  <div class="mc-row">
    <span class="mc-team">{flag_img(m['home'])}{m['home']}</span>
    <span class="mc-score">{m['skor']}</span>
    <span class="mc-team away">{m['away']}{flag_img(m['away'])}</span>
  </div>
  <div class="prob-bar">
    <span style="width:{m['p_home']}%;background:{C_WIN}"></span>
    <span style="width:{m['p_seri']}%;background:{C_DRAW}"></span>
    <span style="width:{m['p_away']}%;background:{C_LOSE}"></span>
  </div>
  <div class="mc-meta">{m['tanggal']}{grp} &nbsp;·&nbsp;
    <span style="color:{C_WIN}">{m['p_home']:.1f}%</span> /
    <span style="color:#94a3b8">{m['p_seri']:.1f}%</span> /
    <span style="color:{C_LOSE}">{m['p_away']:.1f}%</span>
    &nbsp;·&nbsp; prob. skor {m['p_skor']:.1f}% &nbsp;·&nbsp;
    xG {m['xg_home']:.2f}–{m['xg_away']:.2f}</div>
</div>"""


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## ⚽ Piala Dunia 2026")
    st.caption("Prediksi berbasis machine learning")
    st.divider()
    n_sims = st.slider(
        "Jumlah simulasi", min_value=1_000, max_value=50_000, value=10_000, step=1_000
    )
    if st.button("🔄 Jalankan Ulang Simulasi", use_container_width=True, type="primary"):
        st.session_state["pred"] = simulate(n_sims, seed=42)
        st.success(f"Simulasi {n_sims:,}x selesai.")
    st.divider()
    with st.expander("ℹ️ Tentang model"):
        st.markdown(
            "- **Elo dinamis** dari 49 ribu+ laga (1872–2026)\n"
            "- **XGBoost Poisson** → expected goals & skor\n"
            "- **Ensemble W/D/L** divalidasi pada PD 2018 & 2022\n"
            "- **Monte Carlo** dengan bracket resmi FIFA\n\n"
            "Lihat tab *Validasi Model* untuk bukti backtest."
        )

pred = st.session_state.get("pred") or load_predictions()

# ---------------------------------------------------------------- hero
host_flags = "".join(flag_img(t, h=20) for t in ("United States", "Canada", "Mexico"))
n_sims_txt = f"{pred['n_sims']:,}".replace(",", ".") if pred else "—"
st.markdown(
    f"""
<div class="hero">
  <h1>🏆 Prediksi Piala Dunia 2026</h1>
  <div class="sub">Elo dinamis · XGBoost Poisson · simulasi Monte Carlo dengan bracket resmi FIFA
  &nbsp;&nbsp;{host_flags}<span style="color:#94a3b8;font-size:0.85rem">11 Juni – 19 Juli 2026</span></div>
  <span class="badge">🎲 {n_sims_txt} simulasi</span>
  <span class="badge">📊 49.405 laga historis</span>
  <span class="badge">🤖 dibuat {pred['generated_at'][:16] if pred else '—'}</span>
</div>
""",
    unsafe_allow_html=True,
)

if pred is None:
    st.warning(
        "Belum ada hasil simulasi. Jalankan notebook 02 & 03 terlebih dahulu, "
        "atau klik **Jalankan Ulang Simulasi** di sidebar (butuh model tersimpan)."
    )
    st.stop()

tab_juara, tab_grup, tab_laga, tab_bracket, tab_profil, tab_validasi = st.tabs(
    [
        "🏆 Juara",
        "📊 Fase Grup",
        "⚽ Pertandingan",
        "🔮 Bracket",
        "👥 Profil Tim",
        "📈 Validasi Model",
    ]
)

# ---------------------------------------------------------------- tab juara
with tab_juara:
    champ = pd.Series(pred["champion"]).head(15)
    podium = list(champ.items())[:3]
    medals = ["🥇 FAVORIT UTAMA", "🥈 KANDIDAT KUAT", "🥉 KUDA HITAM TERATAS"]
    cards = "".join(
        f"""<div class="pod-card pod-{i + 1}">
              <div class="pod-rank">{medals[i]}</div>
              <div class="pod-team">{flag_img(t, h=20)}{t}</div>
              <div class="pod-pct">{p:.1f}%</div>
            </div>"""
        for i, (t, p) in enumerate(podium)
    )
    st.markdown(f'<div class="pod">{cards}</div>', unsafe_allow_html=True)

    fig = px.bar(
        champ.iloc[::-1],
        orientation="h",
        labels={"index": "", "value": "Probabilitas juara (%)"},
        title="Top 15 Kandidat Juara",
        color=champ.iloc[::-1].values,
        color_continuous_scale=["#1e3a5f", "#22c55e"],
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False, height=520)
    fig.update_traces(
        texttemplate="%{x:.1f}", textposition="outside", cliponaxis=False
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Probabilitas mencapai tiap babak")
    n_show = st.slider("Tampilkan berapa tim", 8, 48, 16, key="n_reach")
    reach = pd.DataFrame(pred["rounds"]).T[ROUNDS]
    reach = reach.sort_values("Juara", ascending=False).head(n_show)
    reach_df = reach.reset_index(names="Tim")
    reach_df.insert(0, " ", reach_df["Tim"].map(flag_url))
    st.dataframe(
        reach_df,
        use_container_width=True,
        hide_index=True,
        height=min(60 + 36 * n_show, 620),
        column_config={
            " ": st.column_config.ImageColumn(" ", width=40),
            **{
                r: st.column_config.ProgressColumn(
                    r, format="%.1f%%", min_value=0, max_value=100
                )
                for r in ROUNDS
            },
        },
    )

# ---------------------------------------------------------------- tab grup
with tab_grup:
    letters = sorted(pred["groups"].keys())
    g = st.pills(
        "Pilih grup",
        letters,
        format_func=lambda x: f"Grup {x}",
        default=letters[0],
    ) or letters[0]

    gdf = pd.DataFrame(pred["groups"][g]).rename(
        columns={
            "tim": "Tim",
            "exp_poin": "Poin (ekspektasi)",
            "p_juara_grup": "Juara grup (%)",
            "p_runner_up": "Runner-up (%)",
            "p_lolos": "Lolos 32 Besar (%)",
            "elo": "Elo",
        }
    )
    gdf.insert(0, " ", gdf["Tim"].map(flag_url))

    c_tab, c_chart = st.columns([11, 9])
    with c_tab:
        st.markdown(f"##### Klasemen prediksi — Grup {g}")
        st.dataframe(
            gdf,
            use_container_width=True,
            hide_index=True,
            column_config={
                " ": st.column_config.ImageColumn(" ", width=40),
                "Poin (ekspektasi)": st.column_config.NumberColumn(format="%.2f"),
                "Juara grup (%)": st.column_config.NumberColumn(format="%.1f"),
                "Runner-up (%)": st.column_config.NumberColumn(format="%.1f"),
                "Lolos 32 Besar (%)": st.column_config.ProgressColumn(
                    "Lolos 32 Besar", format="%.1f%%", min_value=0, max_value=100
                ),
                "Elo": st.column_config.NumberColumn(format="%d"),
            },
        )
    with c_chart:
        fig = px.bar(
            gdf,
            x="Tim",
            y=["Juara grup (%)", "Runner-up (%)"],
            barmode="stack",
            title=f"Peluang finis 2 besar — Grup {g}",
            color_discrete_sequence=["#22c55e", "#3b82f6"],
        )
        fig.update_layout(
            height=330, legend_title_text="", margin=dict(t=40, b=0),
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"##### Jadwal & prediksi Grup {g}")
    laga_grup = [m for m in pred["matches"] if m["grup"] == g]
    cols = st.columns(2)
    for i, m in enumerate(laga_grup):
        with cols[i % 2]:
            st.markdown(match_card(m, show_group=False), unsafe_allow_html=True)

# ---------------------------------------------------------------- tab laga
with tab_laga:
    mdf = pd.DataFrame(pred["matches"])
    c1, c2, c3 = st.columns([3, 4, 3])
    f_grup = c1.multiselect("Filter grup", sorted(mdf["grup"].unique()))
    f_tim = c2.multiselect("Filter tim", sorted(set(mdf["home"]) | set(mdf["away"])))
    mode = c3.segmented_control(
        "Tampilan", ["🃏 Kartu", "📋 Tabel"], default="🃏 Kartu"
    )

    view = mdf
    if f_grup:
        view = view[view["grup"].isin(f_grup)]
    if f_tim:
        view = view[view["home"].isin(f_tim) | view["away"].isin(f_tim)]

    if mode == "📋 Tabel":
        tdf = view.rename(
            columns={
                "tanggal": "Tanggal", "grup": "Grup", "home": "Kandang", "away": "Tandang",
                "p_home": "Menang kandang", "p_seri": "Seri", "p_away": "Menang tandang",
                "skor": "Skor", "p_skor": "Prob. skor (%)",
                "xg_home": "xG kandang", "xg_away": "xG tandang",
            }
        )
        tdf.insert(3, " ", tdf["Kandang"].map(flag_url))
        tdf.insert(6, "  ", tdf["Tandang"].map(flag_url))
        st.dataframe(
            tdf,
            use_container_width=True,
            hide_index=True,
            height=560,
            column_config={
                " ": st.column_config.ImageColumn(" ", width=36),
                "  ": st.column_config.ImageColumn(" ", width=36),
                "Menang kandang": st.column_config.ProgressColumn(
                    "Menang kandang", format="%.1f%%", min_value=0, max_value=100
                ),
                "Seri": st.column_config.ProgressColumn(
                    "Seri", format="%.1f%%", min_value=0, max_value=100
                ),
                "Menang tandang": st.column_config.ProgressColumn(
                    "Menang tandang", format="%.1f%%", min_value=0, max_value=100
                ),
                "Prob. skor (%)": st.column_config.NumberColumn(format="%.1f"),
                "xG kandang": st.column_config.NumberColumn(format="%.2f"),
                "xG tandang": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    else:
        st.caption(
            f"{len(view)} pertandingan · bar: "
            f"🟩 menang kandang · ⬜ seri · 🟦 menang tandang"
        )
        cols = st.columns(2)
        for i, m in enumerate(view.to_dict("records")):
            with cols[i % 2]:
                st.markdown(match_card(m), unsafe_allow_html=True)

# ---------------------------------------------------------------- tab bracket
with tab_bracket:
    st.subheader("Jalur knockout paling mungkin (bracket modal)")
    st.caption(
        "Diturunkan dari klasemen ekspektasi tiap grup lalu memilih pemenang "
        "berprobabilitas tertinggi di tiap laga, mengikuti bracket resmi FIFA."
    )
    mb = pred["modal_bracket"]
    rounds_bk = ["32 Besar", "16 Besar", "Perempat Final", "Semifinal", "Final"]
    cols = st.columns(5)
    for col, rnd in zip(cols, rounds_bk):
        with col:
            st.markdown(f"**{rnd}**")
            for m in mb[rnd]:
                win = m["winner"]
                rows = ""
                for side in ("home", "away"):
                    t = m[side]
                    cls = "bk-row win" if t == win else "bk-row"
                    pct = (
                        f"<span class='bk-prob'>{m['p_winner'] * 100:.0f}%</span>"
                        if t == win
                        else ""
                    )
                    rows += f"<div class='{cls}'><span>{flag_img(t, h=13)}{t}</span>{pct}</div>"
                final_cls = " final" if rnd == "Final" else ""
                st.markdown(
                    f"<div class='bk-card{final_cls}'>{rows}</div>",
                    unsafe_allow_html=True,
                )
    final = mb["Final"][0]
    runner = final["away"] if final["winner"] == final["home"] else final["home"]
    st.markdown(
        f"<div class='champ-banner'>🏆 <b>Prediksi juara: "
        f"{flag_img(final['winner'], h=20)}{final['winner']}</b> — menang atas "
        f"{flag_img(runner, h=16)}{runner} di final ({final['p_winner'] * 100:.0f}%)</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- tab profil
with tab_profil:
    prof = load_profiles()
    all_teams = sorted(pred["team_group"].keys())
    team = st.selectbox("Pilih tim", all_teams, index=all_teams.index("Argentina"))

    grp_row = prof["groups"][prof["groups"].team == team]
    snap_row = prof["snapshot"][prof["snapshot"].team == team]
    coach_row = prof["coaches"][prof["coaches"].team == team]
    at_row = prof["alltime"][prof["alltime"].team == team]
    rank_row = prof["ranking"][prof["ranking"].team == team]

    st.markdown(
        f"<h3 style='margin:4px 0 12px'>{flag_img(team, h=28)}{team}</h3>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Grup", pred["team_group"][team])
    c2.metric("Elo", pred["elo"].get(team, "-"))
    if not rank_row.empty:
        c3.metric("Ranking FIFA", int(rank_row.iloc[0]["rank"]))
    if not grp_row.empty:
        c4.metric(
            "Nilai pasar skuad",
            f"€{grp_row.iloc[0]['squad_market_value_eur_millions']:.0f} jt",
        )
    c5.metric("Peluang juara", f"{pred['champion'].get(team, 0):.1f}%")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Peluang per babak")
        rr = pd.Series(pred["rounds"][team])[ROUNDS]
        fig = px.bar(
            rr,
            labels={"index": "", "value": "Probabilitas (%)"},
            color_discrete_sequence=["#22c55e"],
        )
        fig.update_layout(showlegend=False, height=320, margin=dict(t=10))
        fig.update_traces(
            texttemplate="%{y:.1f}", textposition="outside", cliponaxis=False
        )
        st.plotly_chart(fig, use_container_width=True)

        if not grp_row.empty:
            r = grp_row.iloc[0]
            st.markdown(
                f"<div class='info-card'>⭐ <b>Pemain kunci:</b> {r['key_player']}<br>"
                f"📝 {r['notes']}</div>",
                unsafe_allow_html=True,
            )
        if not snap_row.empty:
            s = snap_row.iloc[0]
            st.markdown(
                f"<div class='info-card'>🎫 <b>Kualifikasi:</b> {s['qualification_method']} · "
                f"<b>Penampilan PD:</b> {s['wc_appearances_before_2026']}x · "
                f"<b>Terbaik:</b> {s['best_wc_finish']}</div>",
                unsafe_allow_html=True,
            )

    with colB:
        st.subheader("Pelatih")
        if not coach_row.empty:
            c = coach_row.iloc[0]
            st.markdown(
                f"<div class='info-card'>👔 <b>{c['coach_name']}</b> "
                f"({c['nationality']}, {c['age_at_wc2026']} th) — sejak {c['coach_since']}<br>"
                f"🧠 Gaya: {c['coaching_style']}<br>"
                f"🏟️ PD sebagai pelatih: {c['wc_appearances_as_coach']}x — "
                f"terbaik: {c['wc_best_finish_as_coach']}<br>"
                f"🏅 {c['notable_achievement']}</div>",
                unsafe_allow_html=True,
            )
        st.subheader("Statistik all-time Piala Dunia")
        if at_row.empty:
            st.caption("Belum pernah tampil di Piala Dunia (debut 2026).")
        else:
            a = at_row.iloc[0]
            st.markdown(
                f"<div class='info-card'>"
                f"📅 Penampilan: <b>{a['total_wc_appearances']}x</b> "
                f"({a['first_wc_year']}–{a['last_wc_year']})<br>"
                f"⚔️ Rekor: {a['total_wins']}M-{a['total_draws']}S-{a['total_losses']}K "
                f"dari {a['total_matches']} laga (win rate {a['win_rate']:.0%})<br>"
                f"⚽ Gol: {a['total_goals_scored']}–{a['total_goals_conceded']} "
                f"(selisih {a['goal_difference']:+d})<br>"
                f"🏆 Gelar: <b>{a['titles']}</b> · Final: {a['finals_reached']} · "
                f"Semifinal: {a['semis_reached']}</div>",
                unsafe_allow_html=True,
            )

    st.subheader("Top pencetak gol internasional (sejak 2022)")
    gs = prof["goalscorers"]
    gs_team = gs[
        (gs.team == team) & (gs.date >= "2022-01-01")
        & (~gs.own_goal.fillna(False)) & gs.scorer.notna()
    ]
    if gs_team.empty:
        st.caption("Tidak ada data pencetak gol terkini.")
    else:
        top_sc = gs_team.scorer.value_counts().head(8).sort_values()
        fig = px.bar(
            top_sc,
            orientation="h",
            labels={"index": "", "value": "Gol"},
            text_auto=True,
            color_discrete_sequence=["#facc15"],
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- tab validasi
with tab_validasi:
    metrics = load_metrics()
    if metrics is None:
        st.info("metrics.json belum ada — jalankan notebook 02 terlebih dahulu.")
    else:
        st.subheader("Backtest walk-forward: Piala Dunia 2018 & 2022")
        st.caption(
            "Model dilatih HANYA dengan data sebelum tiap turnamen, lalu diuji pada "
            "seluruh laga turnamen itu. Log-loss & Brier: makin rendah makin baik."
        )
        label = {
            "model_blend": "Model final (blend)",
            "model_classifier": "Classifier saja",
            "model_poisson": "Poisson saja",
            "baseline_elo_logistic": "Baseline: logistik Elo",
            "baseline_prior": "Baseline: prior frekuensi",
        }
        rows = []
        for year in ("2018", "2022"):
            if year not in metrics:
                continue
            for key, nm in label.items():
                rows.append(
                    {
                        "Piala Dunia": year,
                        "Model": nm,
                        "Log-loss": metrics[year][key]["log_loss"],
                        "Brier": metrics[year][key]["brier"],
                        "Akurasi": metrics[year][key]["accuracy"],
                    }
                )
        vdf = pd.DataFrame(rows)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(
                vdf, x="Model", y="Log-loss", color="Piala Dunia", barmode="group",
                title="Log-loss (makin rendah makin baik)",
                color_discrete_sequence=["#22c55e", "#3b82f6"],
            )
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(
                vdf, x="Model", y="Akurasi", color="Piala Dunia", barmode="group",
                title="Akurasi prediksi hasil (W/D/L)",
                color_discrete_sequence=["#22c55e", "#3b82f6"],
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            vdf.style.format({"Log-loss": "{:.4f}", "Brier": "{:.4f}", "Akurasi": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )

        if "final_blend_weights" in metrics:
            w = metrics["final_blend_weights"]
            st.info(
                f"**Bobot ensemble model final** (divalidasi pada PD 2018 & 2022): "
                f"XGB classifier {w[0]:.0%} · XGB Poisson {w[1]:.0%} · "
                f"Elo-logistik {w[2]:.0%} · prior {w[3]:.0%}. "
                f"Prediksi skor & xG tetap dari XGBoost Poisson; bobot di atas "
                f"hanya untuk probabilitas Menang/Seri/Kalah."
            )

        if "feature_importance" in metrics:
            st.subheader("Feature importance (model gol)")
            imp = pd.Series(metrics["feature_importance"]).sort_values()
            fig = px.bar(
                imp, orientation="h",
                labels={"index": "", "value": "importance"},
                color_discrete_sequence=["#22c55e"],
            )
            fig.update_layout(showlegend=False, height=420)
            st.plotly_chart(fig, use_container_width=True)
