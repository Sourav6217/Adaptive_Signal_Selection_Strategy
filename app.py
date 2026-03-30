import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
# Sidebar toggle button
if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"

toggle = st.button("☰ Menu")

if toggle:
    st.session_state.sidebar_state = (
        "collapsed" if st.session_state.sidebar_state == "expanded" else "expanded"
    )
st.set_page_config(
    page_title="Adaptive Signal Selection | Sourav Manna",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state=st.session_state.sidebar_state
)

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.main { background: #0f172a; }
.block-container { padding: 2rem 2.5rem; max-width: 1400px; }
.card {
    background: linear-gradient(135deg, #1e293b 0%, #162032 100%);
    border: 1px solid #334155; border-radius: 16px;
    padding: 1.5rem; margin-bottom: 1.2rem;
}
.card-red   { background:linear-gradient(135deg,#2d0000,#1a0000); border:1px solid #dc262644; border-radius:16px; padding:1.5rem; margin-bottom:1.2rem; }
.card-green { background:linear-gradient(135deg,#052e16,#071a0e); border:1px solid #16a34a44; border-radius:16px; padding:1.5rem; margin-bottom:1.2rem; }
.card-blue  { background:linear-gradient(135deg,#1e3a5f,#0f2847); border:1px solid #2563eb44; border-radius:16px; padding:1.5rem; margin-bottom:1.2rem; }
.metric-tile { background:#1e293b; border:1px solid #334155; border-radius:12px; padding:1.2rem; text-align:center; }
.metric-value { font-size:2rem; font-weight:700; line-height:1.1; }
.metric-label { font-size:.78rem; color:#94a3b8; margin-top:4px; text-transform:uppercase; letter-spacing:.05em; }
.metric-delta { font-size:.85rem; margin-top:6px; }
.section-title { font-size:2rem; font-weight:700; background:linear-gradient(90deg,#60a5fa,#818cf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:.3rem; }
.section-sub { color:#64748b; font-size:.95rem; margin-bottom:2rem; }
.tag       { display:inline-block; background:#1e3a5f; color:#60a5fa; border:1px solid #2563eb44; border-radius:20px; padding:2px 12px; font-size:.78rem; font-weight:500; margin:2px; }
.tag-green { display:inline-block; background:#052e16; color:#4ade80; border:1px solid #16a34a44; border-radius:20px; padding:2px 12px; font-size:.78rem; font-weight:500; margin:2px; }
.tag-red   { display:inline-block; background:#2d0000;  color:#f87171; border:1px solid #dc262644; border-radius:20px; padding:2px 12px; font-size:.78rem; font-weight:500; margin:2px; }
.tag-amber { display:inline-block; background:#1c1400;  color:#fbbf24; border:1px solid #d9770044; border-radius:20px; padding:2px 12px; font-size:.78rem; font-weight:500; margin:2px; }
.insight-box { background:#1e293b; border-left:4px solid #60a5fa; border-radius:0 12px 12px 0; padding:1rem 1.2rem; margin:.8rem 0; color:#cbd5e1; font-size:.92rem; }
.warn-box    { background:#1c1400;  border-left:4px solid #fbbf24; border-radius:0 12px 12px 0; padding:1rem 1.2rem; margin:.8rem 0; color:#fde68a; font-size:.92rem; }
.formula { background:#0f172a; border:1px solid #334155; border-radius:8px; padding:.7rem 1rem; font-family:'Fira Code',monospace; color:#a5f3fc; font-size:.92rem; margin:.5rem 0; }
.hero-title { font-size:2.6rem; font-weight:800; background:linear-gradient(90deg,#60a5fa 0%,#818cf8 50%,#c084fc 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.2; }
.hero-sub { font-size:1.05rem; color:#94a3b8; margin-top:.5rem; line-height:1.6; }
h1, h2, h3, h4 { color: #f1f5f9 !important; }
p, li { color: #cbd5e1; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CHART STYLING — single function, no **kwargs conflicts
# ─────────────────────────────────────────────
COLOR = dict(
    blue="#60a5fa", indigo="#818cf8", purple="#c084fc",
    green="#4ade80", red="#f87171", amber="#fbbf24",
    teal="#2dd4bf", gray="#94a3b8", white="#f1f5f9",
)

def dk(fig, title="", height=380, yprefix="", ysuffix="", logy=False):
    """Apply consistent dark theme to any Plotly figure."""
    ax = dict(gridcolor="#1e293b", linecolor="#334155",
              tickcolor="#334155", zerolinecolor="#334155")
    yax = dict(**ax)
    if yprefix:
        yax["tickprefix"] = yprefix
    if ysuffix:
        yax["ticksuffix"] = ysuffix
    if logy:
        yax["type"] = "log"
    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter", color="#94a3b8"),
        title=dict(text=title, font=dict(color="#f1f5f9", size=15)),
        legend=dict(bgcolor="#1e293b", bordercolor="#334155",
                    borderwidth=1, font=dict(color="#e2e8f0", size=12)),
        margin=dict(l=55, r=25, t=45, b=45),
        hovermode="x unified",
        height=height,
    )
    fig.update_xaxes(**ax)
    fig.update_yaxes(**yax)
    return fig


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    raw = yf.download("BTC-USD", start="2019-01-01", interval="1d", progress=False)
    df  = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns.name = None
    df["returns"] = df["Close"].pct_change()
    df = df.dropna().copy()

    # Signal 1 — Momentum
    df["ma_short"] = df["Close"].rolling(10).mean()
    df["ma_long"]  = df["Close"].rolling(30).mean()
    df["momentum_signal"] = (df["ma_short"] > df["ma_long"]).astype(int).replace({0: -1})

    # Signal 2 — Mean Reversion
    df["rolling_mean"] = df["Close"].rolling(20).mean()
    df["rolling_std"]  = df["Close"].rolling(20).std()
    df["z_score"]      = (df["Close"] - df["rolling_mean"]) / df["rolling_std"]
    df["mr_signal"]    = 0
    df.loc[df["z_score"] >  1, "mr_signal"] = -1
    df.loc[df["z_score"] < -1, "mr_signal"] =  1

    # Signal 3 — ML
    df["target"]     = (df["returns"].shift(-1) > 0).astype(int)
    df["lag1"]       = df["returns"].shift(1)
    df["lag2"]       = df["returns"].shift(2)
    df["lag3"]       = df["returns"].shift(3)
    df["volatility"] = df["returns"].rolling(10).std()
    df = df.dropna().copy()

    feats = ["lag1", "lag2", "lag3", "volatility"]
    X, y  = df[feats], df["target"]
    split = int(len(df) * 0.70)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    df["ml_pred"]   = model.predict(X)
    df["ml_signal"] = df["ml_pred"].replace({0: -1, 1: 1})
    df["ml_proba"]  = model.predict_proba(X)[:, 1]

    # Strategy returns
    df["strat_mom"] = df["momentum_signal"].shift(1) * df["returns"]
    df["strat_mr"]  = df["mr_signal"].shift(1)        * df["returns"]
    df["strat_ml"]  = df["ml_signal"].shift(1)         * df["returns"]

    # Cumulative
    df["cum_mkt"] = (1 + df["returns"]).cumprod()
    df["cum_mom"] = (1 + df["strat_mom"]).cumprod()
    df["cum_mr"]  = (1 + df["strat_mr"]).cumprod()
    df["cum_ml"]  = (1 + df["strat_ml"]).cumprod()

    # UCB Bandit — naive
    arms   = ["momentum_signal", "mr_signal", "ml_signal"]
    sig_df = df[arms].copy()
    cnt    = {a: 1   for a in arms}
    val    = {a: 0.0 for a in arms}
    sel, brets = [], []
    for i in range(len(df)):
        ucb = {a: val[a] + np.sqrt(2 * np.log(i + 1) / cnt[a]) for a in arms}
        arm = max(ucb, key=ucb.get)
        sel.append(arm)
        r = float(sig_df[arm].iloc[i]) * float(df["returns"].iloc[i])
        brets.append(r)
        cnt[arm] += 1
        val[arm] += (r - val[arm]) / cnt[arm]
    df["bandit_ret"]    = brets
    df["cum_bandit"]    = (1 + df["bandit_ret"]).cumprod()
    df["selected_arm"]  = sel

    # UCB Bandit — improved
    cnt2 = {a: 1   for a in arms}
    val2 = {a: 0.0 for a in arms}
    brets2 = []
    for i in range(len(df)):
        ucb2 = {a: val2[a] + np.sqrt(1 * np.log(i + 1) / cnt2[a]) for a in arms}
        arm2 = max(ucb2, key=ucb2.get)
        s2   = float(sig_df[arm2].iloc[i])
        if abs(s2) < 0.1:
            s2 = 0.0
        r2  = s2 * float(df["returns"].iloc[i])
        rw2 = r2 if r2 > 0 else 0.0
        brets2.append(r2)
        cnt2[arm2] += 1
        val2[arm2] += (rw2 - val2[arm2]) / cnt2[arm2]
    df["bandit_imp_ret"] = brets2
    df["cum_bandit_imp"] = (1 + df["bandit_imp_ret"]).cumprod()

    # Performance metrics
    def mets(ret_col, cum_col):
        r   = df[ret_col].dropna()
        cum = df[cum_col]
        sh  = float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else 0.0
        tot = float(cum.iloc[-1])
        mdd = float(((cum - cum.cummax()) / cum.cummax()).min() * 100)
        win = float((r > 0).sum() / len(r) * 100)
        return dict(sharpe=round(sh, 3), total=round(tot, 3),
                    mdd=round(mdd, 1), win=round(win, 1))

    perf = {
        "Market":   mets("returns",      "cum_mkt"),
        "Momentum": mets("strat_mom",    "cum_mom"),
        "Mean Rev": mets("strat_mr",     "cum_mr"),
        "ML":       mets("strat_ml",     "cum_ml"),
        "Bandit":   mets("bandit_ret",   "cum_bandit"),
        "Bandit+":  mets("bandit_imp_ret", "cum_bandit_imp"),
    }
    return df, perf, model, feats, split


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center;padding:1rem 0 1.5rem 0;'>"
        "<div style='font-size:2.2rem;'>📈</div>"
        "<div style='font-size:1rem;font-weight:700;color:#f1f5f9;margin-top:6px;'>"
        "Adaptive Signal Selection</div>"
        "<div style='font-size:.72rem;color:#64748b;margin-top:4px;'>"
        "PGDM-RBA Capstone 2024–26</div></div>",
        unsafe_allow_html=True,
    )
    sections = [
        "🏠  Overview",
        "📊  Data & EDA",
        "🚀  Signal 1 — Momentum",
        "🔁  Signal 2 — Mean Reversion",
        "🤖  Signal 3 — ML (Random Forest)",
        "⚔️  Strategy Comparison",
        "🎰  Multi-Armed Bandit",
        "💡  Key Findings",
        "🔬  Methodology Deep-Dive",
    ]
    page = st.radio("Navigate", sections, label_visibility="collapsed")
    st.markdown("<hr style='border-color:#334155;margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:.75rem;color:#475569;text-align:center;line-height:1.8;'>"
        "<b style='color:#94a3b8;'>Sourav Manna</b><br>"
        "PGDM-RBA (2024–26)<br>"
        "Guide: Prof. Swapnil Arun Desai<br>"
        "L.N. Welingkar Institute, Mumbai</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────
with st.spinner("🔄  Fetching BTC-USD data from Yahoo Finance…"):
    df, perf, model, feats, split = load_data()


# ─────────────────────────────────────────────
#  UI HELPERS
# ─────────────────────────────────────────────
def metric_card(label, value, delta=None, color="#60a5fa", delta_color=None):
    if delta_color is None:
        delta_color = "#4ade80" if delta and not delta.startswith("-") else "#f87171"
    d_html = (f"<div class='metric-delta' style='color:{delta_color};'>{delta}</div>"
              if delta else "")
    return (f"<div class='metric-tile'>"
            f"<div class='metric-value' style='color:{color};'>{value}</div>"
            f"<div class='metric-label'>{label}</div>{d_html}</div>")


def section_header(icon, title, subtitle):
    st.markdown(
        f"<div style='margin-bottom:1.8rem;'>"
        f"<div style='font-size:.85rem;color:#64748b;text-transform:uppercase;"
        f"letter-spacing:.1em;margin-bottom:4px;'>{icon}</div>"
        f"<div class='section-title'>{title}</div>"
        f"<div class='section-sub'>{subtitle}</div></div>",
        unsafe_allow_html=True,
    )


def dark_table(headers, rows, alt_colors=None):
    if alt_colors is None:
        alt_colors = ["#1e293b", "#162032"]
    th = "".join(
        f"<th style='padding:9px 12px;color:#94a3b8;font-weight:500;"
        f"background:#334155;text-align:center;'>{h}</th>"
        for h in headers
    )
    body = ""
    for i, row in enumerate(rows):
        bg = alt_colors[i % 2]
        tds = "".join(
            f"<td style='padding:8px 12px;text-align:center;"
            f"color:#e2e8f0;border-bottom:1px solid #2d3748;'>{c}</td>"
            for c in row
        )
        body += f"<tr style='background:{bg};'>{tds}</tr>"
    return (f"<div class='card' style='overflow-x:auto;'>"
            f"<table style='width:100%;border-collapse:collapse;font-size:.86rem;'>"
            f"<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>")


# ═══════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == sections[0]:

    st.markdown(
        "<div class='card' style='padding:2.5rem;margin-bottom:2rem;"
        "background:linear-gradient(135deg,#0f172a 0%,#1a1040 50%,#0f172a 100%);"
        "border:1px solid #6366f144;'>"
        "<div class='hero-title'>Adaptive Signal Selection</div>"
        "<div class='hero-title' style='font-size:1.9rem;'>under Market Regime Uncertainty</div>"
        "<div class='hero-sub' style='max-width:720px;margin-top:1rem;'>"
        "A Multi-Armed Bandit framework that dynamically selects between "
        "<b style='color:#60a5fa;'>Momentum</b>, "
        "<b style='color:#4ade80;'>Mean Reversion</b>, and "
        "<b style='color:#c084fc;'>Machine Learning</b> signals "
        "to optimise risk-adjusted returns on Bitcoin (2019–2026).</div>"
        "<div style='margin-top:1.5rem;'>"
        "<span class='tag'>BTC-USD Daily</span>"
        "<span class='tag'>2019–2026</span>"
        "<span class='tag'>2,644 Trading Days</span>"
        "<span class='tag'>UCB Algorithm</span>"
        "<span class='tag'>Random Forest</span>"
        "<span class='tag'>Risk Analytics</span>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, lbl, val, dlt, clr in [
        (c1, "Momentum Sharpe", "+0.43",  "Best signal",          "#60a5fa"),
        (c2, "Momentum Return", "+155%",  "vs Buy-Hold +2,400%",  "#4ade80"),
        (c3, "Mean Rev Sharpe", "-0.20",  "BTC not mean-reverting","#f87171"),
        (c4, "Dataset Size",    "2,644",  "Daily observations",   "#818cf8"),
        (c5, "Signals Tested",  "3",      "Mom · MR · ML",        "#fbbf24"),
    ]:
        with col:
            st.markdown(metric_card(lbl, val, dlt, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("### Research Framework")
        for num, title, desc, clr in [
            ("01", "Data Collection",     "BTC-USD OHLCV via Yahoo Finance (2019–2026)",    "#60a5fa"),
            ("02", "Signal Construction", "Momentum · Mean Reversion · ML (Random Forest)", "#818cf8"),
            ("03", "Signal Evaluation",   "Sharpe · Cumulative Return · Drawdown",           "#c084fc"),
            ("04", "Bandit Framework",    "UCB adaptively selects the best signal per step", "#2dd4bf"),
            ("05", "Analysis",            "Compare adaptive vs fixed · failure modes",       "#4ade80"),
        ]:
            st.markdown(
                f"<div style='display:flex;gap:1rem;align-items:flex-start;"
                f"background:#1e293b;border:1px solid #334155;border-radius:12px;"
                f"padding:1rem 1.2rem;margin-bottom:.7rem;'>"
                f"<div style='font-size:1.3rem;font-weight:800;color:{clr};min-width:32px;'>{num}</div>"
                f"<div><div style='font-weight:600;color:#f1f5f9;font-size:.95rem;'>{title}</div>"
                f"<div style='color:#94a3b8;font-size:.83rem;margin-top:3px;'>{desc}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    with col_r:
        st.markdown("### Strategy Scoreboard")
        rows_sb = [
            ("📈 Market",         "+2,400%", "~0.80", "#1a3a2a"),
            ("⚡ Momentum",        "+155%",   "+0.43", "#1e293b"),
            ("🔁 Mean Reversion",  "-83%",    "-0.20", "#3a1a1a"),
            ("🤖 ML (Raw)",        "Overfit", "9.11*", "#1a1a3a"),
            ("🎰 UCB Bandit",      "Negative","Neg.",  "#2a1a1a"),
        ]
        rows_html = ""
        for name, ret, sharpe, bg in rows_sb:
            rc = "#4ade80" if ret.startswith("+") else "#f87171" if ret.startswith("-") else "#fbbf24"
            sc = ("#4ade80" if sharpe.startswith("+") or "0.8" in sharpe
                  else "#f87171" if sharpe.startswith("-") or sharpe in ("Neg.",)
                  else "#fbbf24")
            rows_html += (
                f"<tr style='background:{bg};border-bottom:1px solid #334155;'>"
                f"<td style='padding:8px 10px;color:#e2e8f0;'>{name}</td>"
                f"<td style='padding:8px 10px;text-align:center;color:{rc};font-weight:600;'>{ret}</td>"
                f"<td style='padding:8px 10px;text-align:center;color:{sc};font-weight:600;'>{sharpe}</td>"
                f"</tr>"
            )
        st.markdown(
            f"<div class='card'><table style='width:100%;border-collapse:collapse;font-size:.85rem;'>"
            f"<tr style='background:#334155;'>"
            f"<th style='padding:8px 10px;color:#94a3b8;'>Strategy</th>"
            f"<th style='padding:8px 10px;text-align:center;color:#94a3b8;'>Return</th>"
            f"<th style='padding:8px 10px;text-align:center;color:#94a3b8;'>Sharpe</th>"
            f"</tr>{rows_html}</table></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='warn-box'><b>Key Takeaway:</b> No single strategy performs consistently "
            "across all BTC regimes. Adaptive selection works only when underlying signals have "
            "genuine regime-dependent performance.</div>",
            unsafe_allow_html=True,
        )

    # BTC price chart
    st.markdown("<br>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], mode="lines", name="BTC-USD Close",
        line=dict(color=COLOR["blue"], width=1.5),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.05)",
    ))
    for date, label, clr in [
        (pd.to_datetime("2020-03-12"), "Covid Crash",    "#f87171"),
        (pd.to_datetime("2021-11-10"), "ATH $69K",       "#4ade80"),
        (pd.to_datetime("2022-11-09"), "FTX Collapse",   "#f87171"),
        (pd.to_datetime("2024-11-05"), "Election Rally", "#4ade80"),
    ]:
        fig.add_vline(x=date, line_dash="dot", line_color=clr, line_width=1, opacity=0.5,
                      annotation_text=label, annotation_font_color=clr,
                      annotation_font_size=10)
    dk(fig, title="Bitcoin (BTC-USD) Price History — Full Study Period",
       height=320, yprefix="$")
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: DATA & EDA
# ═══════════════════════════════════════════════════════════
elif page == sections[1]:
    section_header("📊 DATA & EDA", "Understanding the BTC-USD Dataset",
                   "Statistical properties, return distributions, volatility analysis")

    close, rets = df["Close"], df["returns"]
    c1, c2, c3, c4 = st.columns(4)
    for col, items in [
        (c1, [("Observations", f"{len(df):,}"),
              ("Start", str(df.index[0].date())),
              ("End",   str(df.index[-1].date()))]),
        (c2, [("Min Price",  f"${close.min():,.0f}"),
              ("Max Price",  f"${close.max():,.0f}"),
              ("Mean Price", f"${close.mean():,.0f}")]),
        (c3, [("Daily Mean",  f"{rets.mean()*100:.3f}%"),
              ("Daily Std",   f"{rets.std()*100:.2f}%"),
              ("Ann. Vol",    f"{rets.std()*np.sqrt(252)*100:.1f}%")]),
        (c4, [("Skewness",   f"{rets.skew():.3f}"),
              ("Kurtosis",   f"{rets.kurt():.3f}"),
              ("+ve Days",   f"{(rets>0).mean()*100:.1f}%")]),
    ]:
        with col:
            rows_html = "".join(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:6px 0;border-bottom:1px solid #334155;'>"
                f"<span style='color:#94a3b8;font-size:.82rem;'>{k}</span>"
                f"<span style='color:#f1f5f9;font-weight:600;font-size:.88rem;'>{v}</span></div>"
                for k, v in items
            )
            st.markdown(f"<div class='card'>{rows_html}</div>", unsafe_allow_html=True)

    # Price + Volume
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="BTC Close",
                             line=dict(color=COLOR["blue"], width=1.5),
                             fill="tozeroy", fillcolor="rgba(96,165,250,0.08)"),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color=COLOR["indigo"], opacity=0.5),
                  row=2, col=1)
    dk(fig, title="BTC-USD: Price & Volume (2019–2026)", height=450)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df["returns"] * 100,
                                  mode="lines", name="Daily Returns",
                                  line=dict(color=COLOR["teal"], width=0.8)))
        dk(fig2, title="Daily Returns (%) — Volatility Clustering", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        mu    = df["returns"].mean() * 100
        sigma = df["returns"].std()  * 100
        x_n   = np.linspace(-20, 20, 200)
        y_n   = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_n - mu) / sigma) ** 2)
        y_n   = y_n * len(df) * (40 / 120)
        fig3  = go.Figure()
        fig3.add_trace(go.Histogram(x=df["returns"] * 100, nbinsx=120,
                                    name="Returns", marker_color=COLOR["indigo"], opacity=0.8))
        fig3.add_trace(go.Scatter(x=x_n, y=y_n, mode="lines", name="Normal",
                                  line=dict(color=COLOR["amber"], width=2, dash="dash")))
        dk(fig3, title="Return Distribution vs Normal (Fat Tails)", height=300)
        st.plotly_chart(fig3, use_container_width=True)

    df["roll_vol_30"] = df["returns"].rolling(30).std() * np.sqrt(252) * 100
    df["roll_vol_90"] = df["returns"].rolling(90).std() * np.sqrt(252) * 100
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df.index, y=df["roll_vol_30"], name="30-day Vol",
                              line=dict(color=COLOR["purple"], width=1.2),
                              fill="tozeroy", fillcolor="rgba(192,132,252,0.08)"))
    fig4.add_trace(go.Scatter(x=df.index, y=df["roll_vol_90"], name="90-day Vol",
                              line=dict(color=COLOR["amber"], width=1.5, dash="dot")))
    dk(fig4, title="Rolling Annualised Volatility — Regime Shifts Visible",
       height=280, ysuffix="%")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        "<div class='insight-box'><b style='color:#60a5fa;'>📌 Key EDA Insights</b><br>"
        "<b>Fat Tails:</b> Kurtosis &gt; 5 — extreme events far more frequent than Normal. &nbsp;|&nbsp; "
        "<b>Volatility Clustering:</b> High-vol periods cluster (COVID 2020, FTX 2022). &nbsp;|&nbsp; "
        "<b>Positive Drift:</b> Mean daily +0.18% → ~45% annualised — structural advantage for "
        "long-biased momentum strategies.</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
#  PAGE: MOMENTUM
# ═══════════════════════════════════════════════════════════
elif page == sections[2]:
    section_header("🚀 SIGNAL 1", "Momentum Strategy",
                   "Moving Average Crossover (10-day vs 30-day)")

    m = perf["Momentum"]
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, dlt, clr in [
        (c1, "Sharpe Ratio",      f"+{m['sharpe']}", "Positive risk-adjusted return", "#4ade80"),
        (c2, "Cumulative Return", f"{m['total']:.2f}x", "+155% over 7 years",          "#60a5fa"),
        (c3, "Max Drawdown",      f"{m['mdd']:.1f}%",   "Worst peak-to-trough",         "#f87171"),
        (c4, "Win Rate",          f"{m['win']:.1f}%",   "Days with +ve return",          "#818cf8"),
    ]:
        with col:
            st.markdown(metric_card(lbl, val, dlt, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="BTC Price",
                             line=dict(color=COLOR["gray"], width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_short"], name="MA(10)",
                             line=dict(color=COLOR["amber"], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_long"],  name="MA(30)",
                             line=dict(color=COLOR["blue"], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index,
                             y=df["momentum_signal"].astype(float),
                             name="Signal (+1/-1)", mode="lines",
                             line=dict(color=COLOR["teal"], width=1.5),
                             fill="tozeroy", fillcolor="rgba(45,212,191,0.1)"),
                  row=2, col=1)
    dk(fig, title="Momentum Signal: MA(10) vs MA(30) Crossover", height=480)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df["cum_mkt"], name="Market (Buy & Hold)",
                                  line=dict(color=COLOR["blue"], width=2),
                                  fill="tozeroy", fillcolor="rgba(96,165,250,0.05)"))
        fig2.add_trace(go.Scatter(x=df.index, y=df["cum_mom"], name="Momentum Strategy",
                                  line=dict(color=COLOR["amber"], width=2)))
        dk(fig2, title="Cumulative Returns: Momentum vs Market", height=320)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("### Signal Logic")
        st.markdown(
            "<div class='formula'>"
            "ma_short = Close.rolling(10).mean()<br>"
            "ma_long  = Close.rolling(30).mean()<br><br>"
            "Signal = +1  if ma_short &gt; ma_long<br>"
            "Signal = -1  otherwise<br><br>"
            "Strategy_t = Signal_{t-1} &times; Return_t"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='insight-box'><b>Why it works:</b> BTC momentum is driven by "
            "speculative demand cycles and retail herding — trend-following captures "
            "sustained directional moves.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='warn-box'><b>Limitation:</b> Whipsaws in choppy markets and late "
            "entries/exits reduce performance vs buy-and-hold.</div>",
            unsafe_allow_html=True,
        )

    roll_r = df["strat_mom"].rolling(90)
    rs = np.sqrt(252) * roll_r.mean() / roll_r.std()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index, y=rs, name="Rolling Sharpe",
                              line=dict(color=COLOR["green"], width=1.5)))
    fig3.add_hline(y=0,    line_dash="dot",  line_color=COLOR["red"],   opacity=0.6)
    fig3.add_hline(y=0.43, line_dash="dash", line_color=COLOR["amber"], opacity=0.7,
                   annotation_text="Full-Period Sharpe: 0.43",
                   annotation_font_color=COLOR["amber"])
    dk(fig3, title="Rolling 90-day Sharpe — Regime-Dependent Performance", height=270)
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: MEAN REVERSION
# ═══════════════════════════════════════════════════════════
elif page == sections[3]:
    section_header("🔁 SIGNAL 2", "Mean Reversion Strategy",
                   "Z-Score based contrarian signal — buy dips, sell spikes")

    m = perf["Mean Rev"]
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, dlt, clr in [
        (c1, "Sharpe Ratio",      f"{m['sharpe']}",    "Negative",             "#f87171"),
        (c2, "Cumulative Return", f"{m['total']:.2f}x", "-83% catastrophic",    "#f87171"),
        (c3, "Max Drawdown",      f"{m['mdd']:.1f}%",   "Extreme losses",        "#f87171"),
        (c4, "Win Rate",          f"{m['win']:.1f}%",   "Below 50% baseline",    "#f87171"),
    ]:
        with col:
            st.markdown(metric_card(lbl, val, dlt, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.45, 0.55], vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="BTC Price",
                             line=dict(color=COLOR["gray"], width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rolling_mean"], name="20-day Mean",
                             line=dict(color=COLOR["amber"], width=1.2, dash="dash")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["z_score"], name="Z-Score",
                             line=dict(color=COLOR["purple"], width=1.2)),
                  row=2, col=1)
    fig.add_hline(y=1,  line_dash="dot", line_color=COLOR["red"],   opacity=0.7, row=2, col=1)
    fig.add_hline(y=-1, line_dash="dot", line_color=COLOR["green"], opacity=0.7, row=2, col=1)
    dk(fig, title="Z-Score Signal: Price Deviation from 20-day Rolling Mean", height=460)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df["cum_mkt"], name="Market",
                                  line=dict(color=COLOR["blue"], width=2)))
        fig2.add_trace(go.Scatter(x=df.index, y=df["cum_mr"], name="Mean Reversion",
                                  line=dict(color=COLOR["red"], width=2),
                                  fill="tozeroy", fillcolor="rgba(248,113,113,0.07)"))
        dk(fig2, title="Mean Reversion Strategy vs Market", height=310)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("### Signal Logic")
        st.markdown(
            "<div class='formula'>"
            "Z_t = (Close_t - mean_20) / std_20<br><br>"
            "Signal = -1  if Z_t &gt; +1  (sell)<br>"
            "Signal = +1  if Z_t &lt; -1  (buy)<br>"
            "Signal =  0  otherwise"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='card-red'><b style='color:#f87171;'>❌ Why It Fails on BTC</b>"
            "<ul style='color:#fca5a5;margin-top:8px;padding-left:1.2rem;font-size:.88rem;'>"
            "<li>BTC is <b>momentum-dominated</b> — prices trend, not revert</li>"
            "<li>Selling after Z &gt; 1 means <b>selling into a bull run</b></li>"
            "<li>Buying after Z &lt; -1 means <b>catching falling knives</b></li>"
            "<li>Speculative demand creates <b>extended departures</b> from mean</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════
#  PAGE: ML
# ═══════════════════════════════════════════════════════════
elif page == sections[4]:
    section_header("🤖 SIGNAL 3", "Machine Learning — Random Forest",
                   "Predicting next-day BTC direction using lagged returns & volatility")

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, dlt, clr in [
        (c1, "Model",       "Random Forest", "100 trees, max_depth=5", "#818cf8"),
        (c2, "Features",    "4",             "lag1, lag2, lag3, vol",  "#60a5fa"),
        (c3, "Train Split", "70%",           "Time-based, no shuffle", "#4ade80"),
        (c4, "Task",        "Binary",        "Direction: Up / Down",   "#fbbf24"),
    ]:
        with col:
            st.markdown(metric_card(lbl, val, dlt, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        imps = model.feature_importances_
        fig  = go.Figure(go.Bar(
            x=imps, y=feats, orientation="h",
            marker=dict(color=[COLOR["blue"], COLOR["indigo"],
                               COLOR["purple"], COLOR["teal"]]),
        ))
        dk(fig, title="Feature Importance — Random Forest", height=270)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=df["ml_proba"], nbinsx=50,
                                    marker_color=COLOR["purple"], opacity=0.8, name="P(Up)"))
        fig2.add_vline(x=0.5, line_dash="dash", line_color=COLOR["amber"],
                       annotation_text="Threshold 0.5",
                       annotation_font_color=COLOR["amber"])
        dk(fig2, title="ML Predicted P(Price Up Tomorrow)", height=270)
        st.plotly_chart(fig2, use_container_width=True)

    X_test = df[feats].iloc[split:]
    y_test = df["target"].iloc[split:]
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    acc    = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100

    col_c, col_d = st.columns([2, 3])
    with col_c:
        fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual"),
                           x=["Down", "Up"], y=["Down", "Up"],
                           color_continuous_scale=[[0, "#0f172a"], [1, "#818cf8"]],
                           text_auto=True)
        dk(fig_cm, title=f"Confusion Matrix (Test Set) — Accuracy: {acc:.1f}%", height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_d:
        st.markdown(
            "<div class='warn-box'><b>⚠️ Overfitting Warning</b><br>"
            "Raw backtest Sharpe ~9.11 is spurious — predictions are made on training data. "
            "<b>Out-of-sample accuracy ~50–52%</b>, near random. Lagged returns alone lack "
            "sufficient predictive signal for BTC direction.</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='insight-box'><b>Volatility is the most important feature</b> — "
            "market regime (high vs low vol) is more informative than lagged returns for "
            "direction prediction.</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════
#  PAGE: STRATEGY COMPARISON
# ═══════════════════════════════════════════════════════════
elif page == sections[5]:
    section_header("⚔️ COMPARISON", "Head-to-Head Performance",
                   "Cumulative returns, risk metrics, and regime-dependent behaviour")

    fig = go.Figure()
    for col_n, name, clr, w in [
        ("cum_mkt", "Market",        COLOR["blue"],  2.5),
        ("cum_mom", "Momentum",      COLOR["amber"], 2.0),
        ("cum_mr",  "Mean Reversion",COLOR["red"],   1.5),
    ]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col_n], name=name,
                                 line=dict(color=clr, width=w)))
    dk(fig, title="Strategy Comparison — Full Period (Log Scale)", height=400, logy=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Risk-Adjusted Metrics")
    rows = []
    for s in ["Market", "Momentum", "Mean Rev"]:
        m = perf[s]
        rows.append([s, f"{m['total']:.2f}x",
                     f"{(m['total']-1)*100:+.0f}%",
                     f"{m['sharpe']:+.3f}",
                     f"{m['mdd']:.1f}%",
                     f"{m['win']:.1f}%"])
    st.markdown(dark_table(
        ["Strategy", "Wealth Multiple", "Total Return", "Sharpe", "Max Drawdown", "Win Rate"],
        rows,
        alt_colors=["#1a3a2a", "#1e293b", "#3a1a1a"],
    ), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        for col_n, name, clr in [
            ("cum_mkt", "Market",         COLOR["blue"]),
            ("cum_mom", "Momentum",       COLOR["amber"]),
            ("cum_mr",  "Mean Reversion", COLOR["red"]),
        ]:
            s  = df[col_n]
            dd = (s - s.cummax()) / s.cummax() * 100
            fig2.add_trace(go.Scatter(x=df.index, y=dd, name=name,
                                      line=dict(color=clr, width=1.2)))
        dk(fig2, title="Drawdown Comparison (%)", height=300, ysuffix="%")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure()
        for col_n, name, clr in [
            ("strat_mom", "Momentum",       COLOR["amber"]),
            ("strat_mr",  "Mean Reversion", COLOR["red"]),
        ]:
            rr = df[col_n].rolling(90)
            rs = np.sqrt(252) * rr.mean() / rr.std()
            fig3.add_trace(go.Scatter(x=df.index, y=rs, name=name,
                                      line=dict(color=clr, width=1.2)))
        fig3.add_hline(y=0, line_dash="dash", line_color=COLOR["gray"], opacity=0.5)
        dk(fig3, title="Rolling 90-day Sharpe Comparison", height=300)
        st.plotly_chart(fig3, use_container_width=True)

    regimes = ["Uptrend 20-21", "Correction 21", "Bear 22", "Recovery 23", "Bull II 24-25"]
    fig4 = go.Figure(data=go.Heatmap(
        z=[[3, -2, 2, 3, 3], [-3, -2, -3, -3, -3], [4, -3, -4, 4, 4]],
        x=regimes, y=["Momentum", "Mean Rev", "Buy & Hold"],
        colorscale=[[0, "#2d0000"], [0.35, "#3a1a1a"], [0.5, "#1e293b"],
                    [0.65, "#1a3a2a"], [1, "#052e16"]],
        text=[["Good", "Bad", "Moderate", "Good", "Good"],
              ["Very Bad", "Bad", "Very Bad", "Bad", "Very Bad"],
              ["Best ★", "Drawdown", "Best ★", "Best ★", "Best ★"]],
        texttemplate="%{text}", showscale=False,
    ))
    dk(fig4, title="Qualitative Performance by Market Regime", height=240)
    st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: BANDIT
# ═══════════════════════════════════════════════════════════
elif page == sections[6]:
    section_header("🎰 MULTI-ARMED BANDIT", "UCB Adaptive Signal Selection",
                   "Upper Confidence Bound algorithm dynamically allocates across the three signal arms")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("### UCB Algorithm")
        st.markdown(
            "<div class='formula'>"
            "UCB_arm(t) = Q_arm + sqrt( c &times; ln(t) / N_arm )<br><br>"
            "Q_arm = average reward so far<br>"
            "N_arm = times this arm was selected<br>"
            "c     = exploration constant (2.0 naive, 1.0 improved)<br>"
            "t     = current time step<br><br>"
            "Chosen = argmax UCB_arm(t)"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='display:flex;gap:.7rem;margin-top:1rem;'>"
            "<div class='card' style='flex:1;text-align:center;'>"
            "<div style='font-size:1.6rem;'>🎯</div>"
            "<div style='color:#60a5fa;font-weight:600;font-size:.9rem;'>Exploit</div>"
            "<div style='color:#94a3b8;font-size:.78rem;'>Select best known arm</div></div>"
            "<div class='card' style='flex:1;text-align:center;'>"
            "<div style='font-size:1.6rem;'>🔍</div>"
            "<div style='color:#818cf8;font-weight:600;font-size:.9rem;'>Explore</div>"
            "<div style='color:#94a3b8;font-size:.78rem;'>Try uncertain arms</div></div>"
            "<div class='card' style='flex:1;text-align:center;'>"
            "<div style='font-size:1.6rem;'>⚖️</div>"
            "<div style='color:#4ade80;font-weight:600;font-size:.9rem;'>Balance</div>"
            "<div style='color:#94a3b8;font-size:.78rem;'>Confidence bound manages the trade-off</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col_r:
        st.markdown("### Performance Metrics")
        for name, m, clr in [
            ("Naive Bandit",    perf["Bandit"],  "#f87171"),
            ("Improved Bandit", perf["Bandit+"], "#fbbf24"),
        ]:
            st.markdown(
                f"<div class='card' style='border-color:{clr}33;margin-bottom:.7rem;'>"
                f"<div style='color:{clr};font-weight:600;margin-bottom:8px;'>{name}</div>"
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:.83rem;'>"
                f"<div><span style='color:#64748b;'>Sharpe: </span>"
                f"<span style='color:{clr};font-weight:600;'>{m['sharpe']}</span></div>"
                f"<div><span style='color:#64748b;'>Return: </span>"
                f"<span style='color:{clr};font-weight:600;'>{m['total']:.2f}x</span></div>"
                f"<div><span style='color:#64748b;'>Max DD: </span>"
                f"<span style='color:#f87171;font-weight:600;'>{m['mdd']:.1f}%</span></div>"
                f"<div><span style='color:#64748b;'>Win Rate: </span>"
                f"<span style='color:#94a3b8;font-weight:600;'>{m['win']:.1f}%</span></div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    fig = go.Figure()
    for col_n, name, clr, w in [
        ("cum_mkt",       "Market",          COLOR["blue"],   2.5),
        ("cum_mom",       "Momentum",        COLOR["amber"],  2.0),
        ("cum_mr",        "Mean Reversion",  COLOR["green"],  1.5),
        ("cum_bandit",    "Naive Bandit",    COLOR["red"],    1.5),
        ("cum_bandit_imp","Improved Bandit", COLOR["purple"], 2.0),
    ]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col_n], name=name,
                                 line=dict(color=clr, width=w)))
    dk(fig, title="All Strategies vs Adaptive Bandit — Full Period (Log Scale)",
       height=400, logy=True)
    st.plotly_chart(fig, use_container_width=True)

    arm_labels = {"momentum_signal": "Momentum",
                  "mr_signal":       "Mean Rev",
                  "ml_signal":       "ML"}
    arm_counts = pd.Series(df["selected_arm"]).value_counts()
    arm_counts.index = [arm_labels.get(x, x) for x in arm_counts.index]

    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure(go.Pie(
            labels=arm_counts.index, values=arm_counts.values, hole=0.55,
            marker=dict(colors=[COLOR["amber"], COLOR["red"], COLOR["purple"]]),
        ))
        dk(fig2, title="Arm Selection Frequency (Naive UCB)", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        arm_df    = pd.get_dummies(df["selected_arm"])
        arm_df.columns = [arm_labels.get(c, c) for c in arm_df.columns]
        roll_arms = arm_df.rolling(90).mean()
        fig3 = go.Figure()
        for col_n, clr in zip(arm_df.columns,
                               [COLOR["amber"], COLOR["red"], COLOR["purple"]]):
            if col_n in roll_arms.columns:
                fig3.add_trace(go.Scatter(
                    x=df.index, y=roll_arms[col_n] * 100,
                    name=col_n, line=dict(color=clr, width=1.5),
                    stackgroup="one",
                ))
        dk(fig3, title="Rolling 90-day Arm Selection Share", height=300, ysuffix="%")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("### Why the Bandit Underperforms")
    for title, desc, clr in [
        ("Signal Pool Quality",
         "Mean Reversion is structurally loss-making on BTC. Forced UCB exploration of a "
         "universally bad arm creates permanent drag regardless of algorithm tuning.", "#f87171"),
        ("Non-Stationarity",
         "UCB1 assumes stationary reward distributions. BTC regime shifts invalidate historical "
         "Q estimates — the algorithm is fighting the last war.", "#fbbf24"),
        ("Exploration Cost",
         "Exploring short-biased strategies during bull markets is disproportionately costly. "
         "UCB's bonus is optimal under i.i.d. rewards, not in directional markets.", "#f87171"),
        ("Context Blindness",
         "Standard UCB ignores observable regime features (vol level, trend strength). "
         "A contextual bandit (LinUCB) could condition arm selection on market state.", "#60a5fa"),
    ]:
        st.markdown(
            f"<div style='background:#1e293b;border:1px solid {clr}33;"
            f"border-left:3px solid {clr};border-radius:0 12px 12px 0;"
            f"padding:.9rem 1.1rem;margin-bottom:.6rem;'>"
            f"<span style='color:{clr};font-weight:600;'>{title}: </span>"
            f"<span style='color:#94a3b8;font-size:.88rem;'>{desc}</span></div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════
#  PAGE: KEY FINDINGS
# ═══════════════════════════════════════════════════════════
elif page == sections[7]:
    section_header("💡 KEY FINDINGS", "Research Insights & Conclusions",
                   "What this study reveals about adaptive signal selection in crypto markets")

    findings = [
        ("#f87171", "01", "No Single Strategy is Consistently Superior",
         "Momentum Sharpe +0.43, Mean Reversion -0.20. Performance is strongly regime-dependent.",
         [("Validated", "tag-green"), ("Core Finding", "tag")]),
        ("#60a5fa", "02", "BTC is a Momentum-Dominated Asset",
         "Momentum +155% vs Mean Reversion -83%. Speculative demand cycles and retail herding "
         "create strong persistent trends.",
         [("Asset-Specific", "tag"), ("High Confidence", "tag-green")]),
        ("#fbbf24", "03", "Adaptive Bandits Require Strong Signal Quality",
         "UCB bandit underperforms all strategies when the pool contains a universally "
         "loss-making arm. Signal quality is a prerequisite, not an output.",
         [("Critical Insight", "tag-amber"), ("Novel Contribution", "tag-amber")]),
        ("#c084fc", "04", "UCB Exploration is Costly in Directional Markets",
         "Forced exploration of short-biased strategies during bull markets generates compounding "
         "losses. UCB's i.i.d. reward assumption breaks in trending markets.",
         [("Theoretical", "tag"), ("UCB Limitation", "tag-red")]),
        ("#4ade80", "05", "ML Signals Need Rigorous Out-of-Sample Validation",
         "Spurious Sharpe 9.11 from in-sample evaluation warns practitioners. Walk-forward "
         "validation and regularisation are non-negotiable for financial ML.",
         [("Warning", "tag-red"), ("Best Practice", "tag")]),
        ("#2dd4bf", "06", "Signal Improvement Must Precede Framework Improvement",
         "Correct order: (1) improve signals, (2) ensure regime-diversity, (3) add adaptive "
         "layer. Skipping to step 3 produces the failure seen here.",
         [("Design Principle", "tag-green"), ("Actionable", "tag-green")]),
    ]
    for clr, num, title, body, tags in findings:
        tags_html = " ".join(f"<span class='{tc}'>{t}</span>" for t, tc in tags)
        st.markdown(
            f"<div class='card' style='border-left:4px solid {clr};margin-bottom:1.2rem;'>"
            f"<div style='display:flex;gap:1.2rem;align-items:flex-start;'>"
            f"<div style='font-size:2rem;font-weight:800;color:{clr};"
            f"min-width:48px;line-height:1;opacity:.7;'>{num}</div>"
            f"<div style='flex:1;'>"
            f"<div style='font-weight:700;color:#f1f5f9;font-size:1rem;margin-bottom:6px;'>{title}</div>"
            f"<div style='color:#94a3b8;font-size:.88rem;line-height:1.6;margin-bottom:10px;'>{body}</div>"
            f"{tags_html}</div></div></div>",
            unsafe_allow_html=True,
        )

    # Radar
    categories = ["Sharpe", "Total Return", "Win Rate", "Low Drawdown", "Stability", "Simplicity"]
    scores = {
        "Momentum":   [6, 5, 5, 7, 6, 9],
        "MR":         [1, 1, 4, 1, 2, 9],
        "Bandit":     [1, 1, 4, 1, 2, 3],
        "Buy & Hold": [8, 10, 6, 5, 7, 10],
    }
    fig = go.Figure()
    for (name, vals), clr in zip(scores.items(),
                                  [COLOR["amber"], COLOR["red"],
                                   COLOR["purple"], COLOR["blue"]]):
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            name=name, line=dict(color=clr, width=2),
            fill="toself", fillcolor=clr + "18",
        ))
    fig.update_layout(
        paper_bgcolor="#0f172a",
        polar=dict(bgcolor="#0f172a",
                   radialaxis=dict(gridcolor="#334155", range=[0, 10]),
                   angularaxis=dict(gridcolor="#334155")),
        legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1),
        margin=dict(l=40, r=40, t=50, b=40),
        height=420,
        title=dict(text="Strategy Profiles (0–10 scale)",
                   font=dict(color="#f1f5f9", size=15)),
        font=dict(family="Inter", color="#94a3b8"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div class='card-blue'><b style='color:#60a5fa;font-size:1rem;'>📝 Final Verdict</b>"
        "<p style='color:#cbd5e1;margin-top:.7rem;line-height:1.7;'>"
        "Adaptive signal selection frameworks are theoretically sound but practically constrained by "
        "signal quality. For Bitcoin specifically, a simple momentum strategy remains the strongest performer. "
        "Bandits add value only when signal diversity with genuine regime-switching behaviour exists in the pool. "
        "The study's most valuable contribution is identifying <b>what makes adaptive allocation fail</b> — "
        "a finding directly applicable to practitioners building live systematic trading systems."
        "</p></div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════
#  PAGE: METHODOLOGY
# ═══════════════════════════════════════════════════════════
elif page == sections[8]:
    section_header("🔬 METHODOLOGY", "Technical Deep-Dive",
                   "Signal math, bandit pseudocode, evaluation metrics, and live data sample")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📐 Signal Math", "🎰 Bandit Pseudocode", "📏 Metrics", "🗂️ Data Pipeline"]
    )

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### Signal 1: Momentum")
            st.markdown(
                "<div class='formula'>"
                "ma_short = Close.rolling(10).mean()<br>"
                "ma_long  = Close.rolling(30).mean()<br><br>"
                "signal = +1 if ma_short &gt; ma_long<br>"
                "       = -1 otherwise<br><br>"
                "strat_t = signal_{t-1} &times; return_t<br><br>"
                "Sharpe = &radic;252 &times; mean(r)/std(r)"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("**Params:** 10/30-day windows  \n**Fit:** ✅ Strong on BTC")
        with c2:
            st.markdown("#### Signal 2: Mean Reversion")
            st.markdown(
                "<div class='formula'>"
                "Z = (Close - mean_20) / std_20<br><br>"
                "signal = -1 if Z &gt; +1 (sell)<br>"
                "       = +1 if Z &lt; -1 (buy)<br>"
                "       =  0 otherwise"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("**Params:** 20-day window, ±1σ  \n**Fit:** ❌ Fails on BTC")
        with c3:
            st.markdown("#### Signal 3: ML (RF)")
            st.markdown(
                "<div class='formula'>"
                "Features = [lag1, lag2, lag3, vol10]<br><br>"
                "target = 1 if return_{t+1} &gt; 0<br>"
                "       = 0 otherwise<br><br>"
                "RandomForestClassifier(<br>"
                "&nbsp;n_estimators=100,<br>"
                "&nbsp;max_depth=5<br>"
                ")<br><br>"
                "Train=70%, Test=30% (time-ordered)"
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("**OOS accuracy:** ~50%  \n**Warning:** ⚠️ Overfit in-sample")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### UCB1 — Naive")
            st.code(
                "arms   = [momentum, mr, ml]\n"
                "counts = {arm: 1   for arm in arms}\n"
                "values = {arm: 0.0 for arm in arms}\n\n"
                "for t in range(len(df)):\n"
                "    ucb = {\n"
                "        a: values[a] +\n"
                "           sqrt(2*log(t+1)/counts[a])\n"
                "        for a in arms\n"
                "    }\n"
                "    chosen = argmax(ucb)\n"
                "    reward = signal[chosen,t] * return_t\n"
                "    counts[chosen] += 1\n"
                "    values[chosen] += (\n"
                "        (reward - values[chosen])\n"
                "        / counts[chosen]\n"
                "    )",
                language="python",
            )
        with c2:
            st.markdown("#### UCB — Improved")
            st.code(
                "# 1. Signal Filtering\n"
                "if abs(signal) < 0.1:\n"
                "    signal = 0\n\n"
                "# 2. Asymmetric Reward\n"
                "reward = return if return > 0\n"
                "       = 0      otherwise\n\n"
                "# 3. Reduced Exploration\n"
                "bonus = sqrt(1*log(t+1)/counts[arm])\n"
                "#       c=1 instead of c=2\n\n"
                "# Result: still underperforms —\n"
                "# signal quality is the constraint",
                language="python",
            )

    with tab3:
        st.markdown(dark_table(
            ["Metric", "Formula", "Interpretation"],
            [
                ["Sharpe Ratio",     "√252 × E[r] / σ(r)",     ">1 = good, >2 = excellent"],
                ["Total Return",     "∏(1 + r_t)",             "2.55x = +155% over period"],
                ["Max Drawdown",     "(peak−trough) / peak",   "Lower = worse"],
                ["Win Rate",         "Count(r>0) / T × 100",   ">55% = meaningful edge"],
                ["Ann. Return",      "(1+R)^(252/T) - 1",      "Constant annual equivalent"],
            ],
        ), unsafe_allow_html=True)

    with tab4:
        st.code(
            "import yfinance as yf\n"
            "import pandas as pd\n"
            "import numpy as np\n\n"
            "# Fetch\n"
            "df = yf.download('BTC-USD', start='2019-01-01', interval='1d')\n\n"
            "# Flatten MultiIndex\n"
            "df.columns = df.columns.get_level_values(0)\n"
            "df.columns.name = None\n\n"
            "# Returns\n"
            "df['returns'] = df['Close'].pct_change()\n"
            "df = df.dropna()\n\n"
            "# Features\n"
            "df['lag1']       = df['returns'].shift(1)\n"
            "df['lag2']       = df['returns'].shift(2)\n"
            "df['lag3']       = df['returns'].shift(3)\n"
            "df['volatility'] = df['returns'].rolling(10).std()\n\n"
            "# Target\n"
            "df['target'] = (df['returns'].shift(-1) > 0).astype(int)\n"
            "df = df.dropna()",
            language="python",
        )
        st.markdown("### Latest 10 Rows")
        show_cols = [c for c in ["Close", "returns", "momentum_signal",
                                  "mr_signal", "z_score", "ml_proba"]
                     if c in df.columns]
        latest = df[show_cols].tail(10).copy()
        latest["returns"] = (latest["returns"] * 100).round(3)
        if "ml_proba" in latest.columns:
            latest["ml_proba"] = latest["ml_proba"].round(4)
        if "z_score" in latest.columns:
            latest["z_score"] = latest["z_score"].round(3)
        st.dataframe(
            latest.style.format({"Close": "${:,.0f}", "returns": "{:.3f}%"}),
            use_container_width=True,
        )
