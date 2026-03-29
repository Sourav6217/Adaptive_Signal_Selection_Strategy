import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Adaptive Signal Selection | Sourav Manna",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.2s;
    font-size: 0.9rem;
}
section[data-testid="stSidebar"] .stRadio label:hover { background: #334155; }

/* ── Main background ── */
.main { background: #0f172a; }
.block-container { padding: 2rem 2.5rem; max-width: 1400px; }

/* ── Cards ── */
.card {
    background: linear-gradient(135deg, #1e293b 0%, #162032 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-accent {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2847 100%);
    border: 1px solid #2563eb44;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-green {
    background: linear-gradient(135deg, #052e16 0%, #071a0e 100%);
    border: 1px solid #16a34a44;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-red {
    background: linear-gradient(135deg, #2d0000 0%, #1a0000 100%);
    border: 1px solid #dc262644;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}
.card-amber {
    background: linear-gradient(135deg, #1c1400 0%, #100c00 100%);
    border: 1px solid #d9770044;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
}

/* ── Metric tiles ── */
.metric-tile {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; line-height: 1.1; }
.metric-label { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-delta { font-size: 0.85rem; margin-top: 6px; }

/* ── Typography ── */
h1, h2, h3, h4 { color: #f1f5f9 !important; }
p, li { color: #cbd5e1; }
.section-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.section-sub { color: #64748b; font-size: 0.95rem; margin-bottom: 2rem; }
.tag {
    display: inline-block;
    background: #1e3a5f;
    color: #60a5fa;
    border: 1px solid #2563eb44;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 2px;
}
.tag-green { background: #052e16; color: #4ade80; border-color: #16a34a44; }
.tag-red   { background: #2d0000; color: #f87171; border-color: #dc262644; }
.tag-amber { background: #1c1400; color: #fbbf24; border-color: #d9770044; }
.insight-box {
    background: #1e293b;
    border-left: 4px solid #60a5fa;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    color: #cbd5e1;
    font-size: 0.92rem;
}
.warn-box {
    background: #1c1400;
    border-left: 4px solid #fbbf24;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    color: #fde68a;
    font-size: 0.92rem;
}
.formula {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-family: 'Fira Code', monospace;
    color: #a5f3fc;
    font-size: 0.92rem;
    margin: 0.5rem 0;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa 0%, #818cf8 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    margin-top: 0.5rem;
    line-height: 1.6;
}
.divider { border: none; border-top: 1px solid #334155; margin: 1.5rem 0; }
.stPlotlyChart { border-radius: 12px; overflow: hidden; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOTLY TEMPLATE
# ─────────────────────────────────────────────
DARK_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(family="Inter", color="#94a3b8"),
        title=dict(font=dict(color="#f1f5f9", size=16, family="Inter")),
        xaxis=dict(gridcolor="#1e293b", linecolor="#334155", tickcolor="#334155", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#1e293b", linecolor="#334155", tickcolor="#334155", zerolinecolor="#334155"),
        legend=dict(bgcolor="#1e293b", bordercolor="#334155", borderwidth=1,
                    font=dict(color="#e2e8f0", size=12)),
        margin=dict(l=50, r=30, t=50, b=50),
        hovermode="x unified",
    )
)

COLOR = dict(
    blue="#60a5fa", indigo="#818cf8", purple="#c084fc",
    green="#4ade80", red="#f87171", amber="#fbbf24",
    teal="#2dd4bf", gray="#94a3b8", white="#f1f5f9",
)

# ─────────────────────────────────────────────
#  DATA LOADING (CACHED)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    raw = yf.download("BTC-USD", start="2019-01-01", interval="1d", progress=False)
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns.name = None
    df["returns"] = df["Close"].pct_change()
    df = df.dropna().copy()

    # ── Signal 1: Momentum ──
    df["ma_short"] = df["Close"].rolling(10).mean()
    df["ma_long"]  = df["Close"].rolling(30).mean()
    df["momentum_signal"] = (df["ma_short"] > df["ma_long"]).astype(int).replace({0: -1})

    # ── Signal 2: Mean Reversion ──
    df["rolling_mean"] = df["Close"].rolling(20).mean()
    df["rolling_std"]  = df["Close"].rolling(20).std()
    df["z_score"] = (df["Close"] - df["rolling_mean"]) / df["rolling_std"]
    df["mr_signal"] = 0
    df.loc[df["z_score"] >  1, "mr_signal"] = -1
    df.loc[df["z_score"] < -1, "mr_signal"] =  1

    # ── Signal 3: ML ──
    df["target"] = (df["returns"].shift(-1) > 0).astype(int)
    df["lag1"] = df["returns"].shift(1)
    df["lag2"] = df["returns"].shift(2)
    df["lag3"] = df["returns"].shift(3)
    df["volatility"] = df["returns"].rolling(10).std()
    df = df.dropna().copy()

    feats = ["lag1", "lag2", "lag3", "volatility"]
    X, y = df[feats], df["target"]
    split = int(len(df) * 0.70)
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X.iloc[:split], y.iloc[:split])
    df["ml_pred"]   = model.predict(X)
    df["ml_signal"] = df["ml_pred"].replace({0: -1, 1: 1})
    df["ml_proba"]  = model.predict_proba(X)[:, 1]

    # ── Strategy Returns ──
    df["strategy_returns"]    = df["momentum_signal"].shift(1) * df["returns"]
    df["mr_strategy_returns"] = df["mr_signal"].shift(1) * df["returns"]
    df["ml_strategy_returns"] = df["ml_signal"].shift(1) * df["returns"]

    # ── Cumulative Returns ──
    df["cum_market"]   = (1 + df["returns"]).cumprod()
    df["cum_strategy"] = (1 + df["strategy_returns"]).cumprod()
    df["cum_mr"]       = (1 + df["mr_strategy_returns"]).cumprod()
    df["cum_ml"]       = (1 + df["ml_strategy_returns"]).cumprod()

    # ── UCB Bandit (Naive) ──
    signals_df = df[["momentum_signal", "mr_signal", "ml_signal"]].copy()
    arms = ["momentum_signal", "mr_signal", "ml_signal"]
    counts = {a: 1 for a in arms}
    values = {a: 0.0 for a in arms}
    selected_arms, bandit_rets = [], []
    for i in range(len(df)):
        ucb = {a: values[a] + np.sqrt(2 * np.log(i + 1) / counts[a]) for a in arms}
        arm = max(ucb, key=ucb.get)
        selected_arms.append(arm)
        sig = signals_df[arm].iloc[i]
        ret = sig * df["returns"].iloc[i]
        bandit_rets.append(ret)
        counts[arm] += 1
        values[arm] += (ret - values[arm]) / counts[arm]
    df["bandit_returns"] = bandit_rets
    df["cum_bandit"] = (1 + df["bandit_returns"]).cumprod()
    df["selected_arm"] = selected_arms

    # ── UCB Bandit (Improved) ──
    counts2 = {a: 1 for a in arms}
    values2 = {a: 0.0 for a in arms}
    bandit_rets2 = []
    for i in range(len(df)):
        ucb2 = {a: values2[a] + np.sqrt(1 * np.log(i + 1) / counts2[a]) for a in arms}
        arm2 = max(ucb2, key=ucb2.get)
        sig2 = signals_df[arm2].iloc[i]
        if abs(sig2) < 0.1:
            sig2 = 0
        ret2 = sig2 * df["returns"].iloc[i]
        reward2 = ret2 if ret2 > 0 else 0
        bandit_rets2.append(ret2)
        counts2[arm2] += 1
        values2[arm2] += (reward2 - values2[arm2]) / counts2[arm2]
    df["bandit_imp_returns"] = bandit_rets2
    df["cum_bandit_imp"] = (1 + df["bandit_imp_returns"]).cumprod()

    # ── Metrics helper ──
    def metrics(ret_col):
        r = df[ret_col].dropna()
        sharpe = np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else 0
        cum = df[ret_col.replace("returns", "cum")
                  .replace("strategy_returns", "cum_strategy")
                  .replace("mr_strategy_returns", "cum_mr")
                  .replace("ml_strategy_returns", "cum_ml")
                  .replace("bandit_returns", "cum_bandit")
                  .replace("bandit_imp_returns", "cum_bandit_imp")
                 ]
        total = cum.iloc[-1] if not cum.empty else np.nan
        roll_max = cum.cummax()
        dd = (cum - roll_max) / roll_max
        mdd = dd.min()
        wins = (r > 0).sum() / len(r) * 100
        return dict(sharpe=round(sharpe, 3), total=round(float(total), 3),
                    mdd=round(float(mdd) * 100, 1), win=round(float(wins), 1))

    perf = {
        "Market":      metrics("returns"),
        "Momentum":    metrics("strategy_returns"),
        "Mean Rev":    metrics("mr_strategy_returns"),
        "Bandit":      metrics("bandit_returns"),
        "Bandit+":     metrics("bandit_imp_returns"),
    }
    # ML manually — cum_ml key mismatch
    r_ml = df["ml_strategy_returns"]
    perf["ML"] = dict(
        sharpe=round(float(np.sqrt(252) * r_ml.mean() / r_ml.std()), 2),
        total=round(float(df["cum_ml"].iloc[-1]), 3),
        mdd=round(float(((df["cum_ml"] - df["cum_ml"].cummax()) / df["cum_ml"].cummax()).min() * 100), 1),
        win=round(float((r_ml > 0).sum() / len(r_ml) * 100), 1)
    )
    return df, perf, model, feats, split


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-size:2.2rem;'>📈</div>
        <div style='font-size:1rem; font-weight:700; color:#f1f5f9; margin-top:6px;'>Adaptive Signal Selection</div>
        <div style='font-size:0.72rem; color:#64748b; margin-top:4px;'>PGDM-RBA Capstone 2024–26</div>
    </div>
    """, unsafe_allow_html=True)

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

    st.markdown("<hr style='border-color:#334155; margin:1.2rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; text-align:center; line-height:1.8;'>
        <b style='color:#94a3b8;'>Sourav Manna</b><br>
        PGDM-RBA (2024–26)<br>
        Guide: Prof. Swapnil Arun Desai<br>
        L.N. Welingkar Institute
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("🔄 Fetching BTC-USD data from Yahoo Finance…"):
    df, perf, model, feats, split = load_data()

# ─────────────────────────────────────────────
#  HELPER COMPONENTS
# ─────────────────────────────────────────────
def metric_card(label, value, delta=None, color="#60a5fa", delta_color=None):
    dc = delta_color or ("#4ade80" if (delta or "").startswith("+") or
                         (not (delta or "").startswith("-") and delta) else "#f87171")
    delta_html = f"<div class='metric-delta' style='color:{dc};'>{delta}</div>" if delta else ""
    return f"""
    <div class='metric-tile'>
        <div class='metric-value' style='color:{color};'>{value}</div>
        <div class='metric-label'>{label}</div>
        {delta_html}
    </div>"""


def section_header(icon, title, subtitle):
    st.markdown(f"""
    <div style='margin-bottom:1.8rem;'>
        <div style='font-size:0.85rem; color:#64748b; text-transform:uppercase;
                    letter-spacing:.1em; margin-bottom:4px;'>{icon}</div>
        <div class='section-title'>{title}</div>
        <div class='section-sub'>{subtitle}</div>
    </div>""", unsafe_allow_html=True)


def apply_template(fig):
    fig.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json())
    return fig


# ═══════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == sections[0]:
    # Hero
    st.markdown("""
    <div class='card' style='padding:2.5rem; margin-bottom:2rem;
         background:linear-gradient(135deg,#0f172a 0%,#1a1040 50%,#0f172a 100%);
         border:1px solid #6366f144;'>
        <div class='hero-title'>Adaptive Signal Selection</div>
        <div class='hero-title' style='font-size:1.9rem;'>under Market Regime Uncertainty</div>
        <div class='hero-sub' style='max-width:720px; margin-top:1rem;'>
            A Multi-Armed Bandit framework that dynamically selects between
            <b style='color:#60a5fa;'>Momentum</b>,
            <b style='color:#4ade80;'>Mean Reversion</b>, and
            <b style='color:#c084fc;'>Machine Learning</b> signals
            to optimise risk-adjusted returns on Bitcoin.
        </div>
        <div style='margin-top:1.5rem;'>
            <span class='tag'>Bitcoin BTC-USD</span>
            <span class='tag'>2019 – 2026</span>
            <span class='tag'>2,644 Trading Days</span>
            <span class='tag'>UCB Algorithm</span>
            <span class='tag'>Random Forest</span>
            <span class='tag'>Risk Analytics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi_data = [
        (c1, "Momentum Sharpe", "+0.43", "Best strategy signal", "#60a5fa"),
        (c2, "Momentum Return", "+155%", "vs Buy-Hold +2,400%", "#4ade80"),
        (c3, "Mean Rev Sharpe", "-0.20", "BTC not mean-reverting", "#f87171"),
        (c4, "Dataset Size", "2,644", "Daily observations", "#818cf8"),
        (c5, "Signals Tested", "3", "Momentum · MR · ML", "#fbbf24"),
    ]
    for col, label, val, delta, clr in kpi_data:
        with col:
            st.markdown(metric_card(label, val, delta, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Project flow diagram
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("### Research Framework")
        steps = [
            ("01", "Data Collection", "BTC-USD daily OHLCV via Yahoo Finance API (2019–2026)", "#60a5fa"),
            ("02", "Signal Construction", "Momentum (MA crossover) · Mean Reversion (Z-score) · ML (Random Forest)", "#818cf8"),
            ("03", "Signal Evaluation", "Sharpe Ratio · Cumulative Returns · Drawdown · Regime Analysis", "#c084fc"),
            ("04", "Bandit Framework", "UCB algorithm adaptively selects the best signal at each time step", "#2dd4bf"),
            ("05", "Performance Analysis", "Compare adaptive vs fixed strategies · identify failure modes", "#4ade80"),
        ]
        for num, title, desc, clr in steps:
            st.markdown(f"""
            <div style='display:flex; gap:1rem; align-items:flex-start;
                        background:#1e293b; border:1px solid #334155;
                        border-radius:12px; padding:1rem 1.2rem; margin-bottom:0.7rem;'>
                <div style='font-size:1.3rem; font-weight:800; color:{clr}; min-width:32px;'>{num}</div>
                <div>
                    <div style='font-weight:600; color:#f1f5f9; font-size:0.95rem;'>{title}</div>
                    <div style='color:#94a3b8; font-size:0.83rem; margin-top:3px;'>{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("### Strategy Quick Scoreboard")
        scoreboard = [
            ("📈 Market (Buy & Hold)", "+2,400%", "~0.80", "Benchmark"),
            ("⚡ Momentum", "+155%", "+0.43", "Best Strategy"),
            ("🔁 Mean Reversion", "-83%", "-0.20", "Fails on BTC"),
            ("🤖 ML (Raw)", "Overfit", "9.11*", "Unreliable"),
            ("🎰 UCB Bandit", "Negative", "Negative", "Worst"),
        ]
        tbl_html = """
        <table style='width:100%;border-collapse:collapse;font-size:0.85rem;'>
        <tr style='background:#334155;'>
            <th style='padding:8px 10px;text-align:left;color:#94a3b8;font-weight:500;'>Strategy</th>
            <th style='padding:8px 10px;text-align:center;color:#94a3b8;font-weight:500;'>Return</th>
            <th style='padding:8px 10px;text-align:center;color:#94a3b8;font-weight:500;'>Sharpe</th>
        </tr>"""
        clrs_row = ["#1a3a2a", "#1e293b", "#3a1a1a", "#1a1a3a", "#2a1a1a"]
        for i, (name, ret, sharpe, _) in enumerate(scoreboard):
            bg = clrs_row[i]
            rc = "#4ade80" if ret.startswith("+") else "#f87171" if ret.startswith("-") else "#fbbf24"
            sc = "#4ade80" if "+" in str(sharpe) and not sharpe.startswith("9") else "#f87171" if sharpe.startswith("-") else "#fbbf24"
            tbl_html += f"""
            <tr style='background:{bg};border-bottom:1px solid #334155;'>
                <td style='padding:8px 10px;color:#e2e8f0;'>{name}</td>
                <td style='padding:8px 10px;text-align:center;color:{rc};font-weight:600;'>{ret}</td>
                <td style='padding:8px 10px;text-align:center;color:{sc};font-weight:600;'>{sharpe}</td>
            </tr>"""
        tbl_html += "</table>"
        st.markdown(f"<div class='card'>{tbl_html}</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class='warn-box'>
            <b>Key Takeaway:</b> No single strategy performs consistently across all BTC market regimes.
            Adaptive selection is theoretically sound — but only works when underlying signals have genuine
            regime-dependent performance.
        </div>""", unsafe_allow_html=True)

    # BTC price preview
    st.markdown("<br>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"],
        mode="lines", name="BTC-USD Close",
        line=dict(color=COLOR["blue"], width=1.5),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.05)"
    ))
    # Annotate key events
    events = [
        ("2020-03-12", "$3,800\nCovid Crash", "#f87171"),
        ("2021-11-10", "$69K\nATH 2021", "#4ade80"),
        ("2022-11-09", "$15,700\nFTX Collapse", "#f87171"),
        ("2024-11-05", "$76K\nElection Rally", "#4ade80"),
    ]
    for date, label, clr in events:
        if date in df.index.astype(str):
            price = df.loc[date, "Close"] if date in df.index.astype(str) else None
            if price:
                fig.add_vline(x=date, line_dash="dot", line_color=clr, line_width=1, opacity=0.5)
    fig.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                      title="Bitcoin (BTC-USD) Price History — Full Study Period",
                      height=320, yaxis_title="Price (USD)",
                      yaxis=dict(tickprefix="$", gridcolor="#1e293b"))
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: DATA & EDA
# ═══════════════════════════════════════════════════════════
elif page == sections[1]:
    section_header("📊 DATA & EXPLORATORY ANALYSIS",
                   "Understanding the BTC-USD Dataset",
                   "Statistical properties, return distributions, and volatility analysis — 2019 to 2026")

    # Summary stats
    close = df["Close"]
    rets  = df["returns"]
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (c1, [("Observations", f"{len(df):,}"), ("Start", str(df.index[0].date())), ("End", str(df.index[-1].date()))]),
        (c2, [("Min Price", f"${close.min():,.0f}"), ("Max Price", f"${close.max():,.0f}"), ("Mean Price", f"${close.mean():,.0f}")]),
        (c3, [("Daily Mean Ret", f"{rets.mean()*100:.3f}%"), ("Daily Std Dev", f"{rets.std()*100:.2f}%"), ("Ann. Volatility", f"{rets.std()*np.sqrt(252)*100:.1f}%")]),
        (c4, [("Skewness", f"{rets.skew():.3f}"), ("Kurtosis", f"{rets.kurt():.3f}"), ("Positive Days", f"{(rets>0).mean()*100:.1f}%")]),
    ]
    for col, items in stats:
        with col:
            html = "<div class='card'>"
            for k, v in items:
                html += f"<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #334155;'><span style='color:#94a3b8;font-size:0.82rem;'>{k}</span><span style='color:#f1f5f9;font-weight:600;font-size:0.88rem;'>{v}</span></div>"
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

    # Price + Volume chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="BTC Close",
                             line=dict(color=COLOR["blue"], width=1.5),
                             fill="tozeroy", fillcolor="rgba(96,165,250,0.08)"), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                         marker_color=COLOR["indigo"], opacity=0.5), row=2, col=1)
    fig.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                      title="BTC-USD: Price & Volume (2019–2026)", height=480,
                      yaxis=dict(tickprefix="$", gridcolor="#1e293b"),
                      yaxis2=dict(gridcolor="#1e293b"))
    st.plotly_chart(fig, use_container_width=True)

    # Returns analysis
    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df["returns"] * 100,
                                  mode="lines", name="Daily Returns",
                                  line=dict(color=COLOR["teal"], width=0.8)))
        fig2.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                           title="Daily Returns (%) — Volatility Clustering Visible",
                           height=300, yaxis_title="Return (%)")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=df["returns"] * 100, nbinsx=120,
                                    name="Return Distribution",
                                    marker_color=COLOR["indigo"],
                                    opacity=0.8))
        # Normal overlay
        mu, sigma = df["returns"].mean() * 100, df["returns"].std() * 100
        x_norm = np.linspace(-20, 20, 200)
        y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2)
        y_norm = y_norm * len(df) * (40 / 120)
        fig3.add_trace(go.Scatter(x=x_norm, y=y_norm, mode="lines", name="Normal Dist.",
                                  line=dict(color=COLOR["amber"], width=2, dash="dash")))
        fig3.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                           title="Return Distribution — Fat Tails vs Normal",
                           height=300, xaxis_title="Return (%)", yaxis_title="Frequency")
        st.plotly_chart(fig3, use_container_width=True)

    # Rolling volatility
    df["roll_vol_30"] = df["returns"].rolling(30).std() * np.sqrt(252) * 100
    df["roll_vol_90"] = df["returns"].rolling(90).std() * np.sqrt(252) * 100
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=df.index, y=df["roll_vol_30"], name="30-day Ann. Vol",
                              line=dict(color=COLOR["purple"], width=1.2),
                              fill="tozeroy", fillcolor="rgba(192,132,252,0.08)"))
    fig4.add_trace(go.Scatter(x=df.index, y=df["roll_vol_90"], name="90-day Ann. Vol",
                              line=dict(color=COLOR["amber"], width=1.5, dash="dot")))
    fig4.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                       title="Rolling Annualised Volatility — Regime Shifts Visible",
                       height=300, yaxis_title="Volatility (%)",
                       yaxis=dict(ticksuffix="%", gridcolor="#1e293b"))
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
        <b style='color:#60a5fa;'>📌 Key EDA Insights</b><br>
        <b>Fat Tails:</b> Kurtosis > 5 confirms extreme returns are far more frequent than a normal distribution predicts — risk models assuming normality will underestimate tail risk. &nbsp;|&nbsp;
        <b>Volatility Clustering:</b> High-vol periods cluster together (2020 COVID, 2022 FTX), consistent with GARCH dynamics. &nbsp;|&nbsp;
        <b>Positive Drift:</b> Mean daily return ~+0.18% translates to ~+45% annualised, creating a structural advantage for long-biased momentum strategies.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: MOMENTUM
# ═══════════════════════════════════════════════════════════
elif page == sections[2]:
    section_header("🚀 SIGNAL 1",
                   "Momentum Strategy",
                   "Moving Average Crossover (10-day vs 30-day) — the trend-following signal")

    col1, col2, col3, col4 = st.columns(4)
    m = perf["Momentum"]
    for col, label, val, delta, clr in [
        (col1, "Sharpe Ratio", f"+{m['sharpe']}", "Positive risk-adjusted return", "#4ade80"),
        (col2, "Cumulative Return", f"{m['total']:.2f}x", "+155% over 7 years", "#60a5fa"),
        (col3, "Max Drawdown", f"{m['mdd']:.1f}%", "Worst peak-to-trough", "#f87171"),
        (col4, "Win Rate", f"{m['win']:.1f}%", "Days with positive return", "#818cf8"),
    ]:
        with col:
            st.markdown(metric_card(label, val, delta, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # MA crossover + signals
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="BTC Price",
                             line=dict(color=COLOR["gray"], width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_short"], name="MA(10) Short",
                             line=dict(color=COLOR["amber"], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["ma_long"], name="MA(30) Long",
                             line=dict(color=COLOR["blue"], width=1.5)), row=1, col=1)

    buy_signal  = df[df["momentum_signal"] ==  1]
    sell_signal = df[df["momentum_signal"] == -1]

    fig.add_trace(go.Scatter(x=df.index,
                             y=df["momentum_signal"].map({1: 1, -1: -1}),
                             name="Signal (+1/-1)", mode="lines",
                             line=dict(color=COLOR["teal"], width=1.5),
                             fill="tozeroy", fillcolor="rgba(45,212,191,0.1)"), row=2, col=1)

    fig.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                      title="Momentum Signal: MA(10) vs MA(30) Crossover",
                      height=500, yaxis=dict(tickprefix="$", gridcolor="#1e293b"),
                      yaxis2=dict(gridcolor="#1e293b", tickvals=[-1, 0, 1]))
    st.plotly_chart(fig, use_container_width=True)

    # Cumulative returns
    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df["cum_market"], name="Market (Buy & Hold)",
                                  line=dict(color=COLOR["blue"], width=2),
                                  fill="tozeroy", fillcolor="rgba(96,165,250,0.05)"))
        fig2.add_trace(go.Scatter(x=df.index, y=df["cum_strategy"], name="Momentum Strategy",
                                  line=dict(color=COLOR["amber"], width=2)))
        fig2.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                           title="Cumulative Returns: Momentum vs Market",
                           height=350, yaxis_title="Wealth Multiple (1 = Initial)")
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("### Signal Logic")
        st.markdown("""
        <div class='formula'>MA_short = Close.rolling(10).mean()<br>
MA_long  = Close.rolling(30).mean()<br><br>
Signal = +1  if MA_short > MA_long<br>
Signal = -1  if MA_short ≤ MA_long<br><br>
Strategy_return_t = Signal_{t-1} × Return_t</div>

<div class='insight-box' style='margin-top:1rem;'>
<b>Why it works on BTC:</b> Bitcoin exhibits strong momentum due to speculative demand cycles and retail herding. Trend-following captures sustained directional moves.
</div>
<div class='warn-box'>
<b>Limitation:</b> Whipsaws during choppy markets and late entries/exits during sharp reversals reduce performance vs buy-and-hold.
</div>""", unsafe_allow_html=True)

    # Rolling Sharpe
    roll_ret = df["strategy_returns"].rolling(90)
    rolling_sharpe = np.sqrt(252) * roll_ret.mean() / roll_ret.std()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=df.index, y=rolling_sharpe, name="90-day Rolling Sharpe",
                              line=dict(color=COLOR["green"], width=1.5)))
    fig3.add_hline(y=0, line_dash="dot", line_color=COLOR["red"], opacity=0.6)
    fig3.add_hline(y=0.43, line_dash="dash", line_color=COLOR["amber"],
                   opacity=0.7, annotation_text="Full-Period Sharpe: 0.43")
    fig3.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                       title="Rolling Sharpe Ratio (90-day Window) — Regime-Dependent Performance",
                       height=280, yaxis_title="Sharpe Ratio")
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: MEAN REVERSION
# ═══════════════════════════════════════════════════════════
elif page == sections[3]:
    section_header("🔁 SIGNAL 2",
                   "Mean Reversion Strategy",
                   "Z-Score based contrarian signal — buy dips, sell spikes")

    col1, col2, col3, col4 = st.columns(4)
    m = perf["Mean Rev"]
    for col, label, val, delta, clr in [
        (col1, "Sharpe Ratio", f"{m['sharpe']}", "Negative — loses vs risk", "#f87171"),
        (col2, "Cumulative Return", f"{m['total']:.2f}x", "-83% catastrophic loss", "#f87171"),
        (col3, "Max Drawdown", f"{m['mdd']:.1f}%", "Extreme losses", "#f87171"),
        (col4, "Win Rate", f"{m['win']:.1f}%", "Below 50% baseline", "#f87171"),
    ]:
        with col:
            st.markdown(metric_card(label, val, delta, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Z-score plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="BTC Price",
                             line=dict(color=COLOR["gray"], width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["rolling_mean"], name="20-day Mean",
                             line=dict(color=COLOR["amber"], width=1.2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["z_score"], name="Z-Score",
                             line=dict(color=COLOR["purple"], width=1.2)), row=2, col=1)
    fig.add_hline(y=1,  line_dash="dot", line_color=COLOR["red"],   row=2, opacity=0.7)
    fig.add_hline(y=-1, line_dash="dot", line_color=COLOR["green"], row=2, opacity=0.7)
    fig.add_hline(y=0,  line_dash="solid", line_color=COLOR["gray"], row=2, opacity=0.3)
    fig.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                      title="Z-Score Signal: Price Deviation from 20-day Rolling Mean",
                      height=480, yaxis=dict(tickprefix="$", gridcolor="#1e293b"),
                      yaxis2=dict(gridcolor="#1e293b"))
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index, y=df["cum_market"], name="Market",
                                  line=dict(color=COLOR["blue"], width=2)))
        fig2.add_trace(go.Scatter(x=df.index, y=df["cum_mr"], name="Mean Reversion",
                                  line=dict(color=COLOR["red"], width=2),
                                  fill="tozeroy", fillcolor="rgba(248,113,113,0.07)"))
        fig2.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                           title="Mean Reversion Strategy vs Market",
                           height=320, yaxis_title="Wealth Multiple")
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown("### Signal Logic")
        st.markdown("""
        <div class='formula'>rolling_mean = Close.rolling(20).mean()
rolling_std  = Close.rolling(20).std()

Z_t = (Close_t - rolling_mean) / rolling_std

Signal = -1  if Z_t > +1  (sell: expect ↓)
Signal = +1  if Z_t < -1  (buy:  expect ↑)
Signal =  0  otherwise</div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class='card-red' style='margin-top:1rem;'>
            <b style='color:#f87171;'>❌ Why It Fails on BTC</b><br>
            <ul style='color:#fca5a5; margin-top:8px; padding-left:1.2rem; font-size:0.88rem;'>
                <li>BTC is a <b>momentum-dominated</b> asset — prices trend, not revert</li>
                <li>Selling after Z > 1 means <b>selling into a bull run</b></li>
                <li>Buying after Z &lt; -1 means <b>catching falling knives</b></li>
                <li>Speculative demand creates <b>extended departures</b> from mean</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: ML
# ═══════════════════════════════════════════════════════════
elif page == sections[4]:
    section_header("🤖 SIGNAL 3",
                   "Machine Learning — Random Forest",
                   "Predicting next-day BTC direction using lagged returns & volatility features")

    col1, col2, col3, col4 = st.columns(4)
    for col, label, val, delta, clr in [
        (col1, "Model", "Random Forest", "100 trees, max_depth=5", "#818cf8"),
        (col2, "Features", "4", "lag1, lag2, lag3, volatility", "#60a5fa"),
        (col3, "Train Split", "70%", "Time-based, no shuffle", "#4ade80"),
        (col4, "Task", "Binary", "Direction: Up (1) / Down (0)", "#fbbf24"),
    ]:
        with col:
            st.markdown(metric_card(label, val, delta, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        # Feature importances
        importances = model.feature_importances_
        fig = go.Figure(go.Bar(
            x=importances, y=feats, orientation="h",
            marker=dict(color=[COLOR["blue"], COLOR["indigo"], COLOR["purple"], COLOR["teal"]]),
        ))
        fig.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                          title="Feature Importance — Random Forest",
                          height=280, xaxis_title="Importance Score")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # ML probability distribution
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=df["ml_proba"], nbinsx=50,
                                    marker_color=COLOR["purple"], opacity=0.8,
                                    name="P(Up) Distribution"))
        fig2.add_vline(x=0.5, line_dash="dash", line_color=COLOR["amber"],
                       annotation_text="Decision Threshold 0.5")
        fig2.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                           title="ML Predicted P(Price Up Tomorrow)",
                           height=280, xaxis_title="Probability", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

    # ML signal over time
    fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                         vertical_spacing=0.05)
    fig3.add_trace(go.Scatter(x=df.index, y=df["Close"], name="BTC Price",
                              line=dict(color=COLOR["gray"], width=1)), row=1, col=1)
    fig3.add_trace(go.Scatter(x=df.index, y=df["ml_proba"], name="P(Up)",
                              line=dict(color=COLOR["purple"], width=0.8),
                              fill="tozeroy", fillcolor="rgba(192,132,252,0.1)"), row=2, col=1)
    fig3.add_hline(y=0.5, line_dash="dot", line_color=COLOR["amber"], row=2, opacity=0.6)
    fig3.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                       title="ML Model: Predicted Probability of Upward Move",
                       height=420, yaxis=dict(tickprefix="$"),
                       yaxis2=dict(tickformat=".0%"))
    st.plotly_chart(fig3, use_container_width=True)

    # Confusion matrix — test set only
    X_test = df[feats].iloc[split:]
    y_test = df["target"].iloc[split:]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = (cm[0,0] + cm[1,1]) / cm.sum() * 100

    col_c, col_d = st.columns([2, 3])
    with col_c:
        fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual"),
                           x=["Down", "Up"], y=["Down", "Up"],
                           color_continuous_scale=[[0, "#0f172a"], [1, "#818cf8"]],
                           text_auto=True)
        fig_cm.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                             title=f"Confusion Matrix (Test Set) — Accuracy: {acc:.1f}%",
                             height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_d:
        st.markdown("""
        <div class='warn-box'>
            <b>⚠️ Overfitting Warning</b><br>
            Raw backtested ML performance appears extraordinary (Sharpe ~9.11) because predictions
            are generated on training data. <b>Out-of-sample accuracy (~50–52%)</b> near random, confirming
            that lagged returns alone lack sufficient predictive signal for BTC direction.
            Feature engineering with on-chain metrics, sentiment, or macro variables would be required
            for production-grade ML signals.
        </div>
        <div class='insight-box'>
            <b>Volatility is the most important feature</b> — confirming that
            market "state" (high vs low vol regime) is more informative for direction prediction
            than simple lagged returns.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: STRATEGY COMPARISON
# ═══════════════════════════════════════════════════════════
elif page == sections[5]:
    section_header("⚔️ STRATEGY COMPARISON",
                   "Head-to-Head Performance",
                   "Cumulative returns, risk metrics, and regime-dependent behaviour across all signals")

    # Full comparison chart
    norm_start = df.index[0]
    df_n = df.copy()
    for col in ["cum_market", "cum_strategy", "cum_mr"]:
        df_n[col + "_norm"] = df_n[col] / df_n[col].iloc[0]

    fig = go.Figure()
    traces = [
        ("cum_market_norm",   "Market (Buy & Hold)",  COLOR["blue"],   3.0),
        ("cum_strategy_norm", "Momentum",             COLOR["amber"],  2.0),
        ("cum_mr_norm",       "Mean Reversion",       COLOR["red"],    1.5),
    ]
    for col, name, clr, width in traces:
        fig.add_trace(go.Scatter(x=df_n.index, y=df_n[col], name=name,
                                 line=dict(color=clr, width=width)))
    fig.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                      title="Strategy Comparison — Full Period Normalised Returns",
                      height=400, yaxis_title="Wealth Multiple",
                      yaxis_type="log",
                      annotations=[dict(x=0.01, y=0.97, xref="paper", yref="paper",
                                        text="Log Scale", font=dict(color="#64748b", size=11),
                                        showarrow=False)])
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    st.markdown("### Risk-Adjusted Metrics Summary")
    strategies = ["Market", "Momentum", "Mean Rev"]
    rows = []
    for s in strategies:
        m = perf[s]
        ret_str = f"{(m['total']-1)*100:+.0f}%"
        rows.append([s, f"{m['total']:.2f}x", ret_str, f"{m['sharpe']:+.3f}",
                     f"{m['mdd']:.1f}%", f"{m['win']:.1f}%"])

    headers = ["Strategy", "Wealth Multiple", "Total Return", "Sharpe Ratio", "Max Drawdown", "Win Rate"]
    tbl = f"<table style='width:100%;border-collapse:collapse;font-size:0.88rem;'><tr style='background:#334155;'>"
    for h in headers:
        tbl += f"<th style='padding:10px;color:#94a3b8;font-weight:500;text-align:center;'>{h}</th>"
    tbl += "</tr>"
    row_colors = ["#1a3a2a", "#1e293b", "#3a1a1a"]
    for i, row in enumerate(rows):
        tbl += f"<tr style='background:{row_colors[i]};border-bottom:1px solid #334155;'>"
        for j, cell in enumerate(row):
            clr = "#f1f5f9"
            if j == 3:
                clr = "#4ade80" if "+" in cell else "#f87171"
            elif j == 4:
                clr = "#f87171"
            tbl += f"<td style='padding:10px;text-align:center;color:{clr};font-weight:{"600" if j>0 else "400"};'>{cell}</td>"
        tbl += "</tr>"
    tbl += "</table>"
    st.markdown(f"<div class='card'>{tbl}</div>", unsafe_allow_html=True)

    # Drawdown comparison
    col1, col2 = st.columns(2)
    with col1:
        dd_data = {}
        for col, name in [("cum_market", "Market"), ("cum_strategy", "Momentum"), ("cum_mr", "Mean Rev")]:
            s = df[col]
            dd_data[name] = ((s - s.cummax()) / s.cummax() * 100)
        fig2 = go.Figure()
        clrs2 = [COLOR["blue"], COLOR["amber"], COLOR["red"]]
        for (name, dd), clr in zip(dd_data.items(), clrs2):
            fig2.add_trace(go.Scatter(x=df.index, y=dd, name=name,
                                      line=dict(color=clr, width=1.2),
                                      fill="tozeroy", fillcolor=clr.replace(")", ",0.07)").replace("rgb", "rgba") if "rgb" in clr else clr + "12"))
        fig2.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                           title="Drawdown Comparison (%)",
                           height=320, yaxis_title="Drawdown (%)",
                           yaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Rolling 90-day Sharpe comparison
        fig3 = go.Figure()
        for col, name, clr in [("strategy_returns", "Momentum", COLOR["amber"]),
                                ("mr_strategy_returns", "Mean Reversion", COLOR["red"])]:
            roll = df[col].rolling(90)
            rs = np.sqrt(252) * roll.mean() / roll.std()
            fig3.add_trace(go.Scatter(x=df.index, y=rs, name=name,
                                      line=dict(color=clr, width=1.2)))
        fig3.add_hline(y=0, line_dash="dash", line_color=COLOR["gray"], opacity=0.5)
        fig3.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                           title="Rolling 90-day Sharpe Comparison",
                           height=320, yaxis_title="Sharpe Ratio")
        st.plotly_chart(fig3, use_container_width=True)

    # Regime performance heatmap
    st.markdown("### Regime-Dependent Performance Matrix")
    regimes = ["Strong Uptrend\n(2020-21)", "Sharp Correction\n(Nov 21)", "Bear Market\n(2022)", "Recovery\n(2023)", "Bull II\n(2024-25)"]
    strats = ["Momentum", "Mean Rev", "Buy & Hold"]
    matrix = [[3, -2, 2, 3, 3], [-3, -2, -3, -3, -3], [4, -3, -4, 4, 4]]
    labels = [["Good", "Bad", "Moderate", "Good", "Good"],
              ["Very Bad", "Bad", "Very Bad", "Bad", "Very Bad"],
              ["Best ★", "Drawdown", "Best ★", "Best ★", "Best ★"]]

    fig4 = go.Figure(data=go.Heatmap(
        z=matrix, x=regimes, y=strats,
        colorscale=[[0, "#2d0000"], [0.35, "#3a1a1a"], [0.5, "#1e293b"], [0.65, "#1a3a2a"], [1, "#052e16"]],
        text=labels, texttemplate="%{text}", showscale=False,
    ))
    fig4.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                       title="Qualitative Performance by Market Regime",
                       height=260)
    st.plotly_chart(fig4, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: BANDIT
# ═══════════════════════════════════════════════════════════
elif page == sections[6]:
    section_header("🎰 MULTI-ARMED BANDIT",
                   "UCB Adaptive Signal Selection",
                   "Upper Confidence Bound algorithm dynamically allocates across the three signal arms")

    # UCB explainer
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("### How UCB Works")
        st.markdown("""
        <div class='formula'>UCB_arm(t) = Q_arm(t) + √( c · ln(t) / N_arm(t) )

Where:
  Q_arm = Average reward from arm so far
  N_arm = Times this arm has been selected
  c     = Exploration constant (2.0 naive, 1.0 improved)
  t     = Current time step

→ Select: arm = argmax UCB_arm(t)</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='display:flex; gap:0.7rem; margin-top:1rem;'>
            <div class='card' style='flex:1; text-align:center;'>
                <div style='font-size:1.6rem;'>🎯</div>
                <div style='color:#60a5fa;font-weight:600;font-size:0.9rem;'>Exploit</div>
                <div style='color:#94a3b8;font-size:0.78rem;'>Select best known arm</div>
            </div>
            <div class='card' style='flex:1; text-align:center;'>
                <div style='font-size:1.6rem;'>🔍</div>
                <div style='color:#818cf8;font-weight:600;font-size:0.9rem;'>Explore</div>
                <div style='color:#94a3b8;font-size:0.78rem;'>Try uncertain arms</div>
            </div>
            <div class='card' style='flex:1; text-align:center;'>
                <div style='font-size:1.6rem;'>⚖️</div>
                <div style='color:#4ade80;font-weight:600;font-size:0.9rem;'>Balance</div>
                <div style='color:#94a3b8;font-size:0.78rem;'>Confidence bound manages trade-off</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("### Performance Metrics")
        for name, m, clr in [("Naive Bandit", perf["Bandit"], "#f87171"),
                               ("Improved Bandit", perf["Bandit+"], "#fbbf24")]:
            st.markdown(f"""
            <div class='card' style='border-color:{clr}33; margin-bottom:0.7rem;'>
                <div style='color:{clr}; font-weight:600; margin-bottom:8px;'>{name}</div>
                <div style='display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:0.83rem;'>
                    <div><span style='color:#64748b;'>Sharpe:</span> <span style='color:{clr};font-weight:600;'>{m["sharpe"]}</span></div>
                    <div><span style='color:#64748b;'>Total Ret:</span> <span style='color:{clr};font-weight:600;'>{m["total"]:.2f}x</span></div>
                    <div><span style='color:#64748b;'>Max DD:</span> <span style='color:#f87171;font-weight:600;'>{m["mdd"]:.1f}%</span></div>
                    <div><span style='color:#64748b;'>Win Rate:</span> <span style='color:#94a3b8;font-weight:600;'>{m["win"]:.1f}%</span></div>
                </div>
            </div>""", unsafe_allow_html=True)

    # All strategies including bandit
    fig = go.Figure()
    for col, name, clr, width in [
        ("cum_market",    "Market",          COLOR["blue"],   2.5),
        ("cum_strategy",  "Momentum",        COLOR["amber"],  2.0),
        ("cum_mr",        "Mean Reversion",  COLOR["green"],  1.5),
        ("cum_bandit",    "Naive Bandit",    COLOR["red"],    1.5),
        ("cum_bandit_imp","Improved Bandit", COLOR["purple"], 2.0),
    ]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name,
                                 line=dict(color=clr, width=width)))
    fig.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                      title="All Strategies vs Adaptive Bandit — Full Period (Log Scale)",
                      height=420, yaxis_title="Wealth Multiple",
                      yaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)

    # Arm selection frequency
    arm_counts = pd.Series(df["selected_arm"]).value_counts()
    arm_labels = {"momentum_signal": "Momentum", "mr_signal": "Mean Reversion", "ml_signal": "ML"}
    arm_counts.index = [arm_labels.get(x, x) for x in arm_counts.index]

    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure(go.Pie(
            labels=arm_counts.index,
            values=arm_counts.values,
            hole=0.55,
            marker=dict(colors=[COLOR["amber"], COLOR["red"], COLOR["purple"]]),
        ))
        fig2.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                           title="Arm Selection Frequency (Naive UCB)",
                           height=300,
                           annotations=[dict(text=f"{len(df):,}<br>steps", x=0.5, y=0.5,
                                             font=dict(size=14, color="#f1f5f9"), showarrow=False)])
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Rolling arm selection over time
        arm_df = pd.get_dummies(df["selected_arm"])
        arm_df.columns = [arm_labels.get(c, c) for c in arm_df.columns]
        roll_arms = arm_df.rolling(90).mean()
        fig3 = go.Figure()
        for col_name, clr in zip(arm_df.columns, [COLOR["amber"], COLOR["red"], COLOR["purple"]]):
            if col_name in roll_arms.columns:
                fig3.add_trace(go.Scatter(x=df.index, y=roll_arms[col_name],
                                          name=col_name, line=dict(color=clr, width=1.5),
                                          fill="tozeroy",
                                          fillcolor=clr + "20" if len(clr) == 7 else clr))
        fig3.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                           title="Rolling 90-day Arm Selection Share",
                           height=300, yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig3, use_container_width=True)

    # Failure modes
    st.markdown("### Why the Bandit Underperforms — Failure Mode Analysis")
    failures = [
        ("Signal Pool Quality", "Mean Reversion is structurally loss-making on BTC. Forced UCB exploration of a universally bad arm creates permanent drag.", "#f87171"),
        ("Non-Stationarity",    "UCB1 assumes stationary reward distributions. BTC regime shifts invalidate historical Q estimates — the algorithm is 'fighting the last war'.", "#fbbf24"),
        ("Exploration Cost",    "In a bull market, exploring short-biased strategies during upward moves is disproportionately costly. UCB's exploration bonus is theoretically optimal under i.i.d. rewards, not in directional markets.", "#f87171"),
        ("Delayed Feedback",    "Financial returns are noisy and autocorrelated, violating the i.i.d. reward assumption. The bandit misattributes market noise to signal quality.", "#fbbf24"),
        ("Context Blindness",   "Standard UCB ignores observable regime features (VIX, trend strength). A contextual bandit (LinUCB) could condition selection on market state.", "#60a5fa"),
    ]
    for title, desc, clr in failures:
        st.markdown(f"""
        <div style='background:#1e293b;border:1px solid {clr}33;border-left:3px solid {clr};
                    border-radius:0 12px 12px 0;padding:0.9rem 1.1rem;margin-bottom:0.6rem;'>
            <span style='color:{clr};font-weight:600;'>{title}:</span>
            <span style='color:#94a3b8;font-size:0.88rem;'> {desc}</span>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: KEY FINDINGS
# ═══════════════════════════════════════════════════════════
elif page == sections[7]:
    section_header("💡 KEY FINDINGS",
                   "Research Insights & Conclusions",
                   "What this study reveals about adaptive signal selection in cryptocurrency markets")

    findings = [
        {
            "num": "01", "color": "#f87171",
            "title": "No Single Strategy is Consistently Superior",
            "body": "Momentum achieves Sharpe +0.43, Mean Reversion -0.20. Performance is strongly regime-dependent, validating the theoretical case for adaptive selection.",
            "tags": ["Validated", "Regime Dependence", "Core Finding"],
            "tag_colors": ["tag-green", "tag", "tag"],
        },
        {
            "num": "02", "color": "#60a5fa",
            "title": "BTC is a Momentum-Dominated Asset",
            "body": "The stark contrast between Momentum (+155%) and Mean Reversion (-83%) confirms Bitcoin's speculative, trend-following nature driven by retail herding and demand cycles.",
            "tags": ["Asset-Specific", "Momentum", "High Confidence"],
            "tag_colors": ["tag", "tag-green", "tag-green"],
        },
        {
            "num": "03", "color": "#fbbf24",
            "title": "Adaptive Bandits Require Strong Signal Quality",
            "body": "The UCB bandit underperforms all strategies when the signal pool contains a universally loss-making arm. Signal quality is a prerequisite, not an output, of adaptive frameworks.",
            "tags": ["Critical Insight", "Novel Contribution", "Practical Impact"],
            "tag_colors": ["tag-amber", "tag-amber", "tag"],
        },
        {
            "num": "04", "color": "#c084fc",
            "title": "UCB Exploration is Costly in Directional Markets",
            "body": "Forced exploration of short-biased strategies during sustained bull markets generates compounding losses. UCB's i.i.d. reward assumption breaks down in trending financial markets.",
            "tags": ["Theoretical Insight", "UCB Limitation"],
            "tag_colors": ["tag", "tag-red"],
        },
        {
            "num": "05", "color": "#4ade80",
            "title": "ML Signals Demand Rigorous Out-of-Sample Validation",
            "body": "Spurious Sharpe of 9.11 from in-sample ML evaluation warns against naive backtest evaluation. Walk-forward validation and regularisation are non-negotiable for financial ML.",
            "tags": ["Practitioner Warning", "ML Best Practice"],
            "tag_colors": ["tag-red", "tag"],
        },
        {
            "num": "06", "color": "#2dd4bf",
            "title": "Signal Improvement Must Precede Framework Improvement",
            "body": "Bandit improvements (signal filtering, asymmetric rewards, reduced exploration) produced marginal gains. The correct order: (1) improve signals, (2) ensure diversity, (3) add adaptive layer.",
            "tags": ["Design Principle", "Actionable"],
            "tag_colors": ["tag-green", "tag-green"],
        },
    ]

    for f in findings:
        tags_html = " ".join([f"<span class='{tc}'>{t}</span>" for t, tc in zip(f["tags"], f["tag_colors"])])
        st.markdown(f"""
        <div class='card' style='border-left:4px solid {f["color"]}; margin-bottom:1.2rem;'>
            <div style='display:flex;gap:1.2rem;align-items:flex-start;'>
                <div style='font-size:2rem;font-weight:800;color:{f["color"]};
                            min-width:48px;line-height:1;opacity:0.7;'>{f["num"]}</div>
                <div style='flex:1;'>
                    <div style='font-weight:700;color:#f1f5f9;font-size:1rem;margin-bottom:6px;'>{f["title"]}</div>
                    <div style='color:#94a3b8;font-size:0.88rem;line-height:1.6;margin-bottom:10px;'>{f["body"]}</div>
                    {tags_html}
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

    # Final performance spider/radar
    st.markdown("### Strategy Profile — Risk vs Return Radar")
    categories = ["Sharpe Ratio", "Total Return", "Win Rate", "Low Drawdown", "Stability", "Simplicity"]
    # Normalised 0-10 scores
    scores = {
        "Momentum":  [6, 5, 5, 7, 6, 9],
        "MR":        [1, 1, 4, 1, 2, 9],
        "Bandit":    [1, 1, 4, 1, 2, 3],
        "Buy & Hold":[8, 10, 6, 5, 7, 10],
    }
    fig = go.Figure()
    radar_colors = [COLOR["amber"], COLOR["red"], COLOR["purple"], COLOR["blue"]]
    for (name, vals), clr in zip(scores.items(), radar_colors):
        fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=categories + [categories[0]],
                                      name=name, line=dict(color=clr, width=2),
                                      fill="toself", fillcolor=clr + "20"))
    fig.update_layout(**DARK_TEMPLATE["layout"].to_plotly_json(),
                      title="Qualitative Strategy Profiles (0–10 scale)",
                      polar=dict(bgcolor="#0f172a",
                                 radialaxis=dict(gridcolor="#334155", range=[0, 10]),
                                 angularaxis=dict(gridcolor="#334155")),
                      height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class='card-accent' style='margin-top:0.5rem;'>
        <b style='color:#60a5fa; font-size:1rem;'>📝 Final Verdict</b><br>
        <p style='color:#cbd5e1; margin-top:0.7rem; line-height:1.7;'>
        Adaptive signal selection frameworks are theoretically sound but practically constrained by signal quality.
        For Bitcoin specifically, a simple momentum strategy remains the strongest performer. 
        Bandits add value only when signal diversity with genuine regime-switching behaviour exists in the pool.
        The study's most valuable contribution is identifying <b>what makes adaptive allocation fail</b> — 
        a finding directly applicable to practitioners building live systematic trading systems.
        </p>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  PAGE: METHODOLOGY DEEP-DIVE
# ═══════════════════════════════════════════════════════════
elif page == sections[8]:
    section_header("🔬 METHODOLOGY",
                   "Technical Deep-Dive",
                   "Signal construction, bandit algorithm details, and evaluation framework")

    tab1, tab2, tab3, tab4 = st.tabs(["📐 Signal Math", "🎰 Bandit Algorithm", "📏 Metrics", "🗂️ Data Pipeline"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### Signal 1: Momentum")
            st.markdown("""
<div class='formula'>ma_short = Close.rolling(10).mean()
ma_long  = Close.rolling(30).mean()

signal_t = +1 if ma_short > ma_long
         = -1 otherwise

strat_ret_t = signal_{t-1} × return_t

Sharpe = √252 × mean(strat_ret) / std(strat_ret)</div>

**Parameters:** 10-day / 30-day windows
**Hypothesis:** Trend continuation
**Asset fit:** ✅ Strong on BTC
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### Signal 2: Mean Reversion")
            st.markdown("""
<div class='formula'>μ_t = Close.rolling(20).mean()
σ_t = Close.rolling(20).std()

Z_t = (Close_t - μ_t) / σ_t

signal_t = -1 if Z_t > +1.0
         = +1 if Z_t < -1.0
         =  0 otherwise

Assume: prices revert to μ</div>

**Parameters:** 20-day window, ±1σ threshold
**Hypothesis:** Mean reversion
**Asset fit:** ❌ Fails on BTC
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("#### Signal 3: ML (RF)")
            st.markdown("""
<div class='formula'>Features = [lag1, lag2, lag3, vol10]

target_t = 1 if return_{t+1} > 0
         = 0 otherwise

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

train = first 70% (time-ordered)
test  = last  30%</div>

**Split:** 70% train / 30% test
**Warning:** ⚠️ Low OOS accuracy ~50%
            """, unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### UCB1 — Naive Version")
            st.markdown("""
<div class='formula'>arms = [momentum, mr, ml]
counts = {arm: 1 for arm in arms}
values = {arm: 0.0 for arm in arms}

for t in range(len(df)):
    ucb = {
        a: values[a] + sqrt(2*log(t+1)/counts[a])
        for a in arms
    }
    chosen = argmax(ucb)
    reward = signal[chosen,t] * return_t

    counts[chosen] += 1
    values[chosen] += (reward - values[chosen])
                        / counts[chosen]</div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("#### UCB — Improved Version")
            st.markdown("""
<div class='formula'>Improvement 1: Signal Filtering
  if |signal| < 0.1: signal = 0

Improvement 2: Asymmetric Reward
  reward = return if return > 0
         = 0      otherwise

Improvement 3: Reduced Exploration
  bonus = sqrt(1*log(t+1)/counts[arm])
          ↑ c=1 instead of c=2

Result: Still underperforms — signal
quality is the binding constraint</div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("#### Performance Evaluation Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["Sharpe Ratio", "Total Return", "Max Drawdown", "Win Rate", "Annualised Return"],
            "Formula": [
                "√252 × E[r] / σ(r)",
                "∏(1 + r_t) from t=0 to T",
                "max[(peak - trough) / peak]",
                "Count(r_t > 0) / T × 100",
                "[(1+Total Return)^(252/T)] - 1",
            ],
            "Interpretation": [
                "Risk-adjusted excess return (annualised); >1 = good, >2 = excellent",
                "Terminal wealth from $1 initial investment; 2.55x = +155%",
                "Worst peak-to-trough loss; lower (more negative) = worse",
                "% of days with positive strategy return; >55% = meaningful edge",
                "Equivalent constant annual return accounting for compounding",
            ],
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("#### Data Pipeline")
        st.markdown("""
```python
import yfinance as yf
import pandas as pd
import numpy as np

# Step 1: Fetch data
df = yf.download("BTC-USD", start="2019-01-01", interval="1d")

# Step 2: Flatten MultiIndex columns
df.columns = df.columns.get_level_values(0)
df.columns.name = None

# Step 3: Compute returns
df["returns"] = df["Close"].pct_change()
df = df.dropna()

# Step 4: Feature engineering
df["lag1"]       = df["returns"].shift(1)
df["lag2"]       = df["returns"].shift(2)
df["lag3"]       = df["returns"].shift(3)
df["volatility"] = df["returns"].rolling(10).std()

# Step 5: Create target (next-day direction)
df["target"] = (df["returns"].shift(-1) > 0).astype(int)
df = df.dropna()
```
        """)

    # Live data sample
    st.markdown("### Latest Data Sample (Most Recent 10 Rows)")
    display_cols = ["Close", "returns", "momentum_signal", "mr_signal", "z_score", "ml_proba"]
    available = [c for c in display_cols if c in df.columns]
    latest = df[available].tail(10).copy()
    latest["returns"] = (latest["returns"] * 100).round(3)
    if "ml_proba" in latest.columns:
        latest["ml_proba"] = latest["ml_proba"].round(4)
    if "z_score" in latest.columns:
        latest["z_score"] = latest["z_score"].round(3)
    st.dataframe(latest.style.format({"Close": "${:,.0f}", "returns": "{:.3f}%"}),
                 use_container_width=True)
