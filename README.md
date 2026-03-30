# 📈 Adaptive Signal Selection — Interactive Research Showcase

**Capstone Project | PGDM-RBA (2024–26) | Sourav Manna**  
*Guide: Prof. Swapnil Arun Desai | L.N. Welingkar Institute*

---

## 🌐 Live App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://adaptivesignalselectionstrategy-ggfp8oxyhdl9rjj8ktkw7e.streamlit.app)

---

## 📖 About

An interactive data application showcasing the capstone research:

> **"Adaptive Signal Selection under Market Regime Uncertainty using Risk-Aware Multi-Armed Bandit Framework"**

The app covers:
- 📊 BTC-USD data exploration (2019–2026, 2,644 trading days)
- 🚀 Momentum Strategy (MA Crossover) — Sharpe: +0.43
- 🔁 Mean Reversion Strategy (Z-Score) — Sharpe: -0.20
- 🤖 ML Signal (Random Forest) — overfitting analysis
- ⚔️ Head-to-head strategy comparison
- 🎰 Multi-Armed Bandit (UCB) adaptive selection
- 💡 Key research findings & conclusions

---

## 🚀 Deploy on Streamlit Cloud (GitHub)

### Step 1 — Create GitHub Repository
```bash
git init
git add .
git commit -m "Initial commit — Adaptive Signal Selection app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/adaptive-signal-selection.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub repository
4. Set **Main file path** to: `app.py`
5. Click **"Deploy"** — done! ✅

### Step 3 — Run Locally (Optional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 File Structure

```
adaptive-signal-selection/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Streamlit | Interactive web app framework |
| Plotly | Interactive charts & visualizations |
| yfinance | BTC-USD historical data (Yahoo Finance) |
| scikit-learn | Random Forest classifier |
| pandas / numpy | Data processing |

---

## 📧 Contact

**Sourav Manna** — PGDM-RBA (2024–26)  
S.P. Mandali's L.N. Welingkar Institute of Management Development & Research, Mumbai
