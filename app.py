import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FakeGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Lazy-load predictor (cached so models load once) ───────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor():
    from src.predictor import FakeNewsPredictor
    return FakeNewsPredictor()

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Syne:wght@700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

/* ── Base ── */
.main  { background: #0D1117; }
.block-container { padding-top: 2.8rem; padding-bottom: 2rem; max-width: 1200px; }

/* ── Header ── */
.fg-header {
    background: linear-gradient(120deg, #111827 0%, #161B22 60%, #0f1e2e 100%);
    border: 1px solid #30363D;
    border-radius: 20px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.8rem;
    margin-top: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5rem;
    flex-wrap: nowrap;
    min-height: 90px;
}
.fg-header-left {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    flex: 1;
    min-width: 0;
}
.fg-shield {
    width: 52px;
    height: 52px;
    min-width: 52px;
    background: linear-gradient(135deg, #1d4ed8 0%, #0ea5e9 100%);
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.6rem;
    box-shadow: 0 0 20px rgba(88,166,255,0.2);
}
.fg-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.8rem;
    font-weight: 400;
    color: #F0F6FF;
    letter-spacing: 3px;
    margin: 0;
    line-height: 1;
    white-space: nowrap;
    text-shadow: 0 0 40px rgba(88,166,255,0.15);
}
.fg-tagline {
    font-family: 'DM Sans', sans-serif;
    color: #58A6FF;
    font-size: 0.82rem;
    margin-top: 0.4rem;
    font-weight: 400;
    letter-spacing: 0.01em;
    white-space: nowrap;
}
.fg-header-badges {
    display: flex;
    gap: 0.7rem;
    flex-shrink: 0;
}
.fg-header-badge {
    background: rgba(88,166,255,0.06);
    border: 1px solid rgba(88,166,255,0.15);
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    text-align: center;
    white-space: nowrap;
}
.fg-badge-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 600;
    color: #93C5FD;
    line-height: 1;
    letter-spacing: -0.5px;
}
.fg-badge-lbl {
    font-size: 0.65rem;
    color: #484F58;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    font-weight: 500;
}
.fg-desc { display: none; }

/* ── KPI strip ── */
.kpi-row { display: flex; gap: 0.8rem; margin-bottom: 1.8rem; }
.kpi-card {
    flex: 1;
    background: #13181f;
    border: 1px solid #21272f;
    border-radius: 14px;
    padding: 1.3rem 1.2rem 1.1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(88,166,255,0.4), transparent);
}
.kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.75rem;
    font-weight: 600;
    color: #E6EDF3;
    letter-spacing: -0.5px;
    line-height: 1;
}
.kpi-lbl {
    color: #484F58;
    font-size: 0.72rem;
    margin-top: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 500;
}

/* ── Verdict cards ── */
.verdict-fake {
    background: linear-gradient(135deg, #1a0a0a, #2a0f0f);
    border: 2px solid #FF4B4B;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.verdict-real {
    background: linear-gradient(135deg, #0a1a12, #0f2a1a);
    border: 2px solid #00CC88;
    border-radius: 18px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
}
.verdict-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.verdict-label-fake {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    font-weight: 400;
    letter-spacing: 3px;
    color: #FF4B4B;
}
.verdict-label-real {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    font-weight: 400;
    letter-spacing: 3px;
    color: #00CC88;
}
.verdict-conf {
    color: #484F58;
    font-size: 0.8rem;
    margin-top: 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.03em;
}

/* ── Prob bars ── */
.prob-row { margin: 0.5rem 0; }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #8B949E;
    margin-bottom: 0.3rem;
}
.prob-bar-bg {
    background: #21262D;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
}
.prob-bar-fake { background: #FF4B4B; height: 10px; border-radius: 6px; }
.prob-bar-real { background: #00CC88; height: 10px; border-radius: 6px; }

/* ── Sidebar ── */
.sb-title {
    font-family: 'Syne', sans-serif;
    color: #E6EDF3;
    font-size: 1.2rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
}
.model-perf-card {
    background: #0D1117;
    border: 1px solid #21272f;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    font-size: 0.82rem;
    color: #484F58;
    line-height: 2.2;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.model-perf-card b { color: #8B949E; font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 600; }
.model-perf-card span { font-family: 'JetBrains Mono', monospace; color: #E6EDF3; font-size: 0.8rem; }

/* ── Textarea override ── */
textarea {
    background: #0D1117!important;
    color: #E6EDF3!important;
    border: 1px solid #30363D!important;
    border-radius: 12px!important;
    font-family: 'Inter', sans-serif!important;
    font-size: 0.95rem!important;

    /* Better mobile usability: let textarea be resizable and not rely on Streamlit fixed height */
    min-height: 220px !important;
    max-height: 55vh !important;
    resize: vertical !important;
}
textarea:focus { border-color: #58A6FF!important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
    background: #161B22;
    border-radius: 10px 10px 0 0;
    border: 1px solid #30363D;
    color: #8B949E;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: #1F2937 !important;
    color: #E6EDF3 !important;
    border-bottom-color: #1F2937 !important;
}

/* ── Info box ── */
.info-box {
    background: #161B22;
    border: 1px solid #30363D;
    border-left: 3px solid #58A6FF;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    color: #8B949E;
    font-size: 0.88rem;
    margin-bottom: 1rem;
}

/* ── Footer ── */
.fg-footer {
    text-align: center;
    color: #484F58;
    font-size: 0.82rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #21262D;
}

/* ── History item ── */
.hist-item {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    font-size: 0.88rem;
}
.hist-fake { border-left: 3px solid #FF4B4B; }
.hist-real { border-left: 3px solid #00CC88; }
.hist-meta { color: #484F58; font-size: 0.78rem; margin-top: 0.3rem; }

/* ─────────────────────────────────────────
   MOBILE RESPONSIVE (≤ 768px)
─────────────────────────────────────────*/
@media screen and (max-width: 768px) {

    /* Make textarea less tall on mobile */
    textarea {
        min-height: 170px !important;
        max-height: 45vh !important;
    }

    /* Reduce vertical footprint of charts */
    .js-plotly-plot {
        height: 320px !important;
        max-height: 320px !important;
    }

    /* Reduce some vertical spacing */
    .kpi-row { margin-bottom: 1.2rem !important; }
    .verdict-fake, .verdict-real { padding: 1.25rem 0.9rem !important; }
    .hist-item { margin-bottom: 0.55rem !important; }


    /* Main container */
    .block-container {
        padding-top: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* Header */
    .fg-header {
        flex-direction: column;
        align-items: flex-start;
        padding: 1.2rem;
        gap: 1rem;
        min-height: auto;
    }

    .fg-header-left {
        width: 100%;
    }

    .fg-title {
        font-size: 2rem;
        white-space: normal;
        letter-spacing: 1.5px;
    }

    .fg-tagline {
        white-space: normal;
        font-size: 0.75rem;
        line-height: 1.5;
    }

    /* Header badges */
    .fg-header-badges {
        width: 100%;
        flex-wrap: wrap;
        justify-content: space-between;
    }

    .fg-header-badge {
        flex: 1 1 48%;
        padding: 0.8rem;
    }

    .fg-badge-val {
        font-size: 1.1rem;
    }

    /* KPI Cards */
    .kpi-row {
        flex-wrap: wrap;
        gap: 0.8rem;
    }

    .kpi-card {
        flex: 1 1 calc(50% - 0.4rem);
        min-width: 140px;
        padding: 1rem;
    }

    .kpi-val {
        font-size: 1.4rem;
    }

    .kpi-lbl {
        font-size: 0.65rem;
    }

    /* Verdict cards */
    .verdict-fake,
    .verdict-real {
        padding: 1.5rem 1rem;
    }

    .verdict-icon {
        font-size: 2.2rem;
    }

    .verdict-label-fake,
    .verdict-label-real {
        font-size: 1.8rem;
        letter-spacing: 2px;
    }

    .verdict-conf {
        font-size: 0.75rem;
    }

    /* History cards */
    .hist-item {
        padding: 0.9rem;
        font-size: 0.8rem;
    }

    .hist-meta {
        font-size: 0.72rem;
        word-break: break-word;
    }

    /* Sidebar */
    .model-perf-card {
        font-size: 0.75rem;
        line-height: 1.8;
    }

    /* Text area */
    textarea {
        font-size: 0.9rem !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto;
        flex-wrap: nowrap;
    }

    .stTabs [data-baseweb="tab"] {
        min-width: max-content;
        padding: 0.5rem 0.8rem;
        font-size: 0.85rem;
    }

    /* DataFrames */
    .stDataFrame {
        overflow-x: auto;
    }

    /* Plotly charts */
    .js-plotly-plot {
        width: 100% !important;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        padding: 0.5rem;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        min-height: 44px;
        font-size: 0.95rem;
    }
}

/* Extra small devices */
@media screen and (max-width: 480px) {

    .fg-title {
        font-size: 1.7rem;
    }

    .fg-shield {
        width: 45px;
        height: 45px;
        font-size: 1.4rem;
    }

    .fg-header-badge {
        flex: 1 1 100%;
    }

    .kpi-card {
        flex: 1 1 100%;
    }

    .verdict-label-fake,
    .verdict-label-real {
        font-size: 1.5rem;
    }

    .kpi-val {
        font-size: 1.25rem;
    }
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "last_text" not in st.session_state:
    st.session_state.last_text = ""


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fg-header">
    <div class="fg-header-left">
        <div class="fg-shield">🛡️</div>
        <div>
            <div class="fg-title">FakeGuard</div>
            <div class="fg-tagline">AI-Powered Fake News Detection &nbsp;·&nbsp; NLP &amp; ML</div>
        </div>
    </div>
    <div class="fg-header-badges">
        <div class="fg-header-badge">
            <div class="fg-badge-val">96.3%</div>
            <div class="fg-badge-lbl">Top Accuracy</div>
        </div>
        <div class="fg-header-badge">
            <div class="fg-badge-val">3</div>
            <div class="fg-badge-lbl">ML Models</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-val">96.3%</div>
        <div class="kpi-lbl">Best Accuracy (LR)</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-val">95.9%</div>
        <div class="kpi-lbl">Random Forest</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-val">86.4%</div>
        <div class="kpi-lbl">Naive Bayes</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-val">0.993</div>
        <div class="kpi-lbl">Best ROC-AUC</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-title">⚙️ Configuration</div>', unsafe_allow_html=True)

    model_choice = st.radio(
        "Select Model",
        ["Logistic Regression", "Naive Bayes", "Random Forest"],
        index=0,
        help="Logistic Regression is the best model on WELFake (96.3% accuracy, 0.993 ROC-AUC)"
    )

    st.markdown("---")
    st.markdown("**📊 Model Performance**")
    st.markdown("""
    <div class="model-perf-card">
        <b>Logistic Regression</b><span style="float:right">96.31%</span><br>
        <b>Random Forest</b><span style="float:right">95.92%</span><br>
        <b>Naive Bayes</b><span style="float:right">86.38%</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if st.session_state.history:
        st.markdown(f"**🕓 Session History** ({len(st.session_state.history)} checks)")
        if st.button("🗑 Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    <div style="color:#484F58;font-size:0.8rem;line-height:1.6">
    <b style="color:#8B949E">About FakeGuard</b><br>
    NLP-based detection using TF-IDF bigrams and three classical ML classifiers.
    Trained on WELFake — 72,134 articles merged from 4 sources for robust generalization.
    </div>
    """, unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📰 Detector", "📈 Model Comparison", "🕓 History"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETECTOR
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📰 Analyze an Article")
    st.markdown("""
    <div class="info-box">
    Paste the full text of a news article. Longer articles (100+ words) give more reliable results.
    </div>
    """, unsafe_allow_html=True)

    article_text = st.text_area(
        "News Article",
        placeholder="Paste the full news article here...\n\nExample: Scientists have discovered a new treatment for...",
        # Mobile responsiveness: use CSS-controlled min/max height instead of fixed height
        height=None,
        label_visibility="collapsed"
    )

    word_count = len(article_text.split()) if article_text.strip() else 0
    char_count = len(article_text)

    col_wc, col_cc = st.columns(2)
    with col_wc:
        st.caption(f"Words: **{word_count}**")
    with col_cc:
        st.caption(f"Characters: **{char_count}**")

    # Put the button in its own row so mobile stacking is predictable
    analyze_clicked = st.button(
        "🔍 Analyze Article",
        use_container_width=True,
        disabled=(word_count < 5),
        type="primary"
    )

    if word_count > 0 and word_count < 5:
        st.warning("⚠️ Please enter at least 5 words for a reliable prediction.")

    # ── Run prediction ──
    if analyze_clicked and article_text.strip():
        with st.spinner("Loading models and analyzing..."):
            try:
                predictor = load_predictor()
                result = predictor.predict(article_text, model_choice)
                st.session_state.result = result
                st.session_state.last_text = article_text

                # Save to history
                preview = article_text[:120] + ("..." if len(article_text) > 120 else "")
                st.session_state.history.insert(0, {
                    "model": model_choice,
                    "label": result["label"],
                    "confidence": result["confidence"],
                    "fake_prob": result["probabilities"]["FAKE"],
                    "real_prob": result["probabilities"]["REAL"],
                    "preview": preview,
                })
                # Keep only last 20
                st.session_state.history = st.session_state.history[:20]

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.session_state.result = None

    # ── Display result ──
    if st.session_state.result and st.session_state.last_text == article_text:
        result = st.session_state.result
        label    = result["label"]
        conf     = result["confidence"]
        fake_p   = result["probabilities"]["FAKE"]
        real_p   = result["probabilities"]["REAL"]

        st.markdown("---")
        st.markdown("### 🎯 Verdict")

        res_col, detail_col = st.columns(2)

        with res_col:
            if label == "FAKE":
                st.markdown(f"""
                <div class="verdict-fake">
                    <div class="verdict-icon">🚨</div>
                    <div class="verdict-label-fake">FAKE NEWS</div>
                    <div class="verdict-conf">Confidence: {conf*100:.1f}% · Model: {model_choice}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-real">
                    <div class="verdict-icon">✅</div>
                    <div class="verdict-label-real">REAL NEWS</div>
                    <div class="verdict-conf">Confidence: {conf*100:.1f}% · Model: {model_choice}</div>
                </div>
                """, unsafe_allow_html=True)

        with detail_col:
            st.markdown("**Probability Breakdown**")

            fake_pct = fake_p * 100
            real_pct = real_p * 100

            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-label"><span>🚨 FAKE</span><span>{fake_pct:.1f}%</span></div>
                <div class="prob-bar-bg"><div class="prob-bar-fake" style="width:{fake_pct}%"></div></div>
            </div>
            <div class="prob-row" style="margin-top:0.8rem">
                <div class="prob-label"><span>✅ REAL</span><span>{real_pct:.1f}%</span></div>
                <div class="prob-bar-bg"><div class="prob-bar-real" style="width:{real_pct}%"></div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge chart
            gauge_color = "#FF4B4B" if label == "FAKE" else "#00CC88"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf * 100,
                number={"suffix": "%", "font": {"color": gauge_color, "family": "Space Grotesk", "size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#484F58", "tickfont": {"color": "#484F58", "size": 10}},
                    "bar": {"color": gauge_color, "thickness": 0.25},
                    "bgcolor": "#161B22",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 50],  "color": "#1a1f28"},
                        {"range": [50, 75], "color": "#1e2530"},
                        {"range": [75, 100],"color": "#222d3a"},
                    ],
                    "threshold": {
                        "line": {"color": gauge_color, "width": 3},
                        "thickness": 0.75,
                        "value": conf * 100
                    }
                },
                title={"text": "Confidence", "font": {"color": "#8B949E", "size": 13}},
                domain={"x": [0, 1], "y": [0, 1]}
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=200,
                margin=dict(t=40, b=10, l=20, r=20),
                font_color="#E6EDF3"
            )
            st.plotly_chart(fig_gauge, use_container_width=True, config={"displaylogo": False})

        # Interpretation note
        if conf >= 0.95:
            note = "🔒 Very high confidence — the model is strongly certain about this verdict."
        elif conf >= 0.80:
            note = "📊 High confidence — the model is fairly certain, but review the article manually."
        else:
            note = "⚠️ Moderate confidence — treat this result with caution and verify independently."

        st.markdown(f"""
        <div class="info-box" style="border-left-color: {'#FF4B4B' if label == 'FAKE' else '#00CC88'}">
        {note}
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📈 Model Comparison")

    metrics_df = pd.DataFrame({
        "Model":     ["Logistic Regression", "Random Forest", "Naive Bayes"],
        "Accuracy":  [96.31, 95.92, 86.38],
        "Precision": [95.96, 95.12, 84.53],
        "Recall":    [96.83, 96.96, 89.65],
        "F1 Score":  [96.40, 96.03, 87.01],
        "ROC-AUC":   [99.27, 99.30, 93.76],
    })

    st.dataframe(
        metrics_df.style.background_gradient(
            subset=["Accuracy","Precision","Recall","F1 Score","ROC-AUC"],
            cmap="Greens",
            vmin=80, vmax=100
        ).format("{:.2f}%", subset=["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Grouped bar chart
    metrics_long = metrics_df.melt(
        id_vars="Model",
        value_vars=["Accuracy","Precision","Recall","F1 Score","ROC-AUC"],
        var_name="Metric", value_name="Score"
    )

    fig_bar = px.bar(
        metrics_long, x="Model", y="Score", color="Metric",
        barmode="group",
        text="Score",
        color_discrete_sequence=["#58A6FF","#00CC88","#F2CC60","#FF4B4B"],
        template="plotly_dark"
    )
    fig_bar.update_traces(texttemplate="%{y:.2f}%", textposition="outside", textfont_size=9)
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#161B22",
        height=420,
        title="All Metrics by Model",
        title_font_color="#E6EDF3",
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#8B949E"),
        yaxis=dict(range=[80, 101], tickcolor="#484F58", gridcolor="#21262D"),
        xaxis=dict(tickcolor="#484F58"),
        font_color="#8B949E",
        margin=dict(t=50, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displaylogo": False})

    # Radar chart
    st.markdown("**Radar Comparison**")
    categories = ["Accuracy","Precision","Recall","F1 Score"]
    colors      = ["#58A6FF","#00CC88","#F2CC60"]

    fig_radar = go.Figure()
    for i, row in metrics_df.iterrows():
        vals = [row[c] for c in categories] + [row[categories[0]]]
        cats = categories + [categories[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            name=row["Model"],
            line_color=colors[i],
            fillcolor=colors[i].replace("FF","33").replace("88","33").replace("60","33")
        ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor="#161B22",
            radialaxis=dict(range=[80,100], tickcolor="#484F58", gridcolor="#30363D", color="#484F58"),
            angularaxis=dict(tickcolor="#484F58", gridcolor="#30363D", color="#8B949E")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#8B949E",
        height=420,
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig_radar, use_container_width=True, config={"displaylogo": False})


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🕓 Analysis History")

    if not st.session_state.history:
        st.markdown("""
        <div class="info-box">
        No articles analyzed yet. Go to the <b>Detector</b> tab and analyze an article — it'll show up here.
        </div>
        """, unsafe_allow_html=True)
    else:
        fake_count = sum(1 for h in st.session_state.history if h["label"] == "FAKE")
        real_count = len(st.session_state.history) - fake_count

        hc1, hc2, hc3 = st.columns(3)
        hc1.metric("Total Analyzed", len(st.session_state.history))
        hc2.metric("🚨 Fake", fake_count)
        hc3.metric("✅ Real", real_count)

        st.markdown("<br>", unsafe_allow_html=True)

        for i, item in enumerate(st.session_state.history):
            cls = "hist-fake" if item["label"] == "FAKE" else "hist-real"
            icon = "🚨" if item["label"] == "FAKE" else "✅"
            conf_str = f"{item['confidence']*100:.1f}%"
            st.markdown(f"""
            <div class="hist-item {cls}">
                <span style="color:{'#FF4B4B' if item['label']=='FAKE' else '#00CC88'};font-weight:600">
                    {icon} {item['label']}
                </span>
                &nbsp;·&nbsp;
                <span style="color:#8B949E">{item['model']}</span>
                &nbsp;·&nbsp;
                <span style="color:#8B949E">Confidence: <b style="color:#E6EDF3">{conf_str}</b></span>
                <div class="hist-meta" style="margin-top:0.5rem;color:#8B949E">{item['preview']}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fg-footer">
    Built with Streamlit · Scikit-Learn · Plotly &nbsp;|&nbsp; FakeGuard · Abhik Kundu
</div>
""", unsafe_allow_html=True)