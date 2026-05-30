"""
🧠 AI Market Intelligence Platform
A modern Streamlit dashboard for product sentiment analysis.
"""

import os
import sys
import asyncio
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

import nest_asyncio
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from dotenv import load_dotenv

# Load environment variables from .env if present.
load_dotenv()


sys.path.insert(0, os.path.dirname(__file__))

from src.database import MongoManager
from src.pipeline import run_pipeline
from src.ai_suggestions import generate_ai_suggestions, generate_competitor_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="SentiScope · AI Market Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Base theme */
:root {
    --bg-primary: #0a0e1a;
    --bg-card: #111827;
    --bg-card-hover: #1a2235;
    --accent-cyan: #00d4ff;
    --accent-green: #00f5a0;
    --accent-red: #ff4757;
    --accent-yellow: #ffd32a;
    --accent-purple: #a855f7;
    --text-primary: #e2e8f0;
    --text-muted: #64748b;
    --border: rgba(255,255,255,0.08);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1220 !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Headers */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #a855f7) !important;
    color: #0a0e1a !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-size: 14px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 20px rgba(0, 212, 255, 0.3) !important;
}

/* Input fields */
.stTextInput > div > div > input,
.stSelectbox > div > div > div,
.stMultiSelect > div > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 8px !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 20px !important;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Alert boxes */
.stAlert {
    border-radius: 8px !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Syne', sans-serif !important;
}

/* Progress bars */
.stProgress > div > div { border-radius: 4px !important; }

/* Custom cards */
.sentiment-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
}

.stat-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'Syne', sans-serif;
}

.badge-positive { background: rgba(0,245,160,0.15); color: #00f5a0; border: 1px solid rgba(0,245,160,0.3); }
.badge-negative { background: rgba(255,71,87,0.15); color: #ff4757; border: 1px solid rgba(255,71,87,0.3); }
.badge-neutral  { background: rgba(255,211,42,0.15); color: #ffd32a; border: 1px solid rgba(255,211,42,0.3); }

.page-header {
    border-left: 3px solid var(--accent-cyan);
    padding-left: 16px;
    margin-bottom: 24px;
}

.system-ok   { color: #00f5a0 !important; }
.system-warn { color: #ffd32a !important; }
.system-err  { color: #ff4757 !important; }
</style>
""", unsafe_allow_html=True)




# Apply nest_asyncio patch to allow nested event loops in Streamlit
nest_asyncio.apply()


def run_async(coro):
    """
    Run an async coroutine safely from Streamlit's sync context.
    Uses nest_asyncio to properly handle nested event loops.
    """
    return asyncio.run(coro)




def init_session_state():
    defaults = {
        "analysis_result": None,
        "current_product": "",
        "db_connected": None,
        "db_instance": None,
        "ai_report": None,
        "competitor_result": None,
        "page": "Dashboard",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v




@st.cache_resource
def get_db() -> MongoManager:
    db = MongoManager()
    return db


def ensure_db_connected() -> bool:
    db = get_db()
    if not db.is_connected:
        connected = run_async(db.connect())
        return connected
    return True


def normalize_sentiment_dist(sentiment_data: dict) -> dict:
    """
    Normalize sentiment data to a standard format with counts.
    Handles both old format {positive: X, negative: Y, neutral: Z}
    and new format {positive_count: X, positive_pct: Y, ...}
    """
    if not sentiment_data:
        return {"positive": 0, "negative": 0, "neutral": 0}
    
    # Check if it's the new format
    if "positive_count" in sentiment_data:
        return {
            "positive": sentiment_data.get("positive_count", 0),
            "negative": sentiment_data.get("negative_count", 0),
            "neutral": sentiment_data.get("neutral_count", 0),
        }
    # Otherwise assume it's the old format
    return {
        "positive": sentiment_data.get("positive", 0),
        "negative": sentiment_data.get("negative", 0),
        "neutral": sentiment_data.get("neutral", 0),
    }




SENTIMENT_COLORS = {
    "positive": "#00f5a0",
    "negative": "#ff4757",
    "neutral": "#ffd32a",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "#111827",
        "plot_bgcolor": "#111827",
        "font": {"color": "#e2e8f0", "family": "DM Sans"},
        "legend": {"bgcolor": "rgba(0,0,0,0)"},
        "colorway": ["#00d4ff", "#a855f7", "#00f5a0", "#ff4757", "#ffd32a"],
    }
}


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="#111827",
        plot_bgcolor="#111827",
        font=dict(color="#e2e8f0", family="DM Sans"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig



def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding:20px 0 10px'>
            <span style='font-family:Syne;font-size:22px;font-weight:800;
                         background:linear-gradient(135deg,#00d4ff,#a855f7);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
                🧠Real-Time Customer Sentiment Analysis dashboard
            </span>
            <div style='font-size:15px;color:#64748b;margin-top:4px'>
                AI Market Intelligence Platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        nav_options = ["🏠 Dashboard", "🔍 Review Analysis", "🤖 AI Suggestions",
                       "📈 Related Products", "📦 Data Management", "⚙️ System Status"]
        selected = st.radio("Navigate", nav_options, label_visibility="collapsed")
        st.session_state.page = selected

        st.divider()
        st.markdown("<div style='font-size:11px;color:#64748b'>📡 Connection Status</div>",
                    unsafe_allow_html=True)

        db_ok = ensure_db_connected()
        if db_ok:
            st.markdown("<span class='system-ok'>● MongoDB Connected</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='system-warn'>● MongoDB Offline (in-memory mode)</span>",
                        unsafe_allow_html=True)

        # API key status indicators
        st.markdown("<br><div style='font-size:11px;color:#64748b'>🔑 API Keys</div>",
                    unsafe_allow_html=True)
        keys = {
            "GROQ_API_KEY": "Groq AI",
            "SERPAPI_API_KEY": "SerpAPI",
            "YOUTUBE_API_KEY": "YouTube",
            "MONGO_URI": "MongoDB",
        }
        for env_key, label in keys.items():
            if os.getenv(env_key):
                st.markdown(f"<span class='system-ok' style='font-size:12px'>✓ {label}</span>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='system-warn' style='font-size:12px'>⚠ {label} (not set)</span>",
                            unsafe_allow_html=True)

    return selected



def render_search_section():
    st.markdown("""
    <div class='page-header'>
        <h2 style='margin:0;font-family:Syne;font-size:28px'>Product Intelligence Dashboard</h2>
        <p style='margin:4px 0 0;color:#64748b'>Analyze sentiment across the web in real-time</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        product = st.text_input(
            "Product or brand name",
            placeholder="e.g. iPhone 15, Tesla Model 3, ChatGPT...",
            key="search_input",
            label_visibility="collapsed",
        )
    with col2:
        mode = st.selectbox(
            "Mode",
            options=["hybrid", "transformers", "llm"],
            index=0,
            help="hybrid: fast sentiment + AI aspects | transformers: local only, fastest | llm: full AI on all items",
        )
    with col3:
        force_refresh = st.checkbox("Force Refresh", value=False)
    with col4:
        analyze_btn = st.button("🔍 Analyze", use_container_width=True)

    return product, mode, force_refresh, analyze_btn




def render_dashboard(result: dict):
    if not result:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px'>
            <div style='font-size:48px'>🔍</div>
            <h3 style='font-family:Syne;color:#64748b'>Search for a product to get started</h3>
            <p style='color:#475569'>Enter a product name above and click Analyze</p>
        </div>
        """, unsafe_allow_html=True)
        return

    dist = result["sentiment_dist"]
    total = result["total_reviews"]
    source_dist = result["source_dist"]

    if result.get("from_cache"):
        st.info("📦 Loaded from cache (data < 24 hours old). Toggle 'Force Refresh' to re-scrape.")

    mode_label = result.get("mode", "hybrid")
    mode_colors = {"hybrid": "#00d4ff", "transformers": "#00f5a0", "llm": "#a855f7"}
    st.markdown(
        f"<span style='background:{mode_colors.get(mode_label,'#64748b')}22;"
        f"color:{mode_colors.get(mode_label,'#64748b')};border:1px solid;"
        f"border-radius:20px;padding:3px 12px;font-size:12px;font-family:Syne;font-weight:600'>"
        f"Mode: {mode_label.upper()}</span>",
        unsafe_allow_html=True,
    )

    if result.get("errors"):
        for err in result["errors"]:
            if "No reviews found" in err:
                st.warning(f"⚠️ {err} — Try a different product name or check API keys.")
            else:
                st.error(f"❌ {err}")

    if total == 0:
        return


    dist = normalize_sentiment_dist(result["sentiment_dist"])


    pos_pct = dist["positive"] / total * 100 if total else 0
    neg_pct = dist["negative"] / total * 100 if total else 0
    neu_pct = dist["neutral"] / total * 100 if total else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📊 Total Reviews", total)
    m2.metric("✅ Positive", f"{dist['positive']} ({pos_pct:.0f}%)", delta=f"{pos_pct:.0f}%")
    m3.metric("❌ Negative", f"{dist['negative']} ({neg_pct:.0f}%)", delta=f"-{neg_pct:.0f}%", delta_color="inverse")
    m4.metric("➖ Neutral", f"{dist['neutral']} ({neu_pct:.0f}%)")

    st.markdown("<br>", unsafe_allow_html=True)


    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Sentiment Distribution**")
        fig_pie = px.pie(
            values=list(dist.values()),
            names=list(dist.keys()),
            color=list(dist.keys()),
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.5,
        )
        fig_pie.update_traces(
            textfont_size=13,
            marker=dict(line=dict(color="#0a0e1a", width=2)),
        )
        fig_pie = apply_dark_theme(fig_pie)
        fig_pie.update_layout(height=320, showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("**Source Distribution**")
        if source_dist:
            fig_bar = px.bar(
                x=list(source_dist.keys()),
                y=list(source_dist.values()),
                color=list(source_dist.keys()),
                color_discrete_sequence=["#00d4ff", "#a855f7", "#00f5a0", "#ff4757", "#ffd32a"],
                labels={"x": "Source", "y": "Reviews"},
            )
            fig_bar.update_traces(marker_line_width=0)
            fig_bar = apply_dark_theme(fig_bar)
            fig_bar.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No source data available.")

    #  Word Cloud 
    if result.get("word_frequencies"):
        st.markdown("**Top Keywords**")
        freq_dict = {item["text"]: item["value"] for item in result["word_frequencies"]}
        try:
            wc = WordCloud(
                width=900, height=300,
                background_color="#111827",
                colormap="cool",
                max_words=60,
                prefer_horizontal=0.8,
            ).generate_from_frequencies(freq_dict)
            fig_wc, ax = plt.subplots(figsize=(9, 3), facecolor="#111827")
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig_wc)
            plt.close(fig_wc)
        except Exception as e:
            st.caption(f"Word cloud unavailable: {e}")

    #  Trend chart (if date data available) 
    reviews = result.get("reviews", [])
    if reviews:
        df = pd.DataFrame(reviews)
        if "created_at" in df.columns:
            df["date"] = pd.to_datetime(df["created_at"]).dt.date
            trend = df.groupby(["date", "sentiment"]).size().reset_index(name="count")
            if len(trend) > 1:
                st.markdown("**Sentiment Trend Over Time**")
                fig_trend = px.line(
                    trend, x="date", y="count", color="sentiment",
                    color_discrete_map=SENTIMENT_COLORS,
                    markers=True,
                )
                fig_trend = apply_dark_theme(fig_trend)
                fig_trend.update_layout(height=280)
                st.plotly_chart(fig_trend, use_container_width=True)


#  Review Analysis Tab 

def render_review_analysis(result: dict):
    st.markdown("""
    <div class='page-header'>
        <h2 style='margin:0;font-family:Syne'>Review Analysis</h2>
        <p style='margin:4px 0 0;color:#64748b'>Filter, explore, and understand customer reviews</p>
    </div>
    """, unsafe_allow_html=True)

    reviews = result.get("reviews", [])
    if not reviews:
        st.info("No reviews loaded. Run an analysis first.")
        return

    df = pd.DataFrame(reviews)

    #  Filters 
    f1, f2 = st.columns(2)
    with f1:
        sentiments = ["All"] + sorted(df["sentiment"].dropna().unique().tolist())
        sel_sentiment = st.selectbox("Filter by Sentiment", sentiments)
    with f2:
        sources = ["All"] + sorted(df["source"].dropna().unique().tolist())
        sel_source = st.selectbox("Filter by Source", sources)

    filtered = df.copy()
    if sel_sentiment != "All":
        filtered = filtered[filtered["sentiment"] == sel_sentiment]
    if sel_source != "All":
        filtered = filtered[filtered["source"] == sel_source]

    st.caption(f"Showing {len(filtered)} of {len(df)} reviews")

    #  Paginated table ─
    PAGE_SIZE = 20
    total_pages = max(1, (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE)
    col_pg, _ = st.columns([1, 3])
    with col_pg:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start = (page - 1) * PAGE_SIZE
    page_df = filtered.iloc[start : start + PAGE_SIZE].copy()

    # Format for display
    display_cols = ["review_text", "sentiment", "confidence_score", "source"]
    display_cols = [c for c in display_cols if c in page_df.columns]
    display_df = page_df[display_cols].rename(columns={
        "review_text": "Review",
        "sentiment": "Sentiment",
        "confidence_score": "Confidence",
        "source": "Source",
    })

    # Color-code sentiment column
    def color_sentiment(val):
        colors = {"positive": "#00f5a0", "negative": "#ff4757", "neutral": "#ffd32a"}
        return f"color: {colors.get(val, '#e2e8f0')}"

    if "Sentiment" in display_df.columns:
        styled = display_df.style.map(color_sentiment, subset=["Sentiment"])
    else:
        styled = display_df.style

    st.dataframe(styled, use_container_width=True, height=420)

    #  Confidence distribution 
    if "confidence_score" in filtered.columns:
        st.markdown("**Confidence Score Distribution**")
        fig_hist = px.histogram(
            filtered, x="confidence_score", color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            nbins=20, barmode="overlay", opacity=0.75,
        )
        fig_hist = apply_dark_theme(fig_hist)
        fig_hist.update_layout(height=260)
        st.plotly_chart(fig_hist, use_container_width=True)

    #  Aspects Explorer (hybrid/llm mode only) 
    all_aspects = []
    for _, row in df.iterrows():
        for asp in (row.get("aspects") or []):
            if isinstance(asp, dict) and asp.get("aspect"):
                all_aspects.append({
                    "Aspect": asp["aspect"],
                    "Sentiment": asp.get("sentiment", "neutral"),
                    "Quote": asp.get("quote", "")[:120],
                    "Review Sentiment": row.get("sentiment", "neutral"),
                    "Source": row.get("source", ""),
                })

    if all_aspects:
        st.markdown("**Aspect-Based Insights** *(from hybrid/LLM analysis)*")
        asp_df = pd.DataFrame(all_aspects)
        asp_counts = asp_df.groupby(["Aspect", "Sentiment"]).size().reset_index(name="Count")
        fig_asp = px.bar(
            asp_counts.head(30), x="Aspect", y="Count", color="Sentiment",
            color_discrete_map={"positive": "#00f5a0", "negative": "#ff4757", "neutral": "#ffd32a"},
            barmode="stack",
        )
        fig_asp = apply_dark_theme(fig_asp)
        fig_asp.update_layout(height=300, xaxis_tickangle=-45)
        st.plotly_chart(fig_asp, use_container_width=True)
        st.dataframe(asp_df, use_container_width=True, height=250)
    else:
        if result.get("mode") == "transformers":
            st.caption("💡 Switch to **hybrid** or **llm** mode to extract product aspects.")


#  AI Suggestions Tab 

def render_ai_suggestions(result: dict):
    st.markdown("""
    <div class='page-header'>
        <h2 style='margin:0;font-family:Syne'>AI Product Advisor</h2>
        <p style='margin:4px 0 0;color:#64748b'>Intelligent improvement recommendations powered by AI</p>
    </div>
    """, unsafe_allow_html=True)

    if not result or not result.get("reviews"):
        st.info("No data loaded. Run an analysis first.")
        return

    dist = normalize_sentiment_dist(result["sentiment_dist"])
    total = sum(dist.values()) or 1
    neg_pct = dist["negative"] / total * 100
    neu_pct = dist["neutral"] / total * 100

    # Show trigger info
    col1, col2, col3 = st.columns(3)
    col1.metric("Negative %", f"{neg_pct:.1f}%", delta="Threshold: 30%",
                delta_color="inverse" if neg_pct > 30 else "normal")
    col2.metric("Neutral %", f"{neu_pct:.1f}%", delta="Threshold: 40%",
                delta_color="inverse" if neu_pct > 40 else "normal")
    trigger = neg_pct > 30 or neu_pct > 40
    col3.metric("AI Advisor", "🔴 TRIGGERED" if trigger else "🟢 Not Needed",
                delta="Analysis recommended" if trigger else "Sentiment looks healthy")

    if not trigger:
        st.info(
            f"ℹ️ Sentiment for **{result['product_name']}** looks healthy. "
            f"Negative: {neg_pct:.0f}%, Neutral: {neu_pct:.0f}%. "
            "You can still generate detailed recommendations below."
        )

    generate_btn = st.button("🤖 Generate AI Suggestions", use_container_width=False)

    if generate_btn or st.session_state.get("ai_report"):
        if generate_btn:
            with st.spinner("🤖 Analyzing reviews and generating recommendations..."):
                try:
                    report = run_async(generate_ai_suggestions(
                        result["product_name"],
                        result["reviews"],
                        dist,
                    ))
                    st.session_state.ai_report = report
                    # Save to DB
                    db = get_db()
                    run_async(db.save_ai_report(result.get("product_id", "unknown"), report))
                except Exception as e:
                    st.error(f"AI generation failed: {e}")
                    return

        report = st.session_state.ai_report
        if not report:
            return

        if not report.get("triggered"):
            if not result.get("reviews"):
                st.warning("⚠️ No reviews available for analysis. Run an analysis with valid data sources first.")
            else:
                st.info(report.get("reason", "No AI report triggered."))
            return

        mode = report.get("mode", "unknown")
        if mode == "rule_based_fallback":
            st.warning("⚠️ Generated using rule-based fallback (Groq API key not set or unavailable)")
        else:
            st.success("✅ Generated by Groq AI")

        if report.get("executive_summary"):
            st.markdown(f"""
            <div class='sentiment-card'>
                <div style='font-family:Syne;font-size:11px;letter-spacing:2px;color:#64748b;margin-bottom:8px'>
                    EXECUTIVE SUMMARY
                </div>
                <p style='margin:0;line-height:1.6'>{report['executive_summary']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        cols = st.columns(2)
        sections = [
            ("🚨 Key Complaints", "key_complaints", 0),
            ("🔍 Root Causes", "root_causes", 1),
            ("⚙️ Feature Improvements", "feature_improvements", 0),
            ("🎨 UX Improvements", "ux_improvements", 1),
            ("💰 Pricing Suggestions", "pricing_suggestions", 0),
            ("📣 Marketing Suggestions", "marketing_suggestions", 1),
        ]

        for title, key, col_idx in sections:
            items = report.get(key, [])
            if not items:
                continue
            with cols[col_idx]:
                st.markdown(f"**{title}**")
                for item in items:
                    st.markdown(f"""
                    <div style='background:#1a2235;border-left:3px solid #00d4ff;
                                padding:10px 14px;border-radius:0 8px 8px 0;
                                margin-bottom:8px;font-size:14px;line-height:1.5'>
                        {item}
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)


#  Related Products Tab 

def render_related_products(current_product: str):
    st.markdown("""
    <div class='page-header'>
        <h2 style='margin:0;font-family:Syne'>Competitor & Related Products</h2>
        <p style='margin:4px 0 0;color:#64748b'>Compare sentiment across competing products</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("Add competitor products to compare sentiment side-by-side:")
    competitors_input = st.text_input(
        "Competitor names (comma-separated)",
        placeholder="e.g. Samsung Galaxy S24, Google Pixel 8",
    )

    analyze_comps = st.button("📊 Compare Products", use_container_width=False)

    if analyze_comps and competitors_input:
        competitors = [c.strip() for c in competitors_input.split(",") if c.strip()]
        all_products = [current_product] + competitors

        progress = st.progress(0, text="Analyzing competitors...")
        all_reviews: Dict[str, List[dict]] = {}

        for i, product in enumerate(all_products):
            progress.progress((i + 1) / len(all_products), text=f"Analyzing {product}...")
            try:
                result = run_async(run_pipeline(product, force_refresh=False, db=get_db()))
                all_reviews[product] = result.get("reviews", [])
            except Exception as e:
                st.warning(f"Could not analyze {product}: {e}")
                all_reviews[product] = []

        progress.empty()

        with st.spinner("Generating competitive analysis..."):
            try:
                comp_result = run_async(generate_competitor_analysis(
                    current_product, competitors, all_reviews
                ))
                st.session_state.competitor_result = comp_result
                st.session_state.competitor_products = all_products
                st.session_state.competitor_reviews = all_reviews
            except Exception as e:
                st.error(f"Competitor analysis failed: {e}")

    if st.session_state.get("competitor_result"):
        comp_result = st.session_state.competitor_result
        all_products = st.session_state.get("competitor_products", [])
        all_reviews = st.session_state.get("competitor_reviews", {})
        summaries = comp_result.get("summaries", {})

        if summaries:
            # Comparison bar chart
            st.markdown("**Sentiment Comparison**")
            chart_data = []
            for product, s in summaries.items():
                for sentiment in ["positive", "negative", "neutral"]:
                    chart_data.append({
                        "Product": product,
                        "Sentiment": sentiment.capitalize(),
                        "Percentage": s.get(f"{sentiment}_pct", 0),
                    })

            fig_comp = px.bar(
                pd.DataFrame(chart_data),
                x="Product", y="Percentage", color="Sentiment",
                color_discrete_map={
                    "Positive": "#00f5a0",
                    "Negative": "#ff4757",
                    "Neutral": "#ffd32a",
                },
                barmode="group",
            )
            fig_comp = apply_dark_theme(fig_comp)
            fig_comp.update_layout(height=360)
            st.plotly_chart(fig_comp, use_container_width=True)

            # Summary table
            st.markdown("**Detailed Comparison**")
            table_data = []
            for product, s in summaries.items():
                table_data.append({
                    "Product": product,
                    "Positive %": f"{s.get('positive_pct', 0):.1f}%",
                    "Negative %": f"{s.get('negative_pct', 0):.1f}%",
                    "Neutral %": f"{s.get('neutral_pct', 0):.1f}%",
                    "Total Reviews": s.get("total", 0),
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # AI insights
        if comp_result.get("insights"):
            st.markdown("**AI Insights**")
            for insight in comp_result["insights"]:
                st.markdown(f"""
                <div style='background:#1a2235;border-left:3px solid #a855f7;
                            padding:10px 14px;border-radius:0 8px 8px 0;
                            margin-bottom:8px;font-size:14px'>
                    {insight}
                </div>
                """, unsafe_allow_html=True)

        if comp_result.get("mode") == "rule_based_fallback":
            st.caption("⚠️ Generated using rule-based fallback")


#  System Status Tab ─

def render_system_status():
    st.markdown("""
    <div class='page-header'>
        <h2 style='margin:0;font-family:Syne'>System Status</h2>
        <p style='margin:4px 0 0;color:#64748b'>Monitor APIs, database, and error logs</p>
    </div>
    """, unsafe_allow_html=True)

    #  API Status Cards 
    st.markdown("**API Configuration Status**")
    api_keys = {
        "GROQ_API_KEY": ("Groq AI", "AI suggestions & competitor analysis"),
        "SERPAPI_API_KEY": ("SerpAPI / Google News", "News review scraping"),
        "YOUTUBE_API_KEY": ("YouTube Data API", "Video comment scraping"),
        "REDDIT_CLIENT_ID": ("Reddit API", "Reddit post scraping"),
        "FIRECRAWL_API_KEY": ("Firecrawl", "Article content scraping"),
        "MONGO_URI": ("MongoDB Atlas", "Database storage"),
    }

    cols = st.columns(3)
    for i, (env_key, (label, desc)) in enumerate(api_keys.items()):
        with cols[i % 3]:
            is_set = bool(os.getenv(env_key))
            color = "#00f5a0" if is_set else "#ff4757"
            icon = "✅" if is_set else "❌"
            status = "Configured" if is_set else "Not Set"
            st.markdown(f"""
            <div class='sentiment-card' style='margin-bottom:12px'>
                <div style='font-size:11px;color:#64748b;font-family:Syne'>{label}</div>
                <div style='font-size:13px;color:{color};font-weight:600'>{icon} {status}</div>
                <div style='font-size:11px;color:#475569;margin-top:4px'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    #  Database Status ─
    st.markdown("**Database Status**")
    db = get_db()
    db_ok = db.is_connected
    db_status = "Connected ✅" if db_ok else "Offline – using in-memory cache ⚠️"
    db_color = "#00f5a0" if db_ok else "#ffd32a"
    st.markdown(f"""
    <div class='sentiment-card'>
        <span style='color:{db_color};font-weight:600'>{db_status}</span>
        <span style='color:#64748b;font-size:12px;margin-left:12px'>
            DB: {db._db_name if db_ok else "N/A"}
        </span>
    </div>
    """, unsafe_allow_html=True)

    if not db_ok and st.button("🔄 Retry Connection"):
        with st.spinner("Connecting..."):
            ok = run_async(db.connect())
        if ok:
            st.success("✅ Connected!")
            st.rerun()
        else:
            st.error("Connection failed. Check your MONGO_URI environment variable.")

    st.divider()

    #  Cache Status 
    from src.retriever import get_cache_status
    from src.analysis import get_analyzer_mode
    cache_stats = get_cache_status()
    analyzer_mode = get_analyzer_mode()

    st.markdown("**Runtime Status**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cache Entries", cache_stats["total_entries"])
    c2.metric("Valid Cache", cache_stats["valid_entries"])
    c3.metric("Cache TTL", f"{cache_stats['ttl_seconds']}s")
    c4.metric("Sentiment Engine", analyzer_mode.upper())

    st.divider()

    #  Data Retrieval Troubleshooting 
    st.markdown("**Data Retrieval Sources & Troubleshooting**")
    with st.expander("📊 Why am I not getting data from SerpAPI (Google News)?"):
        st.markdown(f"""
        **Current Status:** {"✅ Configured" if os.getenv('SERPAPI_API_KEY') else "❌ Not Configured"}
        
        **Common Issues & Solutions:**
        
        1. **API Key Not Set** (Most Common)
           - Add `SERPAPI_API_KEY=your_key` to your `.env` file
           - Get a free key at: https://serpapi.com/
           - Free tier includes 100 searches/month
        
        2. **No Credits / Quota Exceeded**
           - Check your SerpAPI account credits
           - Free tier limited to 100 searches/month
           - Upgrade plan if needed
        
        3. **Query Not Indexed**
           - Very obscure product names may not have news results
           - Try searching for more mainstream products
           - SerpAPI Google News engine may have limited coverage for niche items
        
        4. **API Rate Limiting**
           - Respect API rate limits (varies by plan)
           - Implement caching (we do this automatically - 1-hour TTL)
        
        **Fallback Sources** (Always Available):
        - YouTube API: Video comments (set `YOUTUBE_API_KEY`)
        - Reddit: Posts & comments (set Reddit credentials)
        - Firecrawl: Article content scraping (optional)
        
        **Data Quality Tips:**
        - Results are deduplicated automatically
        - Sentiment is analyzed on retrieved text
        - Mix of sources provides better coverage
        """)

    st.divider()

    #  Error Logs 
    st.markdown("**Recent Error Logs**")
    if db_ok:
        try:
            logs = run_async(db.get_recent_logs(50))
            if logs:
                log_df = pd.DataFrame(logs)
                display_cols = ["timestamp", "error_type", "source", "error_message", "resolved"]
                display_cols = [c for c in display_cols if c in log_df.columns]
                st.dataframe(log_df[display_cols], use_container_width=True, height=300)

                # Summary
                summary = run_async(db.get_log_summary())
                s1, s2 = st.columns(2)
                s1.metric("Total Errors", summary["total"])
                s2.metric("Unresolved", summary["unresolved"])
            else:
                st.success("✅ No errors logged.")
        except Exception as e:
            st.warning(f"Could not load logs: {e}")
    else:
        st.info("Connect to MongoDB to view persistent error logs.")
        from src.database import _memory_cache
        mem_logs = _memory_cache["system_logs"]
        if mem_logs:
            st.caption(f"In-memory logs: {len(mem_logs)} entries")
            st.dataframe(pd.DataFrame(mem_logs), use_container_width=True, height=200)
        else:
            st.success("✅ No errors in memory.")

    st.divider()
    st.markdown("**Database Maintenance**")
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        days_to_keep = st.number_input("Delete reviews older than (days)", min_value=1, max_value=365, value=30)
    with m_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Delete Old Records", type="secondary"):
            with st.spinner(f"Deleting records older than {days_to_keep} days..."):
                try:
                    count = run_async(db.delete_old_records(days=days_to_keep))
                    st.success(f"✅ Deleted {count} old review records.")
                except Exception as e:
                    st.error(f"Delete failed: {e}")

    st.divider()
    st.markdown("**Last Analysis**")
    if st.session_state.analysis_result:
        res = st.session_state.analysis_result
        st.markdown(f"""
        <div class='sentiment-card'>
            <b>{res.get('product_name', 'Unknown')}</b>
            &nbsp;·&nbsp;
            <span style='color:#64748b'>{res.get('total_reviews', 0)} reviews</span>
            &nbsp;·&nbsp;
            <span class='stat-badge {"badge-positive" if not res.get("from_cache") else "badge-neutral"}'>
                {"Fresh Data" if not res.get("from_cache") else "Cached"}
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No analysis run yet in this session.")


#  Data Management Tab 

def render_data_management():
    st.markdown("""
    <div class='page-header'>
        <h2 style='margin:0;font-family:Syne'>📦 Data Management</h2>
        <p style='margin:4px 0 0;color:#64748b'>View, search, and manage all retrieved product data</p>
    </div>
    """, unsafe_allow_html=True)

    db = get_db()
    
    # Get all products
    try:
        all_products = run_async(db.get_all_products())
    except Exception as e:
        st.error(f"Failed to load products: {e}")
        return

    if not all_products:
        st.info("No data retrieved yet. Start by analyzing a product in the Dashboard tab.")
        return

    st.markdown(f"**Total Retrieved Products: {len(all_products)}**")
    
    st.divider()

    # Search/filter
    search_term = st.text_input("🔍 Search products", placeholder="e.g. iPhone, Samsung...")
    
    # Filter products based on search
    filtered_products = [
        p for p in all_products
        if search_term.lower() in p.get("product_name", "").lower()
    ] if search_term else all_products

    if not filtered_products:
        st.warning(f"No products found matching '{search_term}'")
        return

    st.markdown(f"**Showing {len(filtered_products)} product(s)**")
    st.markdown("---")

    # Display products
    for idx, product in enumerate(filtered_products):
        product_name = product.get("product_name", "Unknown")
        product_id = product.get("product_id", "")
        last_updated = product.get("last_updated", "")
        total_reviews = product.get("total_reviews", 0)
        
        # Parse timestamp
        if isinstance(last_updated, str):
            try:
                updated_time = datetime.fromisoformat(last_updated).strftime("%Y-%m-%d %H:%M:%S")
            except:
                updated_time = str(last_updated)[:19]
        else:
            try:
                updated_time = last_updated.strftime("%Y-%m-%d %H:%M:%S") if hasattr(last_updated, 'strftime') else str(last_updated)
            except:
                updated_time = "Unknown"

        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div style='background:#111827;border:1px solid rgba(255,255,255,0.08);
                           border-radius:8px;padding:14px;'>
                    <div style='font-weight:600;font-size:15px;color:#e2e8f0'>{product_name}</div>
                    <div style='font-size:12px;color:#64748b;margin-top:6px'>
                        📊 {total_reviews} reviews · 🕐 {updated_time}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("👁️ View", key=f"view_{idx}_{product_id}", use_container_width=True):
                    st.session_state[f"show_details_{idx}"] = True
            
            with col3:
                if st.button("🗑️ Delete", key=f"del_{idx}_{product_id}", use_container_width=True):
                    st.session_state[f"confirm_delete_{idx}"] = True

            # Show details if requested
            if st.session_state.get(f"show_details_{idx}"):
                st.markdown("**Sentiment Summary:**")
                sentiment_summary = product.get("sentiment_summary", {})
                
                sent_cols = st.columns(3)
                with sent_cols[0]:
                    st.metric(
                        "🟢 Positive",
                        f"{sentiment_summary.get('positive_count', 0)}",
                        f"{sentiment_summary.get('positive_pct', 0):.1f}%"
                    )
                with sent_cols[1]:
                    st.metric(
                        "🟡 Neutral",
                        f"{sentiment_summary.get('neutral_count', 0)}",
                        f"{sentiment_summary.get('neutral_pct', 0):.1f}%"
                    )
                with sent_cols[2]:
                    st.metric(
                        "🔴 Negative",
                        f"{sentiment_summary.get('negative_count', 0)}",
                        f"{sentiment_summary.get('negative_pct', 0):.1f}%"
                    )
                
                # Show raw data
                if st.checkbox(f"Show raw product data", key=f"raw_{idx}"):
                    st.json(product)
                
                if st.button("Close Details", key=f"close_{idx}"):
                    st.session_state[f"show_details_{idx}"] = False
                    st.rerun()
            
            # Confirm delete
            if st.session_state.get(f"confirm_delete_{idx}"):
                st.warning(f"Are you sure you want to delete all data for '{product_name}'? This action cannot be undone.")
                del_col1, del_col2 = st.columns(2)
                
                with del_col1:
                    if st.button("✅ Yes, Delete", key=f"confirm_yes_{idx}_{product_id}", use_container_width=True):
                        with st.spinner(f"Deleting '{product_name}'..."):
                            try:
                                success = run_async(db.delete_product(product_id))
                                if success:
                                    st.success(f"✅ Deleted '{product_name}' and all associated reviews.")
                                    st.session_state[f"confirm_delete_{idx}"] = False
                                    # Refresh the page
                                    st.rerun()
                                else:
                                    st.error("Failed to delete product.")
                            except Exception as e:
                                st.error(f"Delete failed: {e}")
                
                with del_col2:
                    if st.button("❌ Cancel", key=f"confirm_no_{idx}", use_container_width=True):
                        st.session_state[f"confirm_delete_{idx}"] = False
                        st.rerun()
            
            st.markdown("---")

    st.divider()
    st.markdown("**Bulk Operations**")
    bulk_col1, bulk_col2 = st.columns(2)
    
    with bulk_col1:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    with bulk_col2:
        if st.button("📊 Export as CSV", use_container_width=True):
            try:
                df = pd.DataFrame(filtered_products)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"products_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")


#  Main App ─

def main():
    init_session_state()

    selected_page = render_sidebar()

    #  Global search bar (always visible) 
    with st.container():
        product, mode, force_refresh, analyze_btn = render_search_section()

    if analyze_btn and product.strip():
        with st.spinner(f"🔍 Analyzing **{product}** in `{mode}` mode..."):
            try:
                result = run_async(run_pipeline(
                    product.strip(),
                    force_refresh=force_refresh,
                    mode=mode,
                    db=get_db(),
                ))
                st.session_state.analysis_result = result
                st.session_state.current_product = product.strip()
                st.session_state.ai_report = None  # Reset AI report on new search
                if result.get("errors"):
                    for e in result["errors"]:
                        logger.warning(f"Pipeline error: {e}")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                logger.error(traceback.format_exc())
    elif analyze_btn and not product.strip():
        st.warning("Please enter a product name.")

    st.divider()

    #  Page routing 
    result = st.session_state.analysis_result
    current_product = st.session_state.current_product

    page_map = {
        "🏠 Dashboard": lambda: render_dashboard(result),
        "🔍 Review Analysis": lambda: render_review_analysis(result) if result else st.info("Run an analysis first."),
        "🤖 AI Suggestions": lambda: render_ai_suggestions(result) if result else st.info("Run an analysis first."),
        "📈 Related Products": lambda: render_related_products(current_product),
        "📦 Data Management": render_data_management,
        "⚙️ System Status": render_system_status,
    }

    render_fn = page_map.get(selected_page, lambda: render_dashboard(result))
    render_fn()

    #  Footer 
    st.markdown("""
    <div style='text-align:center;padding:32px 0 16px;color:#334155;font-size:12px'>
        SentiScope · AI Market Intelligence Platform ·
        Built with Streamlit, MongoDB Atlas, Groq AI & Transformers
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
