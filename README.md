# 🧠 SentiScope — AI Market Intelligence Platform

A modern Streamlit dashboard that transforms the existing FastAPI sentiment analysis backend into a fully-featured, database-backed, AI-driven product intelligence platform.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🏠 **Dashboard** | Real-time sentiment metrics, pie/bar charts, word cloud, trend over time |
| 🔍 **Review Analysis** | Filter by sentiment/source, paginated table, confidence histogram |
| 🤖 **AI Suggestions** | Groq AI-powered improvement recommendations (+ rule-based fallback) |
| 📈 **Related Products** | Multi-product competitor sentiment comparison |
| ⚙️ **System Status** | API key status, DB health, cache stats, error log viewer |

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

**Required for full functionality:**
- `MONGO_URI` — MongoDB Atlas connection string
- `GROQ_API_KEY` — For AI-powered suggestions

**Optional (enables more review sources):**
- `SERPAPI_API_KEY` — Google News
- `YOUTUBE_API_KEY` — YouTube comments
- `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` — Reddit posts
- `FIRECRAWL_API_KEY` — Article scraping

> **Note:** The app works without any API keys using in-memory fallback and rule-based AI suggestions.

### 3. Run the dashboard

```bash
streamlit run app.py
```

---

## 🗄 MongoDB Schema

### `products` collection
```json
{
  "product_id": "uuid5 hash",
  "product_name": "string",
  "last_updated": "datetime",
  "sentiment_summary": {"positive": 0, "negative": 0, "neutral": 0},
  "total_reviews": 0
}
```

### `reviews` collection
```json
{
  "product_id": "string",
  "review_text": "string",
  "sentiment": "positive|negative|neutral",
  "confidence_score": 0.0-1.0,
  "analysis_mode": "transformers|vader_fallback|textblob_fallback",
  "source": "Google News|YouTube|Reddit|Firecrawl",
  "created_at": "datetime"
}
```

### `ai_reports` collection
```json
{
  "product_id": "string",
  "key_complaints": [],
  "root_causes": [],
  "feature_improvements": [],
  "ux_improvements": [],
  "pricing_suggestions": [],
  "marketing_suggestions": [],
  "executive_summary": "string",
  "mode": "ai|rule_based_fallback",
  "created_at": "datetime"
}
```

### `system_logs` collection
```json
{
  "error_type": "string",
  "source": "string",
  "error_message": "string",
  "resolved": false,
  "timestamp": "datetime"
}
```

---

## 🛡 Resilience & Fallback Architecture

```
Sentiment Analysis:
  Transformers (primary)
    → VADER (fallback)
    → TextBlob (fallback)
    → Neutral/0.5 (last resort)

AI Suggestions:
  Groq API (primary)
    → Rule-based keyword analysis (fallback)

Database:
  MongoDB Atlas (primary)
    → In-memory dict cache (fallback)

Data Retrieval:
  SerpAPI + YouTube + Reddit + Firecrawl (parallel)
    → Any working source used
    → Graceful degradation if sources fail

Caching:
  1-hour TTL response cache per source/query
  24-hour data freshness check (skip re-scraping)
```

---

## 🤖 AI Suggestions Logic

Suggestions are triggered when:
- **Negative sentiment > 30%** OR
- **Neutral sentiment > 40%**

The AI advisor produces:
- Key complaints identified from reviews
- Root cause analysis
- Feature improvement recommendations
- UX improvement suggestions
- Pricing strategy suggestions
- Marketing strategy recommendations
- Executive summary

---

## 📁 Project Structure

```
sentiment_dashboard/
├ app.py                  # Main Streamlit application
├ requirements.txt
├ .env.example
└ src/
    ├ __init__.py
    ├ database.py         # MongoDB Atlas manager + in-memory fallback
    ├ retriever.py        # Multi-API data fetching with retry & cache
    ├ analysis.py         # Sentiment analysis with fallback chain
    ├ ai_suggestions.py   # Groq AI + rule-based suggestions
    └ pipeline.py         # Orchestration: retrieve → analyze → store
```

---

## 🔑 Getting API Keys

| Service | Free Tier | Link |
|---|---|---|
| MongoDB Atlas | M0 (512MB) | https://mongodb.com/atlas |
| Groq | 14,400 req/day | https://console.groq.com |
| SerpAPI | 100 searches/month | https://serpapi.com |
| YouTube Data API | 10,000 units/day | https://console.cloud.google.com |
| Reddit API | Free | https://www.reddit.com/prefs/apps |
| Firecrawl | 500 pages/month | https://www.firecrawl.dev |
