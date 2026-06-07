from dotenv import load_dotenv
load_dotenv()

import json
import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import List, Optional

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SwapAnalytics — Schema Extractor",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Pydantic schema (exactly as in core.py) ────────────────────────────────────
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

# ── Model (cached) ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return ChatMistralAI(model="mistral-small-2506")

model = get_model()

# ── Prompt (exactly as in core.py) ────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    ("system", """
Extract movie information from the paragraph
    {format_instructions}
"""),
    ("human", "{paragraph}")
])

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg:           #07090f;
    --bg2:          #0d1117;
    --surface:      #111825;
    --card:         #141d2e;
    --card-hi:      #1a2540;
    --border:       rgba(255,255,255,0.06);
    --border-hi:    rgba(255,255,255,0.11);
    --text:         #eef2ff;
    --text2:        #8b9ab5;
    --muted:        #4a5568;
    --accent:       #6366f1;
    --accent-light: #818cf8;
    --indigo:       #4f46e5;
    --violet:       #7c3aed;
    --gold:         #f59e0b;
    --gold2:        #fbbf24;
    --teal:         #14b8a6;
    --green:        #10b981;
    --rose:         #f43f5e;
    --shadow-lg:    0 8px 32px rgba(0,0,0,0.5);
    --shadow-acc:   rgba(99,102,241,0.25);
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: var(--bg) !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: var(--text) !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.block-container {
    max-width: 1140px !important;
    padding: 0 clamp(16px, 3vw, 52px) 80px !important;
    margin: 0 auto !important;
}

/* ══════════════ TOP NAV ══════════════ */
.topbar {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 16px 0 14px;
    border-bottom: 1px solid var(--border-hi);
    margin-bottom: 40px;
}
.brand { display: flex; align-items: center; gap: 13px; }
.brand-icon {
    width: 44px; height: 44px; border-radius: 12px;
    background: linear-gradient(135deg, var(--accent), var(--violet));
    display: flex; align-items: center; justify-content: center;
    font-size: 21px; flex-shrink: 0;
    box-shadow: 0 0 0 1px rgba(99,102,241,.3), 0 6px 20px rgba(99,102,241,.25);
}
.brand-name {
    font-size: 17px; font-weight: 800;
    color: var(--text); letter-spacing: -0.4px;
}
.brand-tagline { font-size: 11px; color: var(--muted); margin-top: 2px; }
.nav-pills { display: flex; align-items: center; gap: 8px; }
.pill {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px; letter-spacing: 0.3px;
}
.pill-schema {
    background: rgba(99,102,241,.12); border: 1px solid rgba(99,102,241,.25);
    color: var(--accent-light);
}
.pill-live {
    background: rgba(16,185,129,.1); border: 1px solid rgba(16,185,129,.25);
    color: #34d399;
}
.live-dot {
    width: 6px; height: 6px; background: var(--green);
    border-radius: 50%; box-shadow: 0 0 6px rgba(16,185,129,.8);
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ══════════════ HERO ══════════════ */
.hero {
    text-align: center;
    padding: clamp(16px,4vh,44px) 0 clamp(24px,5vh,48px);
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(99,102,241,.1); border: 1px solid rgba(99,102,241,.22);
    color: var(--accent-light); font-size: 11px; font-weight: 600;
    padding: 4px 14px; border-radius: 20px;
    letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 20px;
}
.hero h1 {
    font-size: clamp(28px,5vw,46px);
    font-weight: 800; letter-spacing: -1px;
    line-height: 1.12; margin-bottom: 16px;
}
.hero h1 em {
    font-style: normal;
    background: linear-gradient(100deg, var(--accent), var(--accent-light), var(--teal));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p {
    font-size: clamp(14px,2vw,16px); color: var(--text2);
    max-width: 500px; margin: 0 auto; line-height: 1.7;
}

/* ══════════════ INPUT ══════════════ */
.section-label {
    font-size: 11px; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-bottom: 8px; display: flex; align-items: center; gap: 7px;
}

[data-testid="stTextArea"] label { display: none !important; }
[data-testid="stTextArea"] textarea {
    background: var(--card) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 14px !important;
    color: #eef2ff !important;
    -webkit-text-fill-color: #eef2ff !important;
    caret-color: var(--accent) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14.5px !important; line-height: 1.7 !important;
    padding: 16px 18px !important; resize: vertical !important;
    box-shadow: var(--shadow-lg) !important;
    transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.15), var(--shadow-lg) !important;
    outline: none !important;
}
[data-testid="stTextArea"] textarea::placeholder {
    color: #2e3a50 !important;
    -webkit-text-fill-color: #2e3a50 !important;
}

/* ══════════════ BUTTON ══════════════ */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--accent) 0%, var(--violet) 100%) !important;
    border: none !important; border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important; font-weight: 600 !important;
    padding: 14px 24px !important; letter-spacing: .2px !important;
    box-shadow: 0 6px 24px var(--shadow-acc) !important;
    transition: opacity .2s, transform .15s !important;
}
.stButton > button:hover {
    opacity: .88 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 32px var(--shadow-acc) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ══════════════ DIVIDER ══════════════ */
.sec-div {
    display: flex; align-items: center; gap: 14px;
    margin: 36px 0 28px;
}
.sec-div::before, .sec-div::after {
    content:''; flex:1; height:1px; background:var(--border-hi);
}
.sec-div span {
    font-size: 11px; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: .8px; white-space: nowrap;
}

/* ══════════════ MOVIE HERO CARD ══════════════ */
.movie-hero {
    background: linear-gradient(135deg, rgba(99,102,241,.16), rgba(124,58,237,.1));
    border: 1px solid rgba(99,102,241,.22);
    border-radius: 20px;
    padding: clamp(22px,3vw,36px);
    margin-bottom: 20px;
    position: relative; overflow: hidden;
}
.movie-hero::before {
    content:'';
    position: absolute; top:-60px; right:-60px;
    width:200px; height:200px;
    background: radial-gradient(circle, rgba(99,102,241,.15), transparent 70%);
    pointer-events: none;
}
.movie-title {
    font-size: clamp(22px,4vw,34px);
    font-weight: 800; letter-spacing: -0.6px;
    color: var(--text); margin-bottom: 12px; line-height: 1.1;
}
.movie-meta {
    display: flex; flex-wrap: wrap; align-items: center;
    gap: 10px; margin-bottom: 18px;
}
.meta-chip {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 12px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px;
}
.chip-year  { background:rgba(245,158,11,.12); border:1px solid rgba(245,158,11,.25); color:#fcd34d; }
.chip-dir   { background:rgba(20,184,166,.12); border:1px solid rgba(20,184,166,.25); color:#5eead4; }
.rating-bar {
    display: flex; align-items: center; gap: 10px;
}
.stars { display:flex; gap:3px; }
.star  { font-size:16px; line-height:1; }
.rating-num {
    font-size: 24px; font-weight: 800; color: var(--gold);
    letter-spacing: -0.5px;
}
.rating-label { font-size: 12px; color: var(--muted); }

/* ══════════════ TAG CHIPS ══════════════ */
.tag-section { margin-bottom: 6px; }
.tag-label {
    font-size: 10px; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: .7px; margin-bottom: 8px;
}
.tags { display: flex; flex-wrap: wrap; gap: 7px; }
.tag {
    display: inline-block; font-size: 12.5px; font-weight: 500;
    padding: 5px 13px; border-radius: 20px;
}
.tag-genre { background:rgba(99,102,241,.14); border:1px solid rgba(99,102,241,.25); color:var(--accent-light); }
.tag-cast  { background:rgba(20,184,166,.1);  border:1px solid rgba(20,184,166,.22); color:#5eead4; }

/* ══════════════ SUMMARY CARD ══════════════ */
.summary-card {
    background: var(--card);
    border: 1px solid var(--border-hi);
    border-left: 3px solid var(--accent);
    border-radius: 0 14px 14px 0;
    padding: 20px 24px;
    margin-top: 4px;
}
.summary-eyebrow {
    font-size: 10px; font-weight: 700; color: var(--accent-light);
    text-transform: uppercase; letter-spacing: .8px; margin-bottom: 10px;
}
.summary-text {
    font-size: 15px; color: var(--text2);
    line-height: 1.8; font-weight: 400;
}

/* ══════════════ DATA TABLE ══════════════ */
.data-table {
    background: var(--card);
    border: 1px solid var(--border-hi);
    border-radius: 16px; overflow: hidden;
    margin-top: 4px;
}
.dt-header {
    background: rgba(99,102,241,.1);
    border-bottom: 1px solid var(--border-hi);
    padding: 14px 20px;
    display: flex; align-items: center; gap: 10px;
}
.dt-header-title { font-size:13px; font-weight:700; color:var(--text); }
.dt-row {
    display: flex; align-items: flex-start;
    padding: 14px 20px; gap: 0;
    border-bottom: 1px solid var(--border);
    transition: background .15s;
}
.dt-row:last-child { border-bottom: none; }
.dt-row:hover { background: var(--card-hi); }
.dt-key {
    width: 140px; flex-shrink: 0;
    font-size: 11px; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: .6px;
    padding-top: 2px;
}
.dt-val {
    font-size: 14px; font-weight: 500; color: var(--text);
    line-height: 1.5; flex: 1;
}
.dt-val.null-val { color: var(--muted); font-style: italic; font-weight: 400; }

/* ══════════════ RAW EXPANDER ══════════════ */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-size: 13px !important; font-weight: 600 !important;
    color: var(--text2) !important; padding: 12px 18px !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text) !important; }
.raw-text {
    font-size: 13px; color: var(--text2); white-space: pre-wrap;
    font-family: 'Courier New', monospace; line-height: 1.7;
}

/* ══════════════ JSON CODE BLOCK ══════════════ */
[data-testid="stCode"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1px solid var(--border-hi) !important;
}
[data-testid="stCode"] pre {
    background: var(--card) !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
    padding: 20px !important;
    font-family: 'Courier New', monospace !important;
}
[data-testid="stCode"] code { color: #a5b4fc !important; }
/* Copy button */
[data-testid="stCode"] button {
    background: rgba(99,102,241,.15) !important;
    border: 1px solid rgba(99,102,241,.25) !important;
    border-radius: 8px !important;
    color: var(--accent-light) !important;
    padding: 4px 10px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    transition: background .2s !important;
}
[data-testid="stCode"] button:hover {
    background: rgba(99,102,241,.28) !important;
}

/* ══════════════ SPINNER ══════════════ */
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

/* ══════════════ RESPONSIVE ══════════════ */
@media (max-width: 768px) {
    .nav-pills { display: none; }
    .movie-title { font-size: 22px; }
    .rating-num { font-size: 20px; }
    .dt-key { width: 110px; }
}
@media (max-width: 500px) {
    .movie-meta { gap: 7px; }
    .dt-row { flex-direction: column; gap: 4px; }
    .dt-key { width: 100%; }
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TOP NAV
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">🎭</div>
    <div>
      <div class="brand-name">SwapAnalytics</div>
      <div class="brand-tagline">Pydantic Schema Extraction · Structured AI Output</div>
    </div>
  </div>
  <div class="nav-pills">
    <span class="pill pill-schema">⬡ &nbsp;Pydantic Schema</span>
    <span class="pill pill-schema">Mistral Small 2506</span>
    <span class="pill pill-live"><span class="live-dot"></span> Live</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">⬡ &nbsp;Pydantic · Structured Output</div>
  <h1>Schema-Driven<br><em>Movie Extraction</em></h1>
  <p>Paste a movie paragraph and get a fully validated, schema-structured
     result — title, cast, genre, rating, and more extracted with type safety.</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# INPUT
# ════════════════════════════════════════════════════════════
_, center, _ = st.columns([1, 3, 1])
with center:
    st.markdown('<div class="section-label">📋 &nbsp;Movie Paragraph</div>', unsafe_allow_html=True)
    para = st.text_area(
        label="paragraph",
        placeholder=(
            "Paste a movie paragraph here…\n\n"
            "Example: Interstellar is a visually stunning science fiction epic directed by "
            "Christopher Nolan. Released in 2014, the film stars Matthew McConaughey, "
            "Anne Hathaway, Jessica Chastain, and Michael Caine. It holds a rating of "
            "8.6 on IMDb..."
        ),
        height=180,
        key="para_input",
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    extract_clicked = st.button("⬡  Extract with Schema", key="extract")

# ════════════════════════════════════════════════════════════
# EXTRACT & DISPLAY
# ════════════════════════════════════════════════════════════
if extract_clicked:
    if not para.strip():
        st.warning("⚠️  Please paste a movie paragraph before extracting.")
    else:
        with st.spinner("Extracting structured data with Pydantic schema…"):
            final_prompt = prompt.invoke({
                "paragraph": para,
                "format_instructions": parser.get_format_instructions()
            })
            response = model.invoke(final_prompt)
            raw      = response.content

        # Try parsing with Pydantic parser
        movie: Movie | None = None
        parse_error         = None
        try:
            movie = parser.parse(raw)
        except Exception:
            # Fallback: try extracting JSON block from raw text
            try:
                start = raw.index("{")
                end   = raw.rindex("}") + 1
                movie = Movie.model_validate(json.loads(raw[start:end]))
            except Exception as e:
                parse_error = str(e)

        st.markdown("""
        <div class="sec-div"><span>Extraction Result</span></div>
        """, unsafe_allow_html=True)

        if movie:
            # ── Star rating helper ────────────────────────────────────────
            def render_stars(rating: float | None) -> str:
                if rating is None:
                    return ""
                filled  = int(round(rating / 2))
                empty   = 5 - filled
                return "⭐" * filled + "☆" * empty

            # ── Movie hero card ───────────────────────────────────────────
            year_chip = (f'<span class="meta-chip chip-year">📅 &nbsp;{movie.release_year}</span>'
                         if movie.release_year else "")
            dir_chip  = (f'<span class="meta-chip chip-dir">🎥 &nbsp;{movie.director}</span>'
                         if movie.director else "")
            stars_html = ""
            if movie.rating is not None:
                stars_html = f"""
                <div class="rating-bar" style="margin-top:14px">
                  <div class="stars">{render_stars(movie.rating)}</div>
                  <div class="rating-num">{movie.rating}</div>
                  <div class="rating-label">/ 10 · IMDb</div>
                </div>"""

            st.markdown(f"""
            <div class="movie-hero">
              <div class="movie-title">{movie.title}</div>
              <div class="movie-meta">{year_chip}{dir_chip}</div>
              {stars_html}
            </div>
            """, unsafe_allow_html=True)

            # ── Genre + Cast tags ─────────────────────────────────────────
            left_col, right_col = st.columns(2, gap="large")

            with left_col:
                genre_tags = "".join(
                    f'<span class="tag tag-genre">{g}</span>' for g in movie.genre
                ) or '<span class="tag tag-genre" style="opacity:.5">—</span>'
                st.markdown(f"""
                <div class="tag-section">
                  <div class="tag-label">🎭 &nbsp;Genre</div>
                  <div class="tags">{genre_tags}</div>
                </div>
                """, unsafe_allow_html=True)

            with right_col:
                cast_tags = "".join(
                    f'<span class="tag tag-cast">{c}</span>' for c in movie.cast
                ) or '<span class="tag tag-cast" style="opacity:.5">—</span>'
                st.markdown(f"""
                <div class="tag-section">
                  <div class="tag-label">👥 &nbsp;Main Cast</div>
                  <div class="tags">{cast_tags}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

            # ── Summary ───────────────────────────────────────────────────
            st.markdown(f"""
            <div class="summary-card">
              <div class="summary-eyebrow">📝 &nbsp;Summary</div>
              <div class="summary-text">{movie.summary}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

            # ── Schema data table ─────────────────────────────────────────
            def row(icon, label, value, null=False):
                cls = "dt-val null-val" if null else "dt-val"
                return f"""
                <div class="dt-row">
                  <div class="dt-key">{icon} &nbsp;{label}</div>
                  <div class="{cls}">{value}</div>
                </div>"""

            st.markdown(f"""
            <div class="data-table">
              <div class="dt-header">
                <span style="font-size:16px">⬡</span>
                <span class="dt-header-title">Pydantic Schema · Validated Fields</span>
              </div>
              {row("🎬", "Title",       movie.title)}
              {row("📅", "Release Year",str(movie.release_year) if movie.release_year else "—", not movie.release_year)}
              {row("🎭", "Genre",       ", ".join(movie.genre) if movie.genre else "—", not movie.genre)}
              {row("🎥", "Director",    movie.director or "—", not movie.director)}
              {row("👥", "Cast",        ", ".join(movie.cast) if movie.cast else "—", not movie.cast)}
              {row("⭐", "Rating",      str(movie.rating) if movie.rating is not None else "—", movie.rating is None)}
            </div>
            """, unsafe_allow_html=True)

        else:
            # ── Parse error fallback ──────────────────────────────────────
            st.error("⚠️  Could not parse schema. Showing raw output below.")
            if parse_error:
                st.caption(f"Parser error: {parse_error}")

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        # ── JSON Output (copyable) ─────────────────────────────────────────
        if movie:
            json_str = json.dumps(movie.model_dump(), indent=2, ensure_ascii=False)
        else:
            # Best-effort: try to extract raw JSON block for copy
            try:
                start    = raw.index("{")
                end      = raw.rindex("}") + 1
                json_str = json.dumps(json.loads(raw[start:end]), indent=2, ensure_ascii=False)
            except Exception:
                json_str = raw

        st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
          <span style="font-size:11px;font-weight:700;color:#4a5568;
                       text-transform:uppercase;letter-spacing:.8px;">
            { } &nbsp;JSON Output — Click the copy icon →
          </span>
        </div>
        """, unsafe_allow_html=True)
        st.code(json_str, language="json")

        # ── Raw model output ───────────────────────────────────────────────
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        with st.expander("🗒  View Raw Model Output"):
            st.markdown(f'<div class="raw-text">{raw}</div>', unsafe_allow_html=True)
