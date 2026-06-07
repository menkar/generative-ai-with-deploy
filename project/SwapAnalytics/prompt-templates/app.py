from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SwapAnalytics — Movie Extractor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
    --bg:         #07090f;
    --bg2:        #0d1117;
    --surface:    #131929;
    --card:       #171f30;
    --card-hi:    #1d2640;
    --border:     rgba(255,255,255,0.06);
    --border-hi:  rgba(255,255,255,0.11);
    --text:       #f0f4ff;
    --text2:      #8b9ab5;
    --muted:      #4a5568;
    --accent:     #6366f1;
    --accent2:    #818cf8;
    --gold:       #f59e0b;
    --gold2:      #fbbf24;
    --green:      #10b981;
    --red:        #ef4444;
    --shadow:     0 4px 24px rgba(0,0,0,0.5);
    --shadow-acc: rgba(99,102,241,0.25);
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

/* ── Layout ── */
.block-container {
    max-width: 1100px !important;
    padding: 0 clamp(16px, 3vw, 48px) 80px !important;
    margin: 0 auto !important;
}

/* ════════════════════════════
   TOP NAV BAR
════════════════════════════ */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 0 14px;
    border-bottom: 1px solid var(--border-hi);
    margin-bottom: 40px;
}
.brand { display: flex; align-items: center; gap: 13px; }
.brand-icon {
    width: 42px; height: 42px; border-radius: 11px;
    background: linear-gradient(135deg, var(--gold), #e67e22);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
    box-shadow: 0 0 0 1px rgba(245,158,11,0.3), 0 6px 20px rgba(245,158,11,0.2);
}
.brand-name {
    font-size: 17px; font-weight: 800;
    color: var(--text); letter-spacing: -0.4px;
}
.brand-tagline { font-size: 11px; color: var(--muted); margin-top:2px; font-weight: 400; }
.nav-right { display: flex; align-items: center; gap: 10px; }
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.25);
    color: #34d399; font-size: 11px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px; letter-spacing: 0.3px;
}
.status-dot {
    width: 6px; height: 6px; background: var(--green);
    border-radius: 50%; box-shadow: 0 0 6px rgba(16,185,129,0.8);
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }
.model-pill {
    background: var(--card); border: 1px solid var(--border-hi);
    color: var(--text2); font-size: 11px; font-weight: 500;
    padding: 4px 12px; border-radius: 20px;
}

/* ════════════════════════════
   HERO
════════════════════════════ */
.hero {
    text-align: center;
    padding: clamp(16px, 4vh, 40px) 0 clamp(24px, 4vh, 44px);
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.22);
    color: var(--accent2); font-size: 11px; font-weight: 600;
    padding: 4px 14px; border-radius: 20px;
    letter-spacing: 0.5px; text-transform: uppercase;
    margin-bottom: 20px;
}
.hero h1 {
    font-size: clamp(28px, 5vw, 46px);
    font-weight: 800; letter-spacing: -1px;
    line-height: 1.12; margin-bottom: 16px; color: var(--text);
}
.hero h1 em {
    font-style: normal;
    background: linear-gradient(100deg, var(--gold), var(--gold2), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p {
    font-size: clamp(14px, 2vw, 16px);
    color: var(--text2); max-width: 480px;
    margin: 0 auto; line-height: 1.7;
}

/* ════════════════════════════
   INPUT SECTION
════════════════════════════ */
.section-label {
    font-size: 11px; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-bottom: 8px; display: flex; align-items: center; gap: 7px;
}
.section-label span { font-size: 14px; }

[data-testid="stTextArea"] label { display: none !important; }
[data-testid="stTextArea"] textarea {
    background: var(--card) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 14px !important;
    color: #f0f4ff !important;
    -webkit-text-fill-color: #f0f4ff !important;
    caret-color: var(--accent) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14.5px !important;
    line-height: 1.7 !important;
    padding: 16px 18px !important;
    resize: vertical !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15), var(--shadow) !important;
    outline: none !important;
}
[data-testid="stTextArea"] textarea::placeholder {
    color: #3a4560 !important;
    -webkit-text-fill-color: #3a4560 !important;
}

/* ════════════════════════════
   BUTTON
════════════════════════════ */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--accent) 0%, #7c3aed 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important; font-weight: 600 !important;
    padding: 14px 24px !important;
    letter-spacing: 0.2px !important;
    box-shadow: 0 6px 24px var(--shadow-acc) !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px var(--shadow-acc) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ════════════════════════════
   DIVIDER
════════════════════════════ */
.section-divider {
    display: flex; align-items: center; gap: 14px;
    margin: 36px 0 28px;
}
.section-divider::before, .section-divider::after {
    content:''; flex:1; height:1px; background: var(--border-hi);
}
.divider-label {
    font-size: 11px; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.8px;
    white-space: nowrap;
}

/* ════════════════════════════
   RESULT HEADER BANNER
════════════════════════════ */
.result-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.14) 0%, rgba(124,58,237,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 20px 24px;
    display: flex; align-items: center; gap: 16px;
    margin-bottom: 20px;
}
.result-banner-icon {
    width: 48px; height: 48px; border-radius: 13px;
    background: linear-gradient(135deg, var(--accent), #7c3aed);
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
    box-shadow: 0 6px 20px var(--shadow-acc);
}
.result-banner-title {
    font-size: 17px; font-weight: 700; color: var(--text);
    letter-spacing: -0.3px;
}
.result-banner-sub { font-size: 12px; color: var(--text2); margin-top: 3px; }
.result-banner-right { margin-left: auto; }
.success-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.25);
    color: #34d399; font-size: 11px; font-weight: 600;
    padding: 4px 12px; border-radius: 20px;
}

/* ════════════════════════════
   FIELD CARDS (rendered via st.columns)
════════════════════════════ */
.field-card {
    background: var(--card);
    border: 1px solid var(--border-hi);
    border-radius: 14px;
    padding: 18px 18px 16px;
    height: 100%;
    transition: border-color 0.2s, background 0.2s;
}
.field-card:hover {
    background: var(--card-hi);
    border-color: rgba(99,102,241,0.25);
}
.fc-icon {
    font-size: 20px; margin-bottom: 10px; display: block; line-height:1;
}
.fc-label {
    font-size: 10px; font-weight: 700; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.fc-value {
    font-size: 14px; font-weight: 600; color: var(--text);
    line-height: 1.5; word-break: break-word;
}
.fc-value.null { color: var(--muted); font-weight: 400; font-style: italic; }

/* ════════════════════════════
   SUMMARY BOX
════════════════════════════ */
.summary-box {
    background: var(--card);
    border: 1px solid var(--border-hi);
    border-left: 3px solid var(--accent);
    border-radius: 0 14px 14px 0;
    padding: 20px 22px;
    margin-top: 4px;
}
.summary-eyebrow {
    font-size: 10px; font-weight: 700; color: var(--accent2);
    text-transform: uppercase; letter-spacing: 0.8px;
    margin-bottom: 10px;
}
.summary-text {
    font-size: 15px; color: var(--text2);
    line-height: 1.8; font-weight: 400;
}

/* ════════════════════════════
   RAW OUTPUT EXPANDER
════════════════════════════ */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-size: 13px !important; font-weight: 600 !important;
    color: var(--text2) !important;
    padding: 12px 18px !important;
}
[data-testid="stExpander"] summary:hover { color: var(--text) !important; }

.raw-text {
    font-size: 13px; color: var(--text2);
    white-space: pre-wrap; font-family: 'Inter', monospace;
    line-height: 1.75; padding: 4px 2px;
}

/* ════════════════════════════
   SPINNER
════════════════════════════ */
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

/* ════════════════════════════
   RESPONSIVE
════════════════════════════ */
@media (max-width: 768px) {
    .result-banner { flex-wrap: wrap; }
    .result-banner-right { margin-left: 0; }
    .nav-right { display: none; }
    .hero h1 { font-size: 26px; }
}
</style>
""", unsafe_allow_html=True)

# ── Model (cached) ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return ChatMistralAI(model="mistral-small-2506")

model = get_model()

# ── Prompt (exactly as in core.py) ────────────────────────────────────────────
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional Movie Information Extraction Assistant.

Your task:
Extract useful structured information from a movie paragraph and present it in the exact format below.

Rules:
- Do NOT add explanations
- Do NOT add extra commentary
- Follow the exact format
- If information is missing → write NULL
- Keep summary short (2-3 lines max)
- Do NOT guess unknown facts

Output Format:

Movie Title      : 
Release Year     : 
Genre            : 
Director         : 
Main Cast        : 
Setting/Location : 
Plot             : 
Themes           : 
Ratings          : 
Notable Features : 
Short Summary    : """
    ),
    (
        "human",
        "Extract all useful movie information from the following paragraph:\n\n{paragraph}"
    )
])

# ── Field config ───────────────────────────────────────────────────────────────
FIELDS = [
    ("Movie Title",      "🎬", "Title"),
    ("Release Year",     "📅", "Released"),
    ("Genre",            "🎭", "Genre"),
    ("Director",         "🎥", "Director"),
    ("Main Cast",        "👥", "Main Cast"),
    ("Setting/Location", "🌍", "Setting"),
    ("Plot",             "📖", "Plot"),
    ("Themes",           "💡", "Themes"),
    ("Ratings",          "⭐", "Ratings"),
    ("Notable Features", "🏆", "Notable Features"),
]

def parse_response(raw: str) -> dict:
    result = {}
    for line in raw.strip().splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            matched = next((f[0] for f in FIELDS if f[0].lower() in key.lower()), None)
            if matched:
                result[matched] = val
            elif "summary" in key.lower():
                result["Short Summary"] = val
    return result

# ════════════════════════════════════════════════════════════
# TOP NAV
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
  <div class="brand">
    <div class="brand-icon">🎬</div>
    <div>
      <div class="brand-name">SwapAnalytics</div>
      <div class="brand-tagline">AI-Powered Movie Intelligence</div>
    </div>
  </div>
  <div class="nav-right">
    <span class="model-pill">Mistral Small 2506</span>
    <span class="status-pill"><span class="status-dot"></span> Live</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# HERO
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">✦ &nbsp;AI Information Extraction</div>
  <h1>Extract <em>Movie Intelligence</em><br>from Any Paragraph</h1>
  <p>Paste any movie description and instantly receive structured data —
     cast, genre, director, ratings, themes, and a concise summary.</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# INPUT SECTION
# ════════════════════════════════════════════════════════════
left, center, right = st.columns([1, 3, 1])
with center:
    st.markdown('<div class="section-label"><span>📋</span> Movie Paragraph</div>', unsafe_allow_html=True)
    para = st.text_area(
        label="paragraph",
        placeholder=(
            "Paste your movie paragraph here…\n\n"
            "Example: Interstellar is a visually stunning science fiction epic directed by "
            "Christopher Nolan. Released in 2014, the film stars Matthew McConaughey, "
            "Anne Hathaway, Jessica Chastain, and Michael Caine..."
        ),
        height=180,
        key="para_input",
    )
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    extract_clicked = st.button("🔍  Extract Movie Information", key="extract")

# ════════════════════════════════════════════════════════════
# EXTRACT & DISPLAY
# ════════════════════════════════════════════════════════════
if extract_clicked:
    if not para.strip():
        st.warning("⚠️  Please paste a movie paragraph before extracting.")
    else:
        with st.spinner("Analysing paragraph with AI…"):
            final_prompt = prompt.invoke({"paragraph": para})
            response     = model.invoke(final_prompt)
            raw          = response.content

        fields  = parse_response(raw)
        summary = fields.pop("Short Summary", "")

        # ── Result banner ──────────────────────────────────────────────────
        st.markdown("""
        <div class="section-divider"><span class="divider-label">Extraction Result</span></div>
        <div class="result-banner">
          <div class="result-banner-icon">📊</div>
          <div>
            <div class="result-banner-title">Extraction Complete</div>
            <div class="result-banner-sub">Structured data extracted successfully from your paragraph</div>
          </div>
          <div class="result-banner-right">
            <span class="success-badge">✓ &nbsp;10 Fields Analysed</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Field grid — 2 rows of 5 using st.columns ─────────────────────
        row1 = FIELDS[:5]
        row2 = FIELDS[5:]

        for row in [row1, row2]:
            cols = st.columns(len(row), gap="small")
            for col, (field_key, icon, label) in zip(cols, row):
                value   = fields.get(field_key, "NULL")
                is_null = value.strip().upper() in ("NULL", "", "NOT MENTIONED", "N/A", "—")
                display = "—" if is_null else value
                cls     = "fc-value null" if is_null else "fc-value"
                with col:
                    st.markdown(f"""
                    <div class="field-card">
                      <span class="fc-icon">{icon}</span>
                      <div class="fc-label">{label}</div>
                      <div class="{cls}">{display}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ── Short summary ──────────────────────────────────────────────────
        if summary and summary.strip().upper() not in ("NULL", ""):
            st.markdown(f"""
            <div class="summary-box">
              <div class="summary-eyebrow">📝 &nbsp;Short Summary</div>
              <div class="summary-text">{summary}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Raw output ─────────────────────────────────────────────────────
        with st.expander("🗒  View Raw Model Output"):
            st.markdown(f'<div class="raw-text">{raw}</div>', unsafe_allow_html=True)
