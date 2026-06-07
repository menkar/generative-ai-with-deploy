from dotenv import load_dotenv
import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Chat — Choose Your Mode",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Mode definitions (mirrors the original script exactly) ────────────────────
MODES = {
    "angry": {
        "label":       "Angry Mode",
        "emoji":       "😤",
        "tagline":     "Aggressive & Impatient",
        "description": "Brace yourself — this AI responds with frustration, bluntness, and zero patience.",
        "prompt":      "You are an angry AI agent. You respons aggreessively and impatiently.",
        "color":       "#ef4444",
        "color2":      "#dc2626",
        "glow":        "rgba(239,68,68,0.25)",
        "bg":          "rgba(239,68,68,0.07)",
        "border":      "rgba(239,68,68,0.25)",
        "badge_bg":    "rgba(239,68,68,0.15)",
        "badge_color": "#fca5a5",
    },
    "funny": {
        "label":       "Funny Mode",
        "emoji":       "😄",
        "tagline":     "Humorous & Witty",
        "description": "Expect jokes, puns, and a good laugh — this AI is here to entertain you.",
        "prompt":      "You are a funny AI agent. You respond with humor and jokes.",
        "color":       "#f59e0b",
        "color2":      "#d97706",
        "glow":        "rgba(245,158,11,0.25)",
        "bg":          "rgba(245,158,11,0.07)",
        "border":      "rgba(245,158,11,0.25)",
        "badge_bg":    "rgba(245,158,11,0.15)",
        "badge_color": "#fcd34d",
    },
    "sad": {
        "label":       "Sad Mode",
        "emoji":       "😢",
        "tagline":     "Depressed & Emotional",
        "description": "A melancholic AI that responds in a heavy, emotional, and deeply reflective tone.",
        "prompt":      "You are a sad agent. You resond in a depressed and emotional tone.",
        "color":       "#60a5fa",
        "color2":      "#3b82f6",
        "glow":        "rgba(96,165,250,0.25)",
        "bg":          "rgba(96,165,250,0.07)",
        "border":      "rgba(96,165,250,0.25)",
        "badge_bg":    "rgba(96,165,250,0.15)",
        "badge_color": "#93c5fd",
    },
}

# ── Model (cached) ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return ChatMistralAI(model="mistral-small-2506", temperature=0.9)

model = get_model()

# ── Session state ──────────────────────────────────────────────────────────────
if "mode_key"  not in st.session_state: st.session_state.mode_key  = None
if "messages"  not in st.session_state: st.session_state.messages  = []
if "display"   not in st.session_state: st.session_state.display   = []
if "thinking"  not in st.session_state: st.session_state.thinking  = False

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:         #080b12;
    --surface:    #111520;
    --card:       #161b2e;
    --card2:      #1a2035;
    --border:     rgba(255,255,255,0.07);
    --border-hi:  rgba(255,255,255,0.12);
    --text:       #f1f5f9;
    --text2:      #94a3b8;
    --muted:      #475569;
    --accent:     #6366f1;
    --accent2:    #8b5cf6;
    --shadow-acc: rgba(99,102,241,0.22);
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: var(--bg) !important;
    font-family: 'Inter', system-ui, sans-serif;
    color: var(--text);
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

.block-container {
    max-width: 860px !important;
    padding: 0 clamp(14px, 4vw, 32px) 110px !important;
    margin: 0 auto !important;
}

/* ════════════════════════════════
   SHARED HEADER
════════════════════════════════ */
.app-header {
    position: sticky; top: 0; z-index: 50;
    background: var(--bg);
    border-bottom: 1px solid var(--border);
    padding: 16px 0 14px;
    margin-bottom: 36px;
}
.header-row {
    display: flex; align-items: center;
    justify-content: space-between; gap: 12px;
}
.header-left { display: flex; align-items: center; gap: 14px; }
.header-av {
    width: 46px; height: 46px; border-radius: 13px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
    box-shadow: 0 0 0 1px var(--border-hi), 0 8px 24px var(--shadow-acc);
}
.header-title {
    font-size: 16px; font-weight: 700;
    color: var(--text); letter-spacing: -0.3px;
}
.header-sub {
    font-size: 11.5px; color: var(--muted);
    margin-top: 3px; font-weight: 400;
}
.mode-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    letter-spacing: 0.3px; text-transform: uppercase;
}

/* ════════════════════════════════
   MODE SELECTION SCREEN
════════════════════════════════ */
.select-hero {
    text-align: center;
    padding: clamp(24px, 6vh, 56px) 0 clamp(28px, 5vh, 48px);
}
.select-hero h1 {
    font-size: clamp(26px, 5vw, 38px);
    font-weight: 800; letter-spacing: -0.8px;
    color: var(--text); line-height: 1.15;
    margin-bottom: 14px;
}
.select-hero h1 span {
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.select-hero p {
    font-size: clamp(14px, 2vw, 16px);
    color: var(--text2); max-width: 420px;
    margin: 0 auto; line-height: 1.65;
}

.mode-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 18px;
    margin-top: 10px;
}
.mode-card {
    background: var(--card);
    border: 1px solid var(--border-hi);
    border-radius: 20px;
    padding: 28px 24px 24px;
    cursor: pointer;
    text-align: center;
    transition: transform 0.22s cubic-bezier(0.34,1.56,0.64,1),
                box-shadow 0.22s ease,
                border-color 0.22s ease;
    position: relative; overflow: hidden;
}
.mode-card::before {
    content: '';
    position: absolute; inset: 0;
    background: var(--card-glow, transparent);
    opacity: 0; transition: opacity 0.22s;
    border-radius: inherit;
}
.mode-card:hover { transform: translateY(-6px) scale(1.02); }
.mode-card:hover::before { opacity: 1; }

.card-emoji {
    font-size: 46px; margin-bottom: 16px;
    display: block; line-height: 1;
    filter: drop-shadow(0 4px 12px var(--card-glow, transparent));
}
.card-label {
    font-size: 17px; font-weight: 700;
    letter-spacing: -0.3px; margin-bottom: 6px;
}
.card-tagline {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.5px;
    padding: 3px 10px; border-radius: 20px;
    display: inline-block; margin-bottom: 14px;
}
.card-desc {
    font-size: 13px; color: var(--text2);
    line-height: 1.6;
}

/* ════════════════════════════════
   CHAT SCREEN
════════════════════════════════ */
.date-chip {
    text-align: center; margin: 4px 0 22px;
}
.date-chip span {
    background: var(--card); border: 1px solid var(--border);
    color: var(--muted); font-size: 11px; font-weight: 500;
    padding: 4px 14px; border-radius: 20px;
}

.msg-row {
    display: flex; align-items: flex-end; gap: 10px;
    margin-bottom: 22px;
    animation: slideIn 0.28s cubic-bezier(0.34,1.56,0.64,1);
}
.msg-row.user { flex-direction: row-reverse; }
@keyframes slideIn {
    from { opacity:0; transform: translateY(14px) scale(0.97); }
    to   { opacity:1; transform: translateY(0)    scale(1); }
}

.av {
    width: 36px; height: 36px; border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; flex-shrink: 0; font-weight: 700;
}
.av.bot { background: var(--av-bg, #6366f1); box-shadow: 0 4px 14px var(--av-glow, rgba(99,102,241,0.3)); }
.av.usr {
    background: linear-gradient(135deg, #0ea5e9, #06b6d4);
    box-shadow: 0 4px 14px rgba(14,165,233,0.3);
    color: #fff; font-size: 13px;
}

.bwrap { display: flex; flex-direction: column; max-width: min(74%, 520px); }
.msg-row.user .bwrap { align-items: flex-end; }
.msg-row.bot  .bwrap { align-items: flex-start; }

.bubble {
    padding: 13px 18px; border-radius: 20px;
    font-size: 14.5px; line-height: 1.65; word-break: break-word;
}
.bubble.user {
    background: linear-gradient(135deg, #0ea5e9, #06b6d4);
    color: #fff; border-bottom-right-radius: 5px;
    box-shadow: 0 6px 20px rgba(14,165,233,0.22);
}
.bubble.bot {
    background: var(--card2);
    color: var(--text);
    border: 1px solid var(--border-hi);
    border-bottom-left-radius: 5px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}
.msg-meta {
    font-size: 10.5px; color: var(--muted);
    margin-top: 5px; padding: 0 2px;
}

/* Typing dots */
.typing-row {
    display: flex; align-items: flex-end; gap: 10px;
    margin-bottom: 22px; animation: slideIn 0.28s ease;
}
.typing-bub {
    background: var(--card2); border: 1px solid var(--border-hi);
    border-radius: 20px; border-bottom-left-radius: 5px;
    padding: 15px 20px; display: flex; gap: 5px; align-items: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.2);
}
.dot { width:7px; height:7px; border-radius:50%; animation: typeBounce 1.4s ease-in-out infinite; }
.dot:nth-child(1) { animation-delay:0s; }
.dot:nth-child(2) { animation-delay:.18s; }
.dot:nth-child(3) { animation-delay:.36s; }
@keyframes typeBounce {
    0%,60%,100% { transform:translateY(0);   opacity:.35; }
    30%          { transform:translateY(-7px);opacity:1; }
}

/* ════════════════════════════════
   BUTTONS
════════════════════════════════ */
.stButton > button {
    background: var(--card) !important;
    border: 1px solid var(--border-hi) !important;
    color: var(--muted) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important; font-weight: 500 !important;
    padding: 6px 16px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    border-color: #f87171 !important;
    color: #f87171 !important;
    background: rgba(248,113,113,0.08) !important;
}

/* ════════════════════════════════
   CHAT INPUT
════════════════════════════════ */
[data-testid="stChatInput"] {
    position: fixed !important; bottom: 0 !important;
    left: 50% !important; transform: translateX(-50%) !important;
    width: min(860px, 100vw) !important;
    background: var(--bg) !important;
    padding: 16px clamp(14px, 4vw, 32px) 22px !important;
    border-top: 1px solid var(--border) !important;
    z-index: 100 !important;
}
[data-testid="stChatInput"] > div {
    background: var(--card) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    overflow: hidden !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4),
                0 0 0 3px rgba(99,102,241,0.15) !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--card) !important;
    border: none !important;
    color: #f1f5f9 !important;
    -webkit-text-fill-color: #f1f5f9 !important;
    caret-color: var(--accent) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14.5px !important;
    padding: 14px 18px !important;
    resize: none !important; line-height: 1.5 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border: none !important; border-radius: 10px !important;
    margin: 8px !important; width: 36px !important; height: 36px !important;
    box-shadow: 0 4px 12px var(--shadow-acc) !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
[data-testid="stChatInput"] button:hover {
    opacity: 0.85 !important; transform: scale(1.05) !important;
}
[data-testid="stChatInput"] button svg { fill:#fff !important; }

/* ════════════════════════════════
   RESPONSIVE
════════════════════════════════ */
@media (max-width: 600px) {
    .mode-grid { grid-template-columns: 1fr; gap: 14px; }
    .select-hero { padding: 20px 0 28px; }
    .bwrap { max-width: 85%; }
    .bubble { font-size: 14px; padding: 11px 15px; }
    .av { width:30px; height:30px; font-size:15px; border-radius:9px; }
}
@media (max-width: 400px) {
    .bwrap { max-width: 90%; }
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — MODE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.mode_key is None:

    # Header
    st.markdown("""
    <div class="app-header">
      <div class="header-row">
        <div class="header-left">
          <div class="header-av">🎭</div>
          <div>
            <div class="header-title">AI Chat Assistant</div>
            <div class="header-sub">Powered by Mistral Small 2506</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div class="select-hero">
      <h1>Choose Your <span>AI Personality</span></h1>
      <p>Select a mode to set the tone for your conversation.
         Your choice shapes how the AI will respond to everything you say.</p>
    </div>
    """, unsafe_allow_html=True)

    # Mode cards — rendered via columns so Streamlit buttons work
    cols = st.columns(3, gap="medium")
    for col, (key, m) in zip(cols, MODES.items()):
        with col:
            st.markdown(f"""
            <div class="mode-card"
                 style="--card-glow:{m['glow']};
                        border-color:{m['border']};
                        box-shadow: 0 8px 32px {m['glow']};">
              <span class="card-emoji">{m['emoji']}</span>
              <div class="card-label" style="color:{m['color']}">{m['label']}</div>
              <div class="card-tagline"
                   style="background:{m['badge_bg']};color:{m['badge_color']}">
                {m['tagline']}
              </div>
              <div class="card-desc">{m['description']}</div>
            </div>
            """, unsafe_allow_html=True)

            if st.button(f"Select  {m['emoji']}", key=f"pick_{key}", use_container_width=True):
                st.session_state.mode_key = key
                st.session_state.messages = [SystemMessage(content=m["prompt"])]
                st.session_state.display  = []
                st.session_state.thinking = False
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
else:
    m = MODES[st.session_state.mode_key]

    # Header (with active mode badge + change button)
    st.markdown(f"""
    <div class="app-header">
      <div class="header-row">
        <div class="header-left">
          <div class="header-av" style="background:linear-gradient(135deg,{m['color']},{m['color2']});
               box-shadow:0 8px 24px {m['glow']};">
            {m['emoji']}
          </div>
          <div>
            <div class="header-title">AI Chat Assistant</div>
            <div class="header-sub">
              <span class="mode-badge"
                    style="background:{m['badge_bg']};color:{m['badge_color']};">
                {m['emoji']} {m['label']}
              </span>
              &nbsp;Mistral Small 2506
            </div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Toolbar
    col1, col2, col3 = st.columns([5, 1.3, 1])
    with col2:
        if st.button("🎭 Change Mode", key="change"):
            st.session_state.mode_key = None
            st.session_state.messages = []
            st.session_state.display  = []
            st.session_state.thinking = False
            st.rerun()
    with col3:
        if st.session_state.display:
            if st.button("🗑 Clear", key="clear"):
                st.session_state.messages = [SystemMessage(content=m["prompt"])]
                st.session_state.display  = []
                st.session_state.thinking = False
                st.rerun()

    # Bot avatar gradient vars
    av_style = f"background:linear-gradient(135deg,{m['color']},{m['color2']});box-shadow:0 4px 14px {m['glow']};"

    # Empty state
    if not st.session_state.display:
        st.markdown(f"""
        <div style="text-align:center;padding:clamp(40px,10vh,80px) 20px;">
          <div style="font-size:64px;margin-bottom:18px;
                      filter:drop-shadow(0 0 24px {m['glow']});">{m['emoji']}</div>
          <div style="font-size:20px;font-weight:700;color:var(--text);
                      margin-bottom:10px;letter-spacing:-0.3px;">
            {m['label']} is Active
          </div>
          <div style="font-size:14px;color:var(--text2);max-width:300px;
                      margin:0 auto;line-height:1.6;">
            {m['description']}
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="date-chip"><span>Today</span></div>', unsafe_allow_html=True)

        for msg in st.session_state.display:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-row user">
                  <div class="bwrap">
                    <div class="bubble user">{msg['text']}</div>
                    <div class="msg-meta">You</div>
                  </div>
                  <div class="av usr">SW</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-row bot">
                  <div class="av bot" style="{av_style}">{m['emoji']}</div>
                  <div class="bwrap">
                    <div class="bubble bot">{msg['text']}</div>
                    <div class="msg-meta">AI · {m['label']}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # Typing indicator
    if st.session_state.thinking:
        dot_color = m['color']
        st.markdown(f"""
        <div class="typing-row">
          <div class="av bot" style="{av_style}">{m['emoji']}</div>
          <div class="typing-bub">
            <div class="dot" style="background:{dot_color}"></div>
            <div class="dot" style="background:{dot_color}"></div>
            <div class="dot" style="background:{dot_color}"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Input
    prompt = st.chat_input(f"Message {m['label']}…")

    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.session_state.display.append({"role": "user", "text": prompt})
        st.session_state.thinking = True
        st.rerun()

    # Generate response
    if st.session_state.thinking:
        response = model.invoke(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))
        st.session_state.display.append({"role": "bot", "text": response.content})
        st.session_state.thinking = False
        st.rerun()
