from dotenv import load_dotenv
import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg-base:        #080b12;
    --bg-surface:     #111520;
    --bg-card:        #161b2e;
    --bg-bubble-bot:  #1a2035;
    --border:         rgba(255,255,255,0.07);
    --border-bright:  rgba(255,255,255,0.12);
    --accent:         #6366f1;
    --accent-2:       #8b5cf6;
    --accent-user:    #0ea5e9;
    --accent-user-2:  #06b6d4;
    --text-primary:   #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted:     #475569;
    --green:          #22c55e;
    --shadow-accent:  rgba(99,102,241,0.22);
    --shadow-user:    rgba(14,165,233,0.22);
}

html, body { height: 100%; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] > .main > .block-container {
    background: var(--bg-base) !important;
    font-family: 'Inter', system-ui, sans-serif;
    color: var(--text-primary);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; visibility: hidden !important; }

/* ── Layout ── */
.block-container {
    max-width: 820px !important;
    padding: 0 clamp(12px, 4vw, 28px) 100px !important;
    margin: 0 auto !important;
}

/* ────────────────────────────────────────────
   HEADER
──────────────────────────────────────────── */
.app-header {
    position: sticky;
    top: 0;
    z-index: 50;
    background: var(--bg-base);
    border-bottom: 1px solid var(--border);
    padding: 16px 0 14px;
    margin-bottom: 32px;
    backdrop-filter: blur(12px);
}
.header-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}
.header-left { display: flex; align-items: center; gap: 14px; }

.header-avatar {
    position: relative;
    width: 48px;
    height: 48px;
    border-radius: 14px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    flex-shrink: 0;
    box-shadow: 0 0 0 1px var(--border-bright), 0 8px 24px var(--shadow-accent);
}
.header-avatar::after {
    content: '';
    position: absolute;
    bottom: -2px;
    right: -2px;
    width: 12px;
    height: 12px;
    background: var(--green);
    border: 2px solid var(--bg-base);
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(34,197,94,0.7);
}

.header-title {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.3px;
    line-height: 1.2;
}
.header-sub {
    font-size: 11.5px;
    color: var(--text-muted);
    margin-top: 3px;
    font-weight: 400;
    display: flex;
    align-items: center;
    gap: 6px;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.25);
    color: #a5b4fc;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}
.badge-dot {
    width: 5px; height: 5px;
    background: var(--green);
    border-radius: 50%;
    box-shadow: 0 0 5px rgba(34,197,94,0.8);
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

/* ────────────────────────────────────────────
   DIVIDER / DATE CHIP
──────────────────────────────────────────── */
.date-chip {
    text-align: center;
    margin: 8px 0 24px;
    position: relative;
}
.date-chip span {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 11px;
    font-weight: 500;
    padding: 4px 14px;
    border-radius: 20px;
    letter-spacing: 0.4px;
}

/* ────────────────────────────────────────────
   EMPTY STATE
──────────────────────────────────────────── */
.empty-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: clamp(40px, 10vh, 90px) 20px;
    text-align: center;
    gap: 0;
}
.empty-glow {
    width: 90px;
    height: 90px;
    border-radius: 28px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 38px;
    box-shadow: 0 0 0 1px var(--border-bright),
                0 12px 40px var(--shadow-accent),
                0 0 60px rgba(99,102,241,0.15);
    margin-bottom: 24px;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-8px); }
}
.empty-wrap h3 {
    font-size: 20px;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 10px;
    letter-spacing: -0.3px;
}
.empty-wrap p {
    font-size: 14px;
    color: var(--text-secondary);
    max-width: 320px;
    line-height: 1.6;
    margin-bottom: 28px;
}
.starter-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    max-width: 480px;
}
.chip {
    background: var(--bg-card);
    border: 1px solid var(--border-bright);
    color: var(--text-secondary);
    font-size: 12.5px;
    padding: 8px 16px;
    border-radius: 20px;
    cursor: default;
    transition: all 0.2s;
    font-family: 'Inter', sans-serif;
}
.chip:hover {
    border-color: var(--accent);
    color: #a5b4fc;
    background: rgba(99,102,241,0.08);
}

/* ────────────────────────────────────────────
   MESSAGE ROWS
──────────────────────────────────────────── */
.msg-group { margin-bottom: 24px; }

.msg-row {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    animation: slideIn 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.msg-row + .msg-row { margin-top: 6px; }
.msg-row.user { flex-direction: row-reverse; }
.msg-row.bot  { flex-direction: row; }

@keyframes slideIn {
    from { opacity: 0; transform: translateY(16px) scale(0.97); }
    to   { opacity: 1; transform: translateY(0)    scale(1); }
}

/* ── Avatars ── */
.av {
    width: 36px;
    height: 36px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.av.bot-av {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: 0 4px 14px var(--shadow-accent);
    color: #fff;
    font-size: 18px;
}
.av.user-av {
    background: linear-gradient(135deg, var(--accent-user), var(--accent-user-2));
    box-shadow: 0 4px 14px var(--shadow-user);
    color: #fff;
    font-size: 14px;
}

/* ── Bubble wrapper ── */
.bubble-wrap { display: flex; flex-direction: column; max-width: min(75%, 520px); }
.msg-row.user .bubble-wrap { align-items: flex-end; }
.msg-row.bot  .bubble-wrap { align-items: flex-start; }

/* ── Bubbles ── */
.bubble {
    padding: 13px 18px;
    border-radius: 20px;
    font-size: 14.5px;
    line-height: 1.65;
    word-break: break-word;
    position: relative;
}
.bubble.user {
    background: linear-gradient(135deg, var(--accent-user), var(--accent-user-2));
    color: #fff;
    border-bottom-right-radius: 5px;
    box-shadow: 0 6px 20px var(--shadow-user);
}
.bubble.bot {
    background: var(--bg-bubble-bot);
    color: var(--text-primary);
    border-bottom-left-radius: 5px;
    border: 1px solid var(--border-bright);
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
}

.msg-meta {
    font-size: 10.5px;
    color: var(--text-muted);
    margin-top: 5px;
    padding: 0 2px;
    font-weight: 400;
}

/* ────────────────────────────────────────────
   THINKING INDICATOR
──────────────────────────────────────────── */
.typing-row {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    margin-bottom: 24px;
    animation: slideIn 0.3s ease;
}
.typing-bubble {
    background: var(--bg-bubble-bot);
    border: 1px solid var(--border-bright);
    border-radius: 20px;
    border-bottom-left-radius: 5px;
    padding: 15px 20px;
    display: flex;
    gap: 5px;
    align-items: center;
    box-shadow: 0 6px 20px rgba(0,0,0,0.2);
}
.dot {
    width: 7px;
    height: 7px;
    background: var(--accent);
    border-radius: 50%;
    animation: typeBounce 1.4s ease-in-out infinite;
}
.dot:nth-child(1) { animation-delay: 0s; }
.dot:nth-child(2) { animation-delay: 0.18s; }
.dot:nth-child(3) { animation-delay: 0.36s; }
@keyframes typeBounce {
    0%, 60%, 100% { transform: translateY(0);   opacity: 0.35; }
    30%            { transform: translateY(-7px); opacity: 1; }
}

/* ────────────────────────────────────────────
   CLEAR BUTTON
──────────────────────────────────────────── */
.stButton > button {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    color: var(--text-muted) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 16px !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover {
    border-color: #f87171 !important;
    color: #f87171 !important;
    background: rgba(248,113,113,0.08) !important;
    box-shadow: 0 0 0 3px rgba(248,113,113,0.1) !important;
}

/* ────────────────────────────────────────────
   CHAT INPUT (fixed footer)
──────────────────────────────────────────── */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: min(820px, 100vw) !important;
    background: var(--bg-base) !important;
    padding: 16px clamp(12px, 4vw, 28px) 22px !important;
    z-index: 100 !important;
    border-top: 1px solid var(--border) !important;
}
[data-testid="stChatInput"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
    transition: box-shadow 0.2s, border-color 0.2s !important;
    overflow: hidden !important;
}
[data-testid="stChatInput"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4),
                0 0 0 3px rgba(99,102,241,0.15) !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--bg-card) !important;
    border: none !important;
    color: #f1f5f9 !important;
    caret-color: var(--accent) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14.5px !important;
    padding: 14px 18px !important;
    resize: none !important;
    line-height: 1.5 !important;
    -webkit-text-fill-color: #f1f5f9 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #475569 !important;
    -webkit-text-fill-color: #475569 !important;
}

/* Send button inside chat input */
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    border: none !important;
    border-radius: 10px !important;
    margin: 8px !important;
    width: 36px !important;
    height: 36px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 12px var(--shadow-accent) !important;
}
[data-testid="stChatInput"] button:hover {
    opacity: 0.85 !important;
    transform: scale(1.05) !important;
}
[data-testid="stChatInput"] button svg { fill: #fff !important; }

/* ────────────────────────────────────────────
   RESPONSIVE
──────────────────────────────────────────── */
@media (max-width: 600px) {
    .bubble-wrap { max-width: 85%; }
    .bubble { font-size: 14px; padding: 11px 15px; }
    .av { width: 30px; height: 30px; font-size: 14px; border-radius: 9px; }
    .header-title { font-size: 15px; }
    .empty-glow { width: 72px; height: 72px; font-size: 30px; border-radius: 22px; }
    .empty-wrap h3 { font-size: 17px; }
}
@media (max-width: 400px) {
    .bubble-wrap { max-width: 90%; }
    .starter-chips { display: none; }
}
</style>
""", unsafe_allow_html=True)

# ── Model init (cached) ────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return ChatMistralAI(model="mistral-small-2506", temperature=0.9)

model = get_model()

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a funny AI agent")
    ]
if "display" not in st.session_state:
    st.session_state.display = []   # list of {"role": "user"|"bot", "text": str}
if "thinking" not in st.session_state:
    st.session_state.thinking = False

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="header-inner">
        <div class="header-left">
            <div class="header-avatar">🤖</div>
            <div>
                <div class="header-title">AI Chat Assistant</div>
                <div class="header-sub">
                    <span class="badge"><span class="badge-dot"></span>Live</span>
                    Mistral Small 2506 &nbsp;·&nbsp; Funny Mode
                </div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Toolbar (clear button) ─────────────────────────────────────────────────────
if st.session_state.display:
    col1, col2 = st.columns([7, 1])
    with col2:
        if st.button("🗑 Clear", key="clear"):
            st.session_state.messages = [SystemMessage(content="You are a funny AI agent")]
            st.session_state.display = []
            st.session_state.thinking = False
            st.rerun()

# ── Chat history ───────────────────────────────────────────────────────────────
if not st.session_state.display:
    st.markdown("""
    <div class="empty-wrap">
        <div class="empty-glow">🤖</div>
        <h3>Start a Conversation</h3>
        <p>Ask me anything — I'm programmed to be helpful, witty, and a little bit funny.</p>
        <div class="starter-chips">
            <span class="chip">💡 What is generative AI?</span>
            <span class="chip">😄 Tell me a joke</span>
            <span class="chip">🚀 Explain LLMs simply</span>
            <span class="chip">🤔 What can you do?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown('<div class="date-chip"><span>Today</span></div>', unsafe_allow_html=True)
    for msg in st.session_state.display:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div class="bubble-wrap">
                    <div class="bubble user">{msg["text"]}</div>
                    <div class="msg-meta">You</div>
                </div>
                <div class="av user-av">SW</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-row bot">
                <div class="av bot-av">🤖</div>
                <div class="bubble-wrap">
                    <div class="bubble bot">{msg["text"]}</div>
                    <div class="msg-meta">AI Assistant</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Thinking indicator ─────────────────────────────────────────────────────────
if st.session_state.thinking:
    st.markdown("""
    <div class="typing-row">
        <div class="av bot-av">🤖</div>
        <div class="typing-bubble">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
prompt = st.chat_input("Type your message…")

if prompt:
    # Append to LangChain message history
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Show user message immediately
    st.session_state.display.append({"role": "user", "text": prompt})
    st.session_state.thinking = True
    st.rerun()

# ── Generate response after rerun ─────────────────────────────────────────────
if st.session_state.thinking:
    response = model.invoke(st.session_state.messages)
    st.session_state.messages.append(AIMessage(content=response.content))
    st.session_state.display.append({"role": "bot", "text": response.content})
    st.session_state.thinking = False
    st.rerun()
