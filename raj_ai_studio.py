"""
╔══════════════════════════════════════════════════════════════╗
║          RAJ PATIL AI STUDIO — Premium Chat Interface        ║
║     Data Scientist | ML Engineer | Generative AI Developer   ║
╚══════════════════════════════════════════════════════════════╝

Architecture:
  • raj_ai_studio.py      ← main entry point (this file)
  • requirements.txt      ← dependency manifest

Features:
  • Persistent conversation memory via st.session_state
  • Streaming responses (token-by-token)
  • Session analytics (message count, uptime, model info)
  • Download conversation as Markdown
  • Clear conversation
  • Copy-response button (JS clipboard injection)
  • Multi-model selector
  • Typing indicator
  • Premium dark glassmorphism UI
"""

import streamlit as st
from ollama import Client
import datetime
import time
import json

# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Raj Patil AI Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"
DEFAULT_MODEL = "deepseek-r1:1.5b"
AVAILABLE_MODELS = [
    "deepseek-r1:1.5b",
    "deepseek-r1:7b",
    "llama3.2:3b",
    "mistral:7b",
    "phi3:mini",
    "gemma2:2b",
]
SYSTEM_PROMPT = (
    "You are an expert AI assistant specializing in Data Science, "
    "Machine Learning, and Generative AI. Provide clear, accurate, "
    "and insightful answers. Format responses with markdown when helpful."
)

# ─────────────────────────────────────────────
#  OLLAMA CLIENT
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_ollama_client() -> Client:
    return Client(host=OLLAMA_HOST)


def check_ollama_connection(client: Client) -> bool:
    try:
        client.list()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────
#  SESSION STATE INITIALISATION
# ─────────────────────────────────────────────
def init_session_state() -> None:
    defaults = {
        "messages": [],          # list[dict]  {role, content, timestamp}
        "session_start": datetime.datetime.now(),
        "total_queries": 0,
        "selected_model": DEFAULT_MODEL,
        "ollama_online": None,   # None = unchecked
        "last_response_time": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─────────────────────────────────────────────
#  CONVERSATION HELPERS
# ─────────────────────────────────────────────
def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "timestamp": datetime.datetime.now().strftime("%H:%M"),
    })


def build_ollama_messages() -> list[dict]:
    """Convert session messages to Ollama API format, prepending system prompt."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in st.session_state.messages:
        msgs.append({"role": m["role"], "content": m["content"]})
    return msgs


def conversation_to_markdown() -> str:
    lines = [
        "# Raj Patil AI Studio — Conversation Export",
        f"**Session date:** {st.session_state.session_start.strftime('%Y-%m-%d %H:%M')}",
        f"**Model:** {st.session_state.selected_model}",
        f"**Total messages:** {len(st.session_state.messages)}",
        "---",
        "",
    ]
    for msg in st.session_state.messages:
        role_label = "🧑 You" if msg["role"] == "user" else "🤖 Assistant"
        lines.append(f"### {role_label}  `{msg['timestamp']}`")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  PREMIUM CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
<style>
/* ── Google Font Import ─────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

/* ── CSS Variables ──────────────────────────────────── */
:root {
    --bg-base:      #080c14;
    --bg-surface:   #0d1420;
    --bg-card:      rgba(255,255,255,0.04);
    --bg-card-hover:rgba(255,255,255,0.07);
    --border:       rgba(255,255,255,0.08);
    --border-bright:rgba(0,219,222,0.35);
    --accent-cyan:  #00dbde;
    --accent-teal:  #00b4d8;
    --accent-purple:#7b5ea7;
    --accent-grad:  linear-gradient(135deg,#00dbde,#00b4d8 60%,#7b5ea7);
    --text-primary: #eef2ff;
    --text-secondary:#8892a4;
    --text-muted:   #4a5568;
    --user-bubble:  rgba(0,180,216,0.12);
    --ai-bubble:    rgba(255,255,255,0.04);
    --success:      #10b981;
    --warning:      #f59e0b;
    --danger:       #ef4444;
    --radius-sm:    8px;
    --radius-md:    14px;
    --radius-lg:    20px;
    --shadow-glow:  0 0 30px rgba(0,219,222,0.15);
    --font-display: 'Syne', sans-serif;
    --font-body:    'DM Sans', sans-serif;
    --font-mono:    'DM Mono', monospace;
}

/* ── Base Reset ─────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

/* ── Remove Streamlit default padding ───────────────── */
.block-container {
    padding: 1.5rem 2rem !important;
    max-width: 1200px !important;
}

/* ── Scrollbar ──────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 99px; }

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem 1rem;
}

/* ── Hero Header ─────────────────────────────────────── */
.hero-container {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(0,219,222,0.1);
    border: 1px solid rgba(0,219,222,0.3);
    color: var(--accent-cyan);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 99px;
    margin-bottom: 1rem;
    font-family: var(--font-mono);
}
.hero-title {
    font-family: var(--font-display);
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    background: var(--accent-grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin: 0 0 0.5rem;
    letter-spacing: -1px;
}
.hero-sub {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 400;
    letter-spacing: 0.3px;
}
.hero-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-bright), transparent);
    margin: 1.5rem auto;
    max-width: 500px;
}

/* ── Metric Cards ────────────────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px 16px;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-grad);
    opacity: 0;
    transition: opacity 0.2s;
}
.metric-card:hover::before { opacity: 1; }
.metric-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-bright);
}
.metric-label {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 6px;
    font-family: var(--font-mono);
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    font-family: var(--font-display);
}
.metric-value.online { color: var(--success); }
.metric-value.accent { 
    background: var(--accent-grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Chat Area ───────────────────────────────────────── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 8px 0;
    max-height: 62vh;
    overflow-y: auto;
    scroll-behavior: smooth;
}

/* ── Message Bubbles ─────────────────────────────────── */
.message-row {
    display: flex;
    gap: 12px;
    animation: fadeSlideIn 0.3s ease forwards;
}
.message-row.user { flex-direction: row-reverse; }

@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    font-weight: 700;
}
.avatar.user {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-teal));
    color: var(--bg-base);
}
.avatar.ai {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-teal));
    color: white;
}

.bubble {
    max-width: 78%;
    padding: 14px 18px;
    border-radius: var(--radius-lg);
    line-height: 1.65;
    font-size: 0.93rem;
    position: relative;
}
.bubble.user {
    background: var(--user-bubble);
    border: 1px solid rgba(0,180,216,0.2);
    border-bottom-right-radius: 4px;
    color: var(--text-primary);
}
.bubble.ai {
    background: var(--ai-bubble);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
    color: var(--text-primary);
    box-shadow: var(--shadow-glow);
}

.msg-meta {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 5px;
    font-family: var(--font-mono);
    text-align: right;
}
.message-row.ai .msg-meta { text-align: left; }

/* ── Typing Indicator ────────────────────────────────── */
.typing-indicator {
    display: flex;
    gap: 5px;
    align-items: center;
    padding: 14px 18px;
    background: var(--ai-bubble);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    border-bottom-left-radius: 4px;
    width: fit-content;
}
.typing-dot {
    width: 7px;
    height: 7px;
    background: var(--accent-cyan);
    border-radius: 50%;
    animation: typingBounce 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typingBounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
    30% { transform: translateY(-6px); opacity: 1; }
}

/* ── Input Area ──────────────────────────────────────── */
.input-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 4px;
    transition: border-color 0.2s, box-shadow 0.2s;
    margin-top: 1rem;
}
.input-wrapper:focus-within {
    border-color: var(--border-bright);
    box-shadow: 0 0 0 3px rgba(0,219,222,0.08);
}

/* Streamlit textarea override */
.stTextArea textarea {
    background: transparent !important;
    border: none !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.95rem !important;
    resize: none !important;
    padding: 12px 16px !important;
    outline: none !important;
    box-shadow: none !important;
    border-radius: var(--radius-md) !important;
}
.stTextArea textarea::placeholder { color: var(--text-muted) !important; }
.stTextArea [data-baseweb="base-input"] { background: transparent !important; }

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    border-radius: var(--radius-md) !important;
    border: none !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

/* Primary send button */
div[data-testid="column"]:first-child .stButton > button {
    background: var(--accent-grad) !important;
    color: var(--bg-base) !important;
    padding: 10px 24px !important;
    font-size: 0.95rem !important;
}
div[data-testid="column"]:first-child .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0,219,222,0.3) !important;
    filter: brightness(1.1) !important;
}

/* Secondary / ghost buttons */
div[data-testid="column"]:not(:first-child) .stButton > button {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    font-size: 0.85rem !important;
    padding: 10px 16px !important;
}
div[data-testid="column"]:not(:first-child) .stButton > button:hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--border-bright) !important;
    color: var(--text-primary) !important;
}

/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background: var(--bg-card) !important;
    color: var(--text-secondary) !important;
    border: 1px solid var(--border) !important;
    width: 100% !important;
    text-align: left !important;
    padding: 10px 14px !important;
    font-size: 0.87rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--border-bright) !important;
    color: var(--accent-cyan) !important;
}

/* ── Selectbox ───────────────────────────────────────── */
.stSelectbox [data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}
.stSelectbox [data-baseweb="select"] > div:hover {
    border-color: var(--border-bright) !important;
}

/* ── Info / Warning / Success boxes ─────────────────── */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
    background: var(--bg-card) !important;
    font-family: var(--font-body) !important;
}

/* ── Sidebar skill tags ──────────────────────────────── */
.skill-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 8px 0;
}
.skill-tag {
    background: rgba(0,219,222,0.08);
    border: 1px solid rgba(0,219,222,0.2);
    color: var(--accent-cyan);
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    padding: 3px 9px;
    border-radius: 99px;
    font-family: var(--font-mono);
}

/* ── Sidebar profile block ───────────────────────────── */
.profile-block {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px;
    margin-bottom: 1rem;
    text-align: center;
}
.profile-name {
    font-family: var(--font-display);
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 2px;
}
.profile-role {
    font-size: 11px;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ── Status dot ──────────────────────────────────────── */
.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    margin-right: 5px;
    vertical-align: middle;
}
.status-dot.online  { background: var(--success); box-shadow: 0 0 6px var(--success); }
.status-dot.offline { background: var(--danger);  box-shadow: 0 0 6px var(--danger); }

/* ── Section headings ────────────────────────────────── */
.section-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1rem 0 0.5rem;
    font-family: var(--font-mono);
}

/* ── Footer ──────────────────────────────────────────── */
.footer {
    margin-top: 2.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}
.footer-left {
    font-size: 12px;
    color: var(--text-muted);
    font-family: var(--font-mono);
}
.footer-right {
    font-size: 11px;
    color: var(--text-muted);
    font-family: var(--font-mono);
}
.footer-accent {
    background: var(--accent-grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
}

/* ── Empty state ─────────────────────────────────────── */
.empty-state {
    text-align: center;
    padding: 3rem 1rem;
    color: var(--text-muted);
}
.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}
.empty-state-title {
    font-family: var(--font-display);
    font-size: 1.1rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}
.empty-state-hint {
    font-size: 0.85rem;
    color: var(--text-muted);
    line-height: 1.6;
}

/* ── Streamlit native overrides ──────────────────────── */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-display) !important;
    color: var(--text-primary) !important;
}
p, li, span {
    font-family: var(--font-body) !important;
}
code {
    font-family: var(--font-mono) !important;
    background: rgba(0,219,222,0.08) !important;
    color: var(--accent-cyan) !important;
    padding: 1px 5px !important;
    border-radius: 4px !important;
    font-size: 0.85em !important;
}
pre code {
    background: transparent !important;
    padding: 0 !important;
}
pre {
    background: rgba(0,0,0,0.4) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 14px !important;
}
hr { border-color: var(--border) !important; }

/* Hide Streamlit branding */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
"""

# ─────────────────────────────────────────────
#  CLIPBOARD JS HELPER
# ─────────────────────────────────────────────
def copy_to_clipboard_js(text: str) -> str:
    escaped = text.replace("`", "\\`").replace("\\", "\\\\").replace("\n", "\\n")
    return f"""
<button onclick="navigator.clipboard.writeText(`{escaped}`).then(()=>{{
    this.textContent='✓ Copied';
    setTimeout(()=>this.textContent='Copy',1500);
}})"
style="
    background:rgba(0,219,222,0.1);
    border:1px solid rgba(0,219,222,0.25);
    color:#00dbde;
    border-radius:6px;
    padding:4px 12px;
    font-size:11px;
    font-weight:600;
    cursor:pointer;
    letter-spacing:0.5px;
    transition:all 0.2s;
    font-family:'DM Mono',monospace;
">Copy</button>"""


# ─────────────────────────────────────────────
#  RENDER FUNCTIONS
# ─────────────────────────────────────────────
def render_message(msg: dict, index: int) -> None:
    role = msg["role"]
    content = msg["content"]
    ts = msg.get("timestamp", "")

    avatar = "R" if role == "user" else "✦"
    row_class = "user" if role == "user" else "ai"
    bubble_class = role

    st.markdown(
        f"""
        <div class="message-row {row_class}">
            <div class="avatar {row_class}">{avatar}</div>
            <div>
                <div class="bubble {bubble_class}">{content}</div>
                <div class="msg-meta">{ts}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Copy button only for AI responses
    if role == "assistant":
        st.markdown(copy_to_clipboard_js(content), unsafe_allow_html=True)


def render_sidebar(client: Client, ollama_online: bool) -> None:
    with st.sidebar:
        # ── Profile ──
        st.markdown(
            """
            <div class="profile-block">
                <div style="font-size:2.2rem;margin-bottom:8px;">✦</div>
                <div class="profile-name">Raj Patil</div>
                <div class="profile-role">
                    Data Scientist<br>
                    ML Engineer · Generative AI Developer
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Skills ──
        st.markdown('<p class="section-label">Stack</p>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="skill-tags">
                <span class="skill-tag">Python</span>
                <span class="skill-tag">PyTorch</span>
                <span class="skill-tag">LangChain</span>
                <span class="skill-tag">Ollama</span>
                <span class="skill-tag">Streamlit</span>
                <span class="skill-tag">HuggingFace</span>
                <span class="skill-tag">Scikit-learn</span>
                <span class="skill-tag">MLflow</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<p class="section-label">Engine</p>', unsafe_allow_html=True)

        # Model selector
        selected = st.selectbox(
            "Model",
            options=AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state.selected_model)
            if st.session_state.selected_model in AVAILABLE_MODELS
            else 0,
            label_visibility="collapsed",
        )
        if selected != st.session_state.selected_model:
            st.session_state.selected_model = selected

        # Connection status
        status_class = "online" if ollama_online else "offline"
        status_text  = "Ollama Online" if ollama_online else "Ollama Offline"
        st.markdown(
            f"""<p style="font-size:12px;color:var(--text-secondary);margin:8px 0;">
            <span class="status-dot {status_class}"></span>{status_text}
            </p>""",
            unsafe_allow_html=True,
        )

        st.markdown('<p class="section-label">Session</p>', unsafe_allow_html=True)

        # Analytics
        elapsed = datetime.datetime.now() - st.session_state.session_start
        mins = int(elapsed.total_seconds() // 60)
        secs = int(elapsed.total_seconds() % 60)

        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px;">
                <div class="metric-card">
                    <div class="metric-label">Messages</div>
                    <div class="metric-value accent">{len(st.session_state.messages)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Queries</div>
                    <div class="metric-value accent">{st.session_state.total_queries}</div>
                </div>
                <div class="metric-card" style="grid-column:span 2;">
                    <div class="metric-label">Uptime</div>
                    <div class="metric-value" style="font-size:1rem;">{mins}m {secs:02d}s</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Actions
        st.markdown('<p class="section-label">Actions</p>', unsafe_allow_html=True)

        if st.button("⬇  Download Conversation"):
            if st.session_state.messages:
                md = conversation_to_markdown()
                st.download_button(
                    label="Save .md",
                    data=md,
                    file_name=f"raj_ai_studio_{datetime.date.today()}.md",
                    mime="text/markdown",
                )
            else:
                st.toast("No conversation to download.", icon="ℹ️")

        if st.button("🗑  Clear Conversation"):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.rerun()

        # Footer
        st.markdown(
            """
            <div style="margin-top:2rem;padding-top:1rem;
                        border-top:1px solid var(--border);
                        font-size:10px;color:var(--text-muted);
                        font-family:'DM Mono',monospace;line-height:1.8;">
                RAJ PATIL AI STUDIO<br>
                v2.0 · Streamlit + Ollama<br>
                © 2026
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-badge">✦ AI Studio</div>
            <h1 class="hero-title">Raj Patil AI Studio</h1>
            <p class="hero-sub">
                Data Scientist · Machine Learning Engineer · Generative AI Developer
            </p>
            <div class="hero-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(ollama_online: bool) -> None:
    model_short = st.session_state.selected_model.split(":")[0].upper()
    version_tag = st.session_state.selected_model.split(":")[-1] if ":" in st.session_state.selected_model else "—"
    status_txt  = "● Online" if ollama_online else "● Offline"
    status_cls  = "online" if ollama_online else ""

    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Model</div>
                <div class="metric-value accent">{model_short}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Variant</div>
                <div class="metric-value">{version_tag}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Engine</div>
                <div class="metric-value">Ollama</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Status</div>
                <div class="metric-value {status_cls}">{status_txt}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">✦</div>
                <div class="empty-state-title">Start a Conversation</div>
                <div class="empty-state-hint">
                    Ask about Data Science, Machine Learning, Python,<br>
                    statistical modelling, or any AI topic.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for i, msg in enumerate(st.session_state.messages):
        render_message(msg, i)


def render_input_area(ollama_online: bool) -> None:
    st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
    prompt = st.text_area(
        "Message",
        placeholder="Ask me anything about Data Science, ML, or AI…",
        height=90,
        label_visibility="collapsed",
        key="prompt_input",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col_send, col_clear, col_spacer = st.columns([2, 1, 3])

    with col_send:
        send_clicked = st.button(
            "⚡ Send",
            disabled=not ollama_online,
            use_container_width=True,
        )
    with col_clear:
        if st.button("✕ Clear", use_container_width=True):
            st.session_state.messages = []
            st.session_state.total_queries = 0
            st.rerun()

    if send_clicked:
        handle_send(prompt)


# ─────────────────────────────────────────────
#  CORE SEND / STREAM LOGIC
# ─────────────────────────────────────────────
def handle_send(prompt: str) -> None:
    prompt = (prompt or "").strip()
    if not prompt:
        st.warning("Please enter a message before sending.")
        return

    add_message("user", prompt)
    st.session_state.total_queries += 1

    # Show typing indicator then stream response
    typing_placeholder = st.empty()
    typing_placeholder.markdown(
        """
        <div style="display:flex;gap:12px;align-items:flex-start;">
            <div class="avatar ai">✦</div>
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    client = get_ollama_client()
    t_start = time.time()
    full_response = ""

    try:
        stream = client.chat(
            model=st.session_state.selected_model,
            messages=build_ollama_messages(),
            stream=True,
        )

        response_placeholder = st.empty()

        for chunk in stream:
            delta = chunk.get("message", {}).get("content", "")
            full_response += delta
            # Live stream render
            response_placeholder.markdown(
                f"""
                <div style="display:flex;gap:12px;align-items:flex-start;">
                    <div class="avatar ai">✦</div>
                    <div class="bubble ai">{full_response}▌</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        response_placeholder.empty()

    except Exception as exc:
        typing_placeholder.empty()
        error_msg = (
            f"**Connection Error:** Could not reach Ollama at `{OLLAMA_HOST}`.\n\n"
            f"Make sure Ollama is running (`ollama serve`) and the model "
            f"`{st.session_state.selected_model}` is pulled.\n\n"
            f"*Details:* `{exc}`"
        )
        st.error(error_msg)
        # Remove the user message we just added since we failed
        st.session_state.messages.pop()
        st.session_state.total_queries -= 1
        return

    elapsed = round(time.time() - t_start, 2)
    st.session_state.last_response_time = elapsed

    typing_placeholder.empty()

    if full_response:
        add_message("assistant", full_response)

    st.rerun()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main() -> None:
    init_session_state()

    # Inject CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Check Ollama connection (cached for 30 s via manual flag)
    client = get_ollama_client()
    if st.session_state.ollama_online is None:
        st.session_state.ollama_online = check_ollama_connection(client)
    ollama_online: bool = st.session_state.ollama_online

    # ── Sidebar ──
    render_sidebar(client, ollama_online)

    # ── Main content ──
    render_hero()

    if not ollama_online:
        st.error(
            "**Ollama is not reachable.** "
            f"Start the server with `ollama serve` on `{OLLAMA_HOST}` "
            "and refresh this page."
        )
        if st.button("🔄 Retry Connection"):
            st.session_state.ollama_online = check_ollama_connection(client)
            st.rerun()
        return

    render_metrics(ollama_online)
    render_chat_history()
    render_input_area(ollama_online)

    # ── Footer ──
    last_rt = st.session_state.last_response_time
    rt_str  = f"Last response: {last_rt}s" if last_rt else "No responses yet"

    st.markdown(
        f"""
        <div class="footer">
            <span class="footer-left">
                <span class="footer-accent">Raj Patil AI Studio</span>
                &nbsp;·&nbsp; Data Science Portfolio
            </span>
            <span class="footer-right">
                {rt_str} &nbsp;·&nbsp; Streamlit + Ollama &nbsp;·&nbsp; © 2026
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
