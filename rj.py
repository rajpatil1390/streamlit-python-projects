import streamlit as st
from ollama import Client

# -----------------------------
# OLLAMA CONNECTION
# -----------------------------
client = Client(host="http://localhost:11434")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Raj AI Assistant",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.big-title {
    text-align:center;
    font-size:48px;
    font-weight:bold;
    background: linear-gradient(90deg,#00DBDE,#FC00FF);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle {
    text-align:center;
    color:#AAAAAA;
    font-size:18px;
}

.response-box {
    background-color:#1E1E1E;
    padding:20px;
    border-radius:15px;
    border-left:5px solid #00DBDE;
}

.stButton>button {
    width:100%;
    background:linear-gradient(90deg,#00DBDE,#FC00FF);
    color:white;
    border:none;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
    padding:10px;
}

.stTextArea textarea {
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/4712/4712109.png",
        width=120
    )

    st.title("⚙️ Model Settings")

    st.info("""
    **Model:** DeepSeek-R1 1.5B

    **Framework:** Ollama

    **Developer:** Raj Patil

    **Role:** Data Scientist & AI Engineer
    """)

    st.markdown("---")

    st.metric("Model Status", "Online ✅")
    st.metric("Inference Engine", "Ollama")

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    '<div class="big-title">🤖 Raj AI Analytics Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Powered by DeepSeek-R1 • Streamlit • Ollama</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# -----------------------------
# METRICS
# -----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "DeepSeek")

with col2:
    st.metric("Version", "1.5B")

with col3:
    st.metric("Status", "Ready")

# -----------------------------
# INPUT
# -----------------------------
prompt = st.text_area(
    "💬 Ask Anything",
    placeholder="Example: Explain Random Forest in Data Science...",
    height=200
)

# -----------------------------
# BUTTON
# -----------------------------
if st.button("🚀 Generate AI Response"):

    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("🧠 AI is thinking..."):

            response = client.chat(
                model="deepseek-r1:1.5b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response["message"]["content"]

        st.success("Response Generated Successfully")

        st.markdown("### 📊 AI Response")

        st.markdown(
            f"""
            <div class="response-box">
            {answer}
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption(
    "© 2026 Raj Patil | Data Science Portfolio Project | Ollama + Streamlit"
)