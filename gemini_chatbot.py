import os

import streamlit as st
from google import genai


MODEL_NAME = "gemini-3.8-flash"


def get_gemini_api_key():
    """Read the Gemini API key from Streamlit secrets or environment."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if api_key:
            return api_key
    except Exception:
        pass

    return os.getenv("GEMINI_API_KEY", "")


def build_kcet_context(df):
    """Build a compact context summary from the KCET dataset."""
    colleges = sorted(df["College"].dropna().astype(str).unique().tolist())
    branches = sorted(df["Branch"].dropna().astype(str).unique().tolist())
    locations = sorted(df["Location"].dropna().astype(str).unique().tolist())

    cutoff_columns = [
        str(column)
        for column in df.columns
        if column not in {"College", "Branch", "Location", "CETCode"}
    ]

    return f"""
You are KCET AI Assistant, a helpful admission guidance chatbot.

PROJECT INFORMATION:
- Project: KCET Predictor Pro
- Purpose: Help students explore engineering colleges in Karnataka.
- Dataset records: {len(df)}
- Unique colleges: {len(colleges)}
- Engineering branches: {len(branches)}
- Locations: {len(locations)}

AVAILABLE BRANCHES:
{", ".join(branches[:60])}

AVAILABLE LOCATIONS:
{", ".join(locations[:60])}

AVAILABLE CUTOFF CATEGORIES:
{", ".join(cutoff_columns)}

IMPORTANT RULES:
1. Answer in simple, understandable English.
2. Focus on KCET, Karnataka engineering colleges, admissions,
   counselling, branches, ranks and cutoffs.
3. Give step-by-step explanations when helpful.
4. Never invent cutoff ranks, fees, placement figures or guarantees.
5. Explain that historical cutoffs can change every year.
6. If exact information is unavailable, say so clearly.
7. Do not claim to be an official KEA representative.
8. Be friendly and helpful to students.
9. For exact college predictions, guide students to the ML Predictor page.
10. You may also answer general educational and career questions.
"""


def initialize_chat():
    """Initialize chat history once per Streamlit session."""
    if "gemini_messages" not in st.session_state:
        st.session_state.gemini_messages = [
            {
                "role": "assistant",
                "content": (
                    "Hello! 👋 Welcome to KCET AI Assistant. "
                    "I can help with KCET ranks, engineering branches, "
                    "colleges, cutoffs and admission guidance. "
                    "What would you like to know?"
                ),
            }
        ]


def render_chatbot(df):
    """Render the Gemini AI chatbot page."""
    st.markdown(
        '<div class="eyebrow">YOUR PERSONAL ADMISSION ASSISTANT</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-title">KCET AI Assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="page-subtitle">Ask questions about KCET, engineering '
        'colleges, branches, ranks and admission guidance.</div>',
        unsafe_allow_html=True,
    )

    initialize_chat()
    api_key = get_gemini_api_key()

    if not api_key:
        st.error("Gemini API key is missing.")
        st.info(
            "Add GEMINI_API_KEY to .streamlit/secrets.toml "
            "before using the chatbot."
        )
        return

    try:
        client = genai.Client(api_key=api_key)
    except Exception as error:
        st.error("Could not initialize the Gemini client.")
        st.caption(f"Technical details: {error}")
        return

    col1, col2 = st.columns([4, 1])
    with col1:
        st.caption(f"Powered by Google Gemini · {MODEL_NAME}")
    with col2:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.gemini_messages = []
            initialize_chat()
            st.rerun()

    st.markdown("---")

    for message in st.session_state.gemini_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if len(st.session_state.gemini_messages) == 1:
        st.markdown("### Try asking")
        suggestions = [
            "What is KCET counselling?",
            "Which engineering branches are available?",
            "How does KCET cutoff rank work?",
            "How should I choose my engineering college?",
        ]
        for index, question in enumerate(suggestions):
            if st.button(
                question,
                key=f"gemini_suggestion_{index}",
                use_container_width=True,
            ):
                st.session_state.gemini_pending_question = question
                st.rerun()

    typed_question = st.chat_input("Ask your KCET question here...")
    suggested_question = st.session_state.pop(
        "gemini_pending_question", None
    )
    user_question = typed_question or suggested_question

    if not user_question:
        return

    user_question = user_question.strip()
    if not user_question:
        return

    st.session_state.gemini_messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Build a short conversation transcript for the Interactions API.
    recent_messages = st.session_state.gemini_messages[-12:]
    transcript = []
    for message in recent_messages:
        speaker = "User" if message["role"] == "user" else "Assistant"
        transcript.append(f"{speaker}: {message['content']}")

    prompt = (
        build_kcet_context(df)
        + "\n\nCONVERSATION:\n"
        + "\n".join(transcript)
        + "\n\nAssistant:"
    )

    with st.chat_message("assistant"):
        with st.spinner("Gemini is thinking..."):
            try:
                # Use Google's current Interactions API.
                interaction = client.interactions.create(
                    model=MODEL_NAME,
                    input=prompt,
                )
                answer = getattr(interaction, "output_text", None)

                if not answer:
                    answer = (
                        "I couldn't generate a text response. "
                        "Please try asking in another way."
                    )

                st.markdown(answer)
                st.session_state.gemini_messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                st.error(
                    "Gemini could not process your question. "
                    "Check the API key, internet connection, SDK version "
                    "and model access."
                )
                st.caption(f"Technical details: {error}")
