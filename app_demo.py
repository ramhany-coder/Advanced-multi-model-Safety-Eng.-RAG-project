import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# Page config should be the first Streamlit command
st.set_page_config(
    page_title="Multimodal OSHA RAG Assistant",
    page_icon="🦺",
    layout="wide",
)

# Load local .env safely
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def load_streamlit_secrets():
    secret_keys = [
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGSMITH_PROJECT",
        "LANGCHAIN_PROJECT",
    ]

    try:
        for key in secret_keys:
            try:
                value = st.secrets.get(key, None)
                if value:
                    os.environ[key] = str(value)
            except Exception:
                continue
    except Exception:
        pass


load_streamlit_secrets()

# Base URL of the FastAPI app in api/app.py (run separately with
# `uvicorn api.app:app --reload`). Used only by the API Agent Tester tab below.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

# Import workflow AFTER secrets are loaded
try:
   from workflow import client as WorkflowClass
except Exception as e:
    st.error("Could not import workflow class from workflow.py")
    st.exception(e)
    st.stop()

st.title("🦺 Multimodal OSHA Compliance RAG Assistant")
st.caption(
    "Text + Image + Audio demo for a multilingual, privacy-aware, English-normalized RAG pipeline."
)


# -----------------------------
# Helpers
# -----------------------------
def file_to_base64(uploaded_file) -> Optional[str]:
    """Convert a Streamlit uploaded file into base64 string."""
    if uploaded_file is None:
        return None
    return base64.b64encode(uploaded_file.read()).decode("utf-8")


def get_file_extension(uploaded_file, default: str) -> str:
    """Extract extension without dot from uploaded filename."""
    if uploaded_file is None:
        return default
    suffix = Path(uploaded_file.name).suffix.replace(".", "").lower()
    return suffix or default


def init_workflow():
    """Initialize workflow once per session."""
    if "workflow" not in st.session_state:
        st.session_state.workflow = WorkflowClass()
    return st.session_state.workflow


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def build_initial_state(
    query: Optional[str],
    image_b64: Optional[str],
    audio_b64: Optional[str],
    audio_format: Optional[str],
    chat_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Build a flexible state dictionary compatible with your current/future State schema.

    Important:
    - The pipeline should store English internal response in `response`.
    - The final user-facing translated response can be `native_response` or `final_response`.
    - Rejected responses should be handled by `rejection_response_agent`, then translated.
    """

    state = {
        # User input
        "query": query if query else None,
        "image_bytes": image_b64,
        "audio_bytes": audio_b64,
        "audio_format": audio_format,

        # Conversation
        "chat_hist": chat_history,

        # Multilingual fields
        "language": None,
        "language_code": None,
        "origin_en": None,
        "clean_query": None,
        "eng_query": None,

        # Image/audio processing
        "image_bytes_cleaned": None,
        "image_exp": None,
        "audio_transcript": None,
        "audio_language": None,

        # RAG fields
        "rewritten_query": None,
        "merged": None,
        "context": None,
        "cached": None,

        # Response fields
        "response": None,          # English internal answer
        "native_response": None,   # Translated user-facing answer in your current agents.py
        "final_response": None,    # Alternative future name
        "rank": None,
        "rejected": None,
    }

    return state


def get_user_facing_response(result: Dict[str, Any]) -> str:
    """
    Choose the safest display response.

    Priority:
    1. native_response: current response_translator output.
    2. final_response: alternative naming.
    3. response: English fallback.
    """
    return (
        result.get("native_response")
        or result.get("final_response")
        or result.get("response")
        or "No response was generated."
    )


def is_rejected_result(result: Dict[str, Any]) -> bool:
    """
    Detect rejected answer safely.

    A response is considered rejected if:
    - rejection_response_agent set rejected=True, or
    - rank is present and <= 6.
    """
    if result.get("rejected") is True:
        return True

    rank = result.get("rank")
    try:
        return rank is not None and int(rank) <= 6
    except Exception:
        return False


def render_result_metadata(result: Dict[str, Any]):
    """Render compact metadata after assistant response."""
    cached = result.get("cached")
    rank = result.get("rank")
    language = result.get("language")
    language_code = result.get("language_code")
    k = result.get("k")

    source = "Local OSHA Knowledge Base"
    cache_status = "Cache hit" if cached else "Cache miss / new run"

    cols = st.columns(5)
    cols[0].metric("Source", source)
    cols[1].metric("Cache", cache_status)
    cols[2].metric("Rank", str(rank) if rank is not None else "N/A")
    cols[3].metric("Language", f"{language or 'N/A'} ({language_code or '-'})")
    cols[4].metric("Top-K", str(k) if k is not None else "N/A")


def render_debug_panel(result: Dict[str, Any]):
    """Optional developer/debug panel for portfolio demo transparency."""
    with st.expander("Developer Trace / Internal State"):
        st.markdown("#### English Query")
        st.code(result.get("eng_query") or "N/A")

        st.markdown("#### Rewritten Query")
        st.code(result.get("rewritten_query") or "N/A")

        st.markdown("#### Merged Retrieval Payload")
        st.code(result.get("merged") or "N/A")

        if result.get("audio_transcript"):
            st.markdown("#### Audio Transcript")
            st.code(result.get("audio_transcript"))

        if result.get("image_exp"):
            st.markdown("#### Image Explanation")
            st.code(result.get("image_exp"))

        st.markdown("#### English Internal Response")
        st.write(result.get("response") or "N/A")

        context = result.get("context")
        if context:
            st.markdown("#### Retrieved Context Preview")
            st.write(context[:3] if isinstance(context, list) else context)


# -----------------------------
# API Agent Tester helpers (calls the FastAPI app in api/app.py)
# -----------------------------

# Mirrors api.endpoints.STAGE_AGENTS -- kept as a static fallback so the tab
# still works if the API isn't reachable yet when the page loads.
FALLBACK_STAGE_NAMES = [
    "audio-transcription",
    "language-detector",
    "query-pii",
    "image-pii",
    "user-query-translator",
    "rewrite",
    "image-analysis",
    "merger",
    "cache-check",
    "cache-write",
    "doc-id-mapper",
    "retriever",
    "reranker",
    "responser",
    "ranker",
    "rejection-response",
    "response-translator",
]

# Small example StatePayload per stage, so the JSON editor starts with
# something runnable instead of an empty object.
STAGE_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "audio-transcription": {"audio_bytes": "<base64_audio>", "audio_format": "mp3"},
    "language-detector": {"query": "هل العامل محتاج حزام أمان وهو واقف على السقالة؟"},
    "query-pii": {"query": "My name is John Smith, is this scaffold safe?", "language_code": "en"},
    "image-pii": {"image_bytes": "<base64_image>"},
    "user-query-translator": {"clean_query": "Does a worker need fall protection?", "language": "English"},
    "rewrite": {"eng_query": "Does a worker need fall protection on a scaffold?", "chat_hist": []},
    "image-analysis": {"image_bytes_cleaned": "<base64_cleaned_image>"},
    "merger": {"rewritten_query": "fall protection scaffold requirements", "image_exp": ""},
    "cache-check": {"merged": "fall protection scaffold requirements"},
    "cache-write": {"merged": "fall protection scaffold requirements", "response": "Example grounded answer.", "cached": False},
    "doc-id-mapper": {"merged": "fall protection scaffold requirements"},
    "retriever": {"merged": "fall protection scaffold requirements", "k": 5},
    "reranker": {"merged": "fall protection scaffold requirements", "context": [], "content": []},
    "responser": {"merged": "fall protection scaffold requirements", "context": [], "content": []},
    "ranker": {"eng_query": "Does a worker need fall protection?", "response": "Example grounded answer.", "context": [], "content": []},
    "rejection-response": {"rank": 3},
    "response-translator": {"response": "Example grounded answer.", "language": "Arabic", "language_code": "ar"},
}


@st.cache_data(ttl=30)
def fetch_stage_names(base_url: str) -> List[str]:
    """GET /api/agents from the running FastAPI app; falls back to the static list."""
    try:
        resp = requests.get(f"{base_url}/api/agents", timeout=3)
        resp.raise_for_status()
        agents = resp.json().get("agents")
        if agents:
            return agents
    except Exception:
        pass
    return FALLBACK_STAGE_NAMES


def call_agent_api(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST payload to {base_url}{path} and return {"ok", "status_code", "data"}."""
    try:
        resp = requests.post(f"{base_url}{path}", json=payload, timeout=120)
    except requests.exceptions.RequestException as e:
        return {"ok": False, "status_code": None, "data": {"error": str(e)}}

    try:
        data = resp.json()
    except ValueError:
        data = {"error": resp.text}

    return {"ok": resp.ok, "status_code": resp.status_code, "data": data}


# -----------------------------
# Session Initialization
# -----------------------------
init_session_state()
workflow = init_workflow()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Demo Inputs")

    uploaded_image = st.file_uploader(
        "Upload construction/site image",
        type=["png", "jpg", "jpeg", "webp"],
        help="Optional: upload an image for visual safety analysis.",
    )

    uploaded_audio = st.file_uploader(
        "Upload audio note",
        type=["mp3", "wav", "m4a", "ogg", "webm"],
        help="Optional: upload an audio question or field note.",
    )

    show_debug = st.toggle(
        "Show developer trace",
        value=True,
        help="Useful for portfolio demo: shows translation, merged query, rank, and retrieved context.",
    )

    clear_chat = st.button("Clear chat")
    if clear_chat:
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

    st.divider()
    st.markdown("### Suggested demo prompts")
    st.markdown(
        """
- هل العامل محتاج حزام أمان وهو واقف على السقالة؟
- Does this scaffold setup need fall protection?
- Inspect this image for possible OSHA construction safety issues.
- Summarize the safety concern from my voice note.
        """
    )


tab_chat, tab_api = st.tabs(["💬 Chat Demo", "🧪 API Agent Tester"])


with tab_chat:
    # -----------------------------
    # Chat History
    # -----------------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    # -----------------------------
    # Input Area
    # -----------------------------
    query = st.chat_input(
        "Ask an OSHA safety question in English, Arabic, or upload image/audio from the sidebar..."
    )

    run_file_only = False
    if uploaded_image or uploaded_audio:
        run_file_only = st.button(
            "Run analysis with uploaded file(s)",
            type="primary",
            help="Use this when you uploaded image/audio without typing a chat message.",
        )


    # -----------------------------
    # Main Execution
    # -----------------------------
    should_run = bool(query) or run_file_only

    if should_run:
        user_display_text = query or "Analyze the uploaded file(s)."

        st.session_state.messages.append({"role": "user", "content": user_display_text})

        with st.chat_message("user"):
            st.write(user_display_text)
            if uploaded_image:
                st.image(uploaded_image, caption="Uploaded image", use_container_width=True)
            if uploaded_audio:
                st.audio(uploaded_audio)

        # Important: read uploaded files once per run after displaying.
        # Streamlit file object pointer can be consumed, so seek back if possible.
        image_b64 = None
        audio_b64 = None
        audio_format = None

        if uploaded_image:
            uploaded_image.seek(0)
            image_b64 = file_to_base64(uploaded_image)

        if uploaded_audio:
            uploaded_audio.seek(0)
            audio_b64 = file_to_base64(uploaded_audio)
            audio_format = get_file_extension(uploaded_audio, "mp3")

        chat_history = st.session_state.messages[:-1]

        initial_state = build_initial_state(
            query=query,
            image_b64=image_b64,
            audio_b64=audio_b64,
            audio_format=audio_format,
            chat_history=chat_history,
        )

        with st.chat_message("assistant"):
            with st.spinner("Running multimodal RAG pipeline..."):
                try:
                    result = workflow.run(initial_state)
                    st.session_state.last_result = result

                    response = get_user_facing_response(result)
                    rejected = is_rejected_result(result)

                    if rejected:
                        st.warning(
                            "The QA ranker rejected the generated answer as insufficiently reliable. "
                            "Showing a safe fallback response instead."
                        )
                    else:
                        st.success("Answer generated and validated.")

                    st.write(response)

                    render_result_metadata(result)

                    if show_debug:
                        render_debug_panel(result)

                except Exception as e:
                    response = (
                        "The demo encountered a runtime error while processing the request. "
                        "Please check the workflow, API keys, vector store, and uploaded file format."
                    )
                    st.error(response)
                    with st.expander("Error details"):
                        st.exception(e)

                    result = {"response": response, "native_response": response}

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": get_user_facing_response(st.session_state.last_result or result),
            }
        )


    # -----------------------------
    # Footer
    # -----------------------------
    st.divider()
    st.caption(
        "Portfolio demo: multilingual multimodal RAG with PII filtering, English-normalized retrieval/cache, QA ranking, and rejection-safe output."
    )


with tab_api:
    st.caption(
        f"Calls the FastAPI app in `api/app.py` over HTTP at `{API_BASE_URL}` "
        "(start it with `uvicorn api.app:app --reload`). Each agent can be exercised "
        "in isolation here, independently of the in-process chat demo on the other tab."
    )

    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=2)
        api_reachable = health.ok
    except Exception:
        api_reachable = False

    if not api_reachable:
        st.warning(
            f"Could not reach the API at {API_BASE_URL}. Start it with "
            "`uvicorn api.app:app --reload` from the project root, then reload this page."
        )

    st.subheader("Run a single agent")

    stage_names = fetch_stage_names(API_BASE_URL)
    selected_stage = st.selectbox("Agent stage", stage_names, key="stage_select")

    default_payload = STAGE_EXAMPLES.get(selected_stage, {})
    payload_text = st.text_area(
        "Request body (StatePayload JSON -- only the fields this agent reads matter)",
        value=json.dumps(default_payload, indent=2, ensure_ascii=False),
        height=200,
        key=f"payload_{selected_stage}",
    )

    if st.button("Run agent", type="primary", key="run_stage_btn"):
        try:
            payload = json.loads(payload_text) if payload_text.strip() else {}
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON: {e}")
        else:
            with st.spinner(f"Calling POST /api/agents/{selected_stage} ..."):
                outcome = call_agent_api(API_BASE_URL, f"/api/agents/{selected_stage}", payload)

            if outcome["ok"]:
                st.success(f"Stage '{selected_stage}' completed.")
            else:
                st.error(f"Stage '{selected_stage}' failed (HTTP {outcome['status_code']}).")
            st.json(outcome["data"])

    st.divider()
    st.subheader("Run the full pipeline")

    api_query = st.text_input("Query", key="api_query")
    col1, col2 = st.columns(2)
    with col1:
        api_image = st.file_uploader(
            "Image (optional)", type=["png", "jpg", "jpeg", "webp"], key="api_image"
        )
    with col2:
        api_audio = st.file_uploader(
            "Audio (optional)", type=["mp3", "wav", "m4a", "ogg", "webm"], key="api_audio"
        )
    api_chat_hist_text = st.text_area("chat_hist (JSON list)", value="[]", key="api_chat_hist")

    if st.button("Run full pipeline", type="primary", key="run_pipeline_btn"):
        try:
            chat_hist = json.loads(api_chat_hist_text) if api_chat_hist_text.strip() else []
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON in chat_hist: {e}")
        else:
            image_b64 = None
            audio_b64 = None
            audio_format = None

            if api_image:
                api_image.seek(0)
                image_b64 = file_to_base64(api_image)
            if api_audio:
                api_audio.seek(0)
                audio_b64 = file_to_base64(api_audio)
                audio_format = get_file_extension(api_audio, "mp3")

            pipeline_payload = {
                "query": api_query or None,
                "image_bytes": image_b64,
                "audio_bytes": audio_b64,
                "audio_format": audio_format,
                "chat_hist": chat_hist,
            }

            with st.spinner("Calling POST /api/pipeline/run ..."):
                outcome = call_agent_api(API_BASE_URL, "/api/pipeline/run", pipeline_payload)

            if outcome["ok"]:
                st.success("Pipeline completed.")
                st.write(get_user_facing_response(outcome["data"]))
                timings = outcome["data"].get("stage_timings")
                if timings:
                    st.markdown("#### Stage timings (seconds)")
                    st.json(timings)
            else:
                st.error(f"Pipeline failed (HTTP {outcome['status_code']}).")
            st.json(outcome["data"])
