import base64
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

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

# Base URL of the FastAPI app in api/app.py. It is started automatically,
# in-process, the first time this app runs (see start_embedded_api_server
# below) -- no separate `uvicorn` command needed. Used by the API Agent
# Tester tab below.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")

USER_AVATAR = "🧑‍💻"
ASSISTANT_AVATAR = "🦺"

# Import the instrumented pipeline runner AFTER secrets are loaded. This is
# the same run_pipeline/build_initial_state used by the FastAPI app (see
# api/workflow.py) -- it wraps every agent node with a timer and returns the
# full final state plus per-agent ("stage_timings") and overall
# ("total_latency_seconds") latency, so the Chat Demo tab and the API Agent
# Tester tab report identical traces.
try:
    from api.workflow import (
        build_initial_state as build_pipeline_state,
        run_pipeline,
        PipelineStageError,
    )
except Exception as e:
    st.error("Could not import the instrumented pipeline runner from api/workflow.py")
    st.exception(e)
    st.stop()

from agents.prompt_registry import PROMPT_FIELDS, get_defaults


# -----------------------------
# Embedded FastAPI/uvicorn server
# -----------------------------
def _api_is_reachable(base_url: str) -> bool:
    try:
        return requests.get(f"{base_url}/health", timeout=1).ok
    except Exception:
        return False


@st.cache_resource(show_spinner=False)
def start_embedded_api_server(base_url: str):
    """
    Launch the FastAPI app (api/app.py) with uvicorn in a background daemon
    thread inside this same Streamlit process, so the user never has to run
    `uvicorn api.app:app --reload` separately.

    st.cache_resource makes this run exactly once per Streamlit server
    process, even though the script re-executes on every rerun.
    """
    if _api_is_reachable(base_url):
        # Something (e.g. a manually started uvicorn) is already serving here.
        return None

    import uvicorn

    from api.app import app as fastapi_app

    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000

    config = uvicorn.Config(fastapi_app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="embedded-uvicorn", daemon=True)
    thread.start()

    deadline = time.time() + 15
    while time.time() < deadline:
        if _api_is_reachable(base_url):
            break
        time.sleep(0.25)

    return server


start_embedded_api_server(API_BASE_URL)

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


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_result" not in st.session_state:
        st.session_state.last_result = None


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


def stream_words(text: str, delay: float = 0.02):
    """Yield text word-by-word so st.write_stream can render a typing effect."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)


def render_chat_message(message: Dict[str, Any], streaming: bool = False):
    """Render one chat turn (history replay or the just-generated turn)."""
    role = message["role"]
    avatar = USER_AVATAR if role == "user" else ASSISTANT_AVATAR

    with st.chat_message(role, avatar=avatar):
        if message.get("image"):
            st.image(message["image"], caption="Uploaded image", use_container_width=True)
        if message.get("audio"):
            st.audio(message["audio"])

        if role == "assistant" and message.get("rejected"):
            st.warning(
                "The QA ranker rejected the generated answer as insufficiently reliable. "
                "Showing a safe fallback response instead."
            )
        elif role == "assistant" and message.get("result"):
            st.success("Answer generated and validated.")

        if streaming:
            st.write_stream(stream_words(message["content"]))
        else:
            st.write(message["content"])

        result = message.get("result")
        if result:
            with st.expander("Details & developer trace"):
                render_result_metadata(result)
                if message.get("show_debug"):
                    render_debug_panel(result)


def render_result_metadata(result: Dict[str, Any]):
    """Render compact metadata after assistant response."""
    cached = result.get("cached")
    rank = result.get("rank")
    language = result.get("language")
    language_code = result.get("language_code")
    k = result.get("k")
    total_latency = result.get("total_latency_seconds")

    source = "Local OSHA Knowledge Base"
    cache_status = "Cache hit" if cached else "Cache miss / new run"

    cols = st.columns(6)
    cols[0].metric("Source", source)
    cols[1].metric("Cache", cache_status)
    cols[2].metric("Rank", str(rank) if rank is not None else "N/A")
    cols[3].metric("Language", f"{language or 'N/A'} ({language_code or '-'})")
    cols[4].metric("Top-K", str(k) if k is not None else "N/A")
    cols[5].metric(
        "Total latency",
        f"{total_latency:.2f}s" if total_latency is not None else "N/A",
    )

    if result.get("failed_stage"):
        st.error(
            f"Pipeline aborted at stage '{result['failed_stage']}' after "
            f"{result.get('stage_latency_seconds', 'N/A')}s: {result.get('error')}"
        )


# Fields whose raw value is large/binary and unhelpful to dump verbatim in the
# state trace -- shown as a length/preview instead of the full payload.
_LARGE_FIELDS = {"image_bytes", "image_bytes_cleaned", "audio_bytes"}

# Rendered separately (latency, curated highlights, chat history), so the
# generic "every other variable" dump below doesn't repeat them.
_TRACE_HIDDEN_FIELDS = {
    "stage_timings",
    "total_latency_seconds",
    "failed_stage",
    "stage_latency_seconds",
    "chat_hist",
}


def _preview_state_value(key: str, value: Any) -> Any:
    if key in _LARGE_FIELDS and isinstance(value, str) and value:
        return f"<{len(value)} base64 chars, omitted>"
    if isinstance(value, str) and len(value) > 1000:
        return value[:1000] + f"... <{len(value)} chars total, truncated>"
    return value


def render_stage_timings(result: Dict[str, Any]):
    """Per-agent latency (each timed node in workflow.py) plus overall latency."""
    timings: Dict[str, float] = result.get("stage_timings") or {}
    total_latency = result.get("total_latency_seconds")

    if not timings and total_latency is None:
        st.caption("No latency data on this result (pipeline runner may have failed before timing started).")
        return

    st.markdown("#### Latency -- per agent and overall")
    st.metric("Overall pipeline latency", f"{total_latency:.3f}s" if total_latency is not None else "N/A")

    if timings:
        ordered = dict(sorted(timings.items(), key=lambda kv: kv[1], reverse=True))
        st.bar_chart(ordered)
        st.table(
            [{"agent": name, "latency_seconds": seconds} for name, seconds in ordered.items()]
        )


def render_full_state(result: Dict[str, Any]):
    """Dump every remaining state variable so the full pipeline trace is visible."""
    st.markdown("#### Full pipeline state (all variables)")
    state_view = {
        key: _preview_state_value(key, value)
        for key, value in sorted(result.items())
        if key not in _TRACE_HIDDEN_FIELDS
    }
    st.json(state_view, expanded=False)


def render_debug_panel(result: Dict[str, Any]):
    """Optional developer/debug panel for portfolio demo transparency."""
    with st.expander("Developer Trace / Internal State", expanded=False):
        render_stage_timings(result)
        st.divider()

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

        content = result.get("content")
        if content:
            st.markdown("#### Retrieved Content Preview")
            st.write(content[:3] if isinstance(content, list) else content)

        st.divider()
        render_full_state(result)


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
    "retriever": {"merged": "fall protection scaffold requirements", "k": 5},
    "reranker": {"merged": "fall protection scaffold requirements", "content": []},
    "responser": {"merged": "fall protection scaffold requirements", "content": []},
    "ranker": {"eng_query": "Does a worker need fall protection?", "response": "Example grounded answer.", "content": []},
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


tab_chat, tab_api, tab_prompts = st.tabs(
    ["💬 Chat Demo", "🧪 API Agent Tester", "✏️ Prompt Editor"]
)


with tab_chat:
    # -----------------------------
    # Chat History
    # -----------------------------
    for message in st.session_state.messages:
        render_chat_message(message)


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

        # .getvalue() reads the full buffer without consuming/moving the
        # cursor, so it's safe to call before the seek(0)+read() below.
        image_raw = uploaded_image.getvalue() if uploaded_image else None
        audio_raw = uploaded_audio.getvalue() if uploaded_audio else None

        user_message = {
            "role": "user",
            "content": user_display_text,
            "image": image_raw,
            "audio": audio_raw,
        }
        st.session_state.messages.append(user_message)
        render_chat_message(user_message)

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

        initial_state = build_pipeline_state(
            query=query,
            image_bytes=image_b64,
            audio_bytes=audio_b64,
            audio_format=audio_format,
            chat_hist=chat_history,
        )

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            with st.spinner("Running multimodal RAG pipeline..."):
                try:
                    # run_pipeline (api/workflow.py) times every agent node and
                    # returns the full final state plus stage_timings /
                    # total_latency_seconds alongside it.
                    result = run_pipeline(initial_state)
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

                except PipelineStageError as e:
                    # A stage raised, but every stage that completed before it
                    # was still timed and its output is still in e.state --
                    # keep that trace instead of throwing it away.
                    result = {
                        **e.state,
                        "stage_timings": e.stage_timings,
                        "total_latency_seconds": e.total_latency_seconds,
                        "failed_stage": e.stage,
                        "stage_latency_seconds": round(e.elapsed, 3),
                        "error": str(e.original),
                    }
                    st.session_state.last_result = result
                    response = (
                        f"The pipeline failed at the '{e.stage}' stage after {e.elapsed:.2f}s. "
                        "See the developer trace below for the partial state and per-agent timings."
                    )
                    rejected = False
                    st.error(response)
                    with st.expander("Error details"):
                        st.exception(e.original)

                except Exception as e:
                    response = (
                        "The demo encountered a runtime error while processing the request. "
                        "Please check the workflow, API keys, vector store, and uploaded file format."
                    )
                    rejected = False
                    result = None
                    st.error(response)
                    with st.expander("Error details"):
                        st.exception(e)

            # Streamed outside the spinner so the typing effect is visible.
            st.write_stream(stream_words(response))

            if result:
                with st.expander("Details & developer trace"):
                    render_result_metadata(result)
                    if show_debug:
                        render_debug_panel(result)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
                "result": result,
                "rejected": rejected,
                "show_debug": show_debug,
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
        "(started automatically, in-process, when this Streamlit app launches). "
        "Each agent can be exercised in isolation here, independently of the "
        "in-process chat demo on the other tab."
    )

    api_reachable = _api_is_reachable(API_BASE_URL)

    if not api_reachable:
        st.warning(
            f"Could not reach the embedded API at {API_BASE_URL} yet. It may still be "
            "starting up -- reload this page in a moment. If it keeps failing, check the "
            "terminal running Streamlit for startup errors (e.g. the port may be in use)."
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
                total_latency = outcome["data"].get("total_latency_seconds")
                if total_latency is not None:
                    st.metric("Overall pipeline latency", f"{total_latency:.3f}s")
                timings = outcome["data"].get("stage_timings")
                if timings:
                    st.markdown("#### Stage timings (seconds, per agent)")
                    ordered = dict(sorted(timings.items(), key=lambda kv: kv[1], reverse=True))
                    st.bar_chart(ordered)
                    st.json(ordered)
            else:
                st.error(f"Pipeline failed (HTTP {outcome['status_code']}).")
                total_latency = (outcome["data"].get("detail") or {}).get("total_latency_seconds") if isinstance(outcome["data"], dict) else None
                if total_latency is not None:
                    st.metric("Latency before failure", f"{total_latency:.3f}s")
            st.json(outcome["data"])


with tab_prompts:
    st.caption(
        "Live-edit any agent's system prompt to see how it changes the pipeline's "
        "behavior, without restarting the app. Changes take effect immediately for "
        "the Chat Demo, the API Agent Tester tab, and the embedded API server -- "
        "for the whole running app, not just this browser tab. They are **not** "
        "written back to the prompts.py files and are lost on restart; use "
        "'Reset to default' (or restart the app) to discard them."
    )

    defaults = get_defaults()

    if st.button("Reset ALL prompts to defaults", key="reset_all_prompts"):
        for pf in PROMPT_FIELDS:
            pf.set(defaults[pf.key])
        st.success("All prompts reset to defaults.")
        st.rerun()

    agents_grouped: Dict[str, List] = {}
    for pf in PROMPT_FIELDS:
        agents_grouped.setdefault(pf.agent, []).append(pf)

    for agent_name, fields in agents_grouped.items():
        with st.expander(f"🧩 {agent_name}", expanded=False):
            for pf in fields:
                current_value = pf.get()
                is_modified = current_value != defaults[pf.key]

                label = pf.label + (" — modified" if is_modified else "")
                if pf.help:
                    st.caption(pf.help)

                edited = st.text_area(
                    label,
                    value=current_value,
                    height=320,
                    key=f"prompt_edit_{pf.key}",
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Save", key=f"save_{pf.key}", type="primary"):
                        missing = pf.missing_placeholders(edited)
                        if missing:
                            st.error(
                                "Cannot save -- missing required placeholder(s): "
                                f"{', '.join(missing)}"
                            )
                        else:
                            pf.set(edited)
                            st.success(f"Saved '{pf.label}' for {agent_name}.")
                            st.rerun()
                with col_b:
                    if st.button(
                        "Reset to default",
                        key=f"reset_{pf.key}",
                        disabled=not is_modified,
                    ):
                        pf.set(defaults[pf.key])
                        st.success(f"Reset '{pf.label}' for {agent_name} to default.")
                        st.rerun()
