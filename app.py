"""
app.py — AMIN Streamlit Frontend

A clean research UI that talks to the FastAPI backend running at
http://127.0.0.1:8000.

Run with:
    streamlit run app.py
"""

import time
import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE = "http://127.0.0.1:8000"
RESEARCH_ENDPOINT = f"{API_BASE}/research"
HEALTH_ENDPOINT = f"{API_BASE}/health"
REQUEST_TIMEOUT = 300  # seconds — long-running multi-agent pipelines need time


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AMIN · Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        /* ── Global typography ── */
        html, body, [class*="css"] { font-family: "Inter", sans-serif; }

        /* ── Header bar ── */
        .amin-header {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            border-radius: 12px;
            padding: 2rem 2.5rem;
            margin-bottom: 1.5rem;
            color: white;
        }
        .amin-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
        .amin-header p  { margin: 0.4rem 0 0; opacity: 0.75; font-size: 0.95rem; }

        /* ── Status badges ── */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }
        .badge-green  { background: #d1fae5; color: #065f46; }
        .badge-red    { background: #fee2e2; color: #991b1b; }
        .badge-yellow { background: #fef9c3; color: #713f12; }
        .badge-blue   { background: #dbeafe; color: #1e3a8a; }

        /* ── Pipeline steps ── */
        .pipeline-step {
            display: flex;
            align-items: flex-start;
            gap: 0.75rem;
            padding: 0.6rem 0.9rem;
            border-left: 3px solid #e5e7eb;
            margin-bottom: 0.4rem;
            border-radius: 0 6px 6px 0;
            background: #f9fafb;
        }
        .pipeline-step.active  { border-left-color: #3b82f6; background: #eff6ff; }
        .pipeline-step.done    { border-left-color: #22c55e; background: #f0fdf4; }

        /* ── Report card ── */
        .report-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        }

        /* ── Metric cards ── */
        .metric-row {
            display: flex;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            flex: 1;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1rem 1.25rem;
            text-align: center;
        }
        .metric-card .value { font-size: 1.75rem; font-weight: 700; color: #1e293b; }
        .metric-card .label { font-size: 0.78rem; color: #64748b; margin-top: 0.2rem; }

        /* ── Hide Streamlit branding ── */
        #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper: backend health check
# ---------------------------------------------------------------------------
@st.cache_data(ttl=10)
def check_backend_health() -> bool:
    """Returns True if the FastAPI backend is reachable and healthy."""
    try:
        r = httpx.get(HEALTH_ENDPOINT, timeout=5)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helper: call /research
# ---------------------------------------------------------------------------
def run_research(query: str, context: dict) -> dict:
    """POSTs to /research and returns the parsed JSON response."""
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        response = client.post(
            RESEARCH_ENDPOINT,
            json={"query": query, "context": context},
        )
        response.raise_for_status()
        return response.json()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    # Backend status
    st.markdown("**Backend Status**")
    healthy = check_backend_health()
    if healthy:
        st.markdown(
            '<span class="badge badge-green">🟢 API Online</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="badge badge-red">🔴 API Offline</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"Expecting FastAPI at `{API_BASE}`")

    st.divider()

    # Optional context fields
    st.markdown("**Research Context** *(optional)*")
    sector = st.text_input("Industry / Sector", placeholder="e.g. Electric Vehicles")
    date_range = st.text_input("Date Range", placeholder="e.g. 2024–2025")
    region = st.text_input("Geographic Focus", placeholder="e.g. North America")

    st.divider()
    st.markdown("**Resources**")
    st.markdown(
        "- [📖 API Docs](http://127.0.0.1:8000/docs)\n"
        "- [❤️ Health Check](http://127.0.0.1:8000/health)\n"
        "- [🗺 Architecture](./ARCHITECTURE.md)"
    )
    st.divider()
    st.caption("AMIN v0.1.0 · Powered by LangGraph + Gemini")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="amin-header">
        <h1>📊 Autonomous Market Intelligence Network</h1>
        <p>Multi-agent research pipeline · Supervisor → Web Researcher → Critic · Powered by Google Gemini</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------
st.markdown("### 🔍 Research Query")

query = st.text_area(
    label="Enter your market research question",
    placeholder=(
        "e.g. What are the key growth drivers and competitive dynamics "
        "in the global EV battery market for 2025?"
    ),
    height=110,
    label_visibility="collapsed",
)

col_btn, col_hint = st.columns([1, 5])
with col_btn:
    submit = st.button(
        "🚀 Run Research",
        type="primary",
        disabled=not healthy,
        use_container_width=True,
    )
with col_hint:
    if not healthy:
        st.warning(
            "⚠️ FastAPI backend is offline. Start it with `uvicorn main:app --reload`",
            icon="⚠️",
        )

st.divider()


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None
if "elapsed" not in st.session_state:
    st.session_state.elapsed = None


# ---------------------------------------------------------------------------
# Run pipeline on submit
# ---------------------------------------------------------------------------
if submit:
    if not query.strip():
        st.error("Please enter a research query before submitting.", icon="❌")
    else:
        st.session_state.result = None
        st.session_state.error = None
        st.session_state.elapsed = None

        # Build context dict from sidebar inputs
        context = {}
        if sector:
            context["sector"] = sector
        if date_range:
            context["date_range"] = date_range
        if region:
            context["region"] = region

        # ── Agent pipeline progress UI ───────────────────────────────────
        st.markdown("### 🤖 Agent Pipeline")
        progress_placeholder = st.empty()

        stages = [
            ("🧠", "Supervisor", "Decomposing query into research sub-tasks…"),
            ("🔎", "Web Researcher", "Gathering and summarising market intelligence…"),
            ("🔬", "Critic / Validator", "Scoring findings for relevance & confidence…"),
            ("📝", "Supervisor", "Assembling validated findings into final report…"),
        ]

        def render_stages(active_idx: int, done_indices: list[int]):
            html = ""
            for i, (icon, name, desc) in enumerate(stages):
                if i in done_indices:
                    css = "done"
                    status = "✅"
                elif i == active_idx:
                    css = "active"
                    status = "⏳"
                else:
                    css = ""
                    status = "○"
                html += (
                    f'<div class="pipeline-step {css}">'
                    f'  <span style="font-size:1.2rem">{status}</span>'
                    f'  <div>'
                    f'    <strong>{icon} {name}</strong>'
                    f'    <div style="font-size:0.83rem;color:#6b7280">{desc}</div>'
                    f'  </div>'
                    f'</div>'
                )
            progress_placeholder.markdown(html, unsafe_allow_html=True)

        spinner_placeholder = st.empty()
        t_start = time.time()

        with spinner_placeholder:
            with st.spinner("Agents are thinking… this may take 30–90 seconds."):
                # Animate stages while the blocking HTTP call runs
                # (Streamlit is single-threaded; we show an initial state then fire)
                render_stages(active_idx=0, done_indices=[])

                try:
                    data = run_research(query, context)
                    st.session_state.result = data
                    st.session_state.elapsed = round(time.time() - t_start, 1)
                    # Show all stages as done
                    render_stages(active_idx=-1, done_indices=list(range(len(stages))))
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.json().get("detail", str(exc))
                    st.session_state.error = f"API error {exc.response.status_code}: {detail}"
                    render_stages(active_idx=-1, done_indices=[])
                except Exception as exc:
                    st.session_state.error = str(exc)
                    render_stages(active_idx=-1, done_indices=[])

        spinner_placeholder.empty()


# ---------------------------------------------------------------------------
# Display error
# ---------------------------------------------------------------------------
if st.session_state.error:
    st.error(f"**Pipeline failed:** {st.session_state.error}", icon="🚨")


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if st.session_state.result:
    data = st.session_state.result

    st.markdown("---")
    st.markdown("### 📈 Research Results")

    # ── Metric summary row ───────────────────────────────────────────────
    approved_label = "✅ Approved" if data.get("approved") else "⚠️ Forced"
    approved_color = "#22c55e" if data.get("approved") else "#f59e0b"
    iterations = data.get("iterations", 0)
    plan = data.get("plan", [])

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="value" style="color:{approved_color}">{approved_label}</div>
                <div class="label">Critic Verdict</div>
            </div>
            <div class="metric-card">
                <div class="value">{iterations}</div>
                <div class="label">Refinement Iteration{"s" if iterations != 1 else ""}</div>
            </div>
            <div class="metric-card">
                <div class="value">{len(plan)}</div>
                <div class="label">Research Sub-tasks</div>
            </div>
            <div class="metric-card">
                <div class="value">{st.session_state.elapsed}s</div>
                <div class="label">Total Pipeline Time</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sub-tasks expander ───────────────────────────────────────────────
    if plan:
        with st.expander("🗂 Research Sub-tasks Generated by Supervisor", expanded=False):
            for i, task in enumerate(plan, 1):
                st.markdown(f"**{i}.** {task}")

    # ── Final report ─────────────────────────────────────────────────────
    st.markdown("#### 📄 Final Market Intelligence Report")
    report = data.get("final_report", "")

    if report:
        st.markdown(
            '<div class="report-card">',
            unsafe_allow_html=True,
        )
        st.markdown(report)
        st.markdown("</div>", unsafe_allow_html=True)

        # Download button
        st.download_button(
            label="⬇️ Download Report (.md)",
            data=report,
            file_name="amin_report.md",
            mime="text/markdown",
            use_container_width=False,
        )
    else:
        st.warning("The pipeline completed but returned an empty report.", icon="⚠️")

    # ── Raw JSON expander ────────────────────────────────────────────────
    with st.expander("🔧 Raw API Response", expanded=False):
        st.json(data)


# ---------------------------------------------------------------------------
# Empty state hint
# ---------------------------------------------------------------------------
elif not submit:
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 1rem;color:#9ca3af;">
            <div style="font-size:3.5rem">📊</div>
            <p style="font-size:1.05rem;margin-top:0.5rem">
                Enter a market research query above and click <strong>Run Research</strong>.
            </p>
            <p style="font-size:0.85rem">
                The multi-agent pipeline will decompose your query, gather intelligence,
                validate findings, and return a structured report.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
