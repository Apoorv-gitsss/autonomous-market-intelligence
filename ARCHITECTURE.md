# ARCHITECTURE.md — Autonomous Market Intelligence Network (AMIN)

> **Version:** 0.1.0 | **Stack:** FastAPI · LangGraph · Google Gemini · Python 3.9+

This document is the authoritative system design reference for AMIN. It covers
the high-level architecture, component responsibilities, multi-agent state data
flow, token usage strategy, and a forward-looking scaling roadmap.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Technology Stack](#2-technology-stack)
3. [High-Level Architecture Diagram](#3-high-level-architecture-diagram)
4. [Component Breakdown](#4-component-breakdown)
   - 4.1 [FastAPI Layer](#41-fastapi-layer)
   - 4.2 [LangGraph StateGraph Engine](#42-langgraph-stategraph-engine)
   - 4.3 [Google Gemini LLM Backend](#43-google-gemini-llm-backend)
5. [Multi-Agent State Data Flow](#5-multi-agent-state-data-flow)
   - 5.1 [Shared State Schema](#51-shared-state-schema)
   - 5.2 [Agent Pipeline Walkthrough](#52-agent-pipeline-walkthrough)
   - 5.3 [Refinement Loop Logic](#53-refinement-loop-logic)
6. [Token Usage Management](#6-token-usage-management)
7. [API Contract](#7-api-contract)
8. [Environment Configuration](#8-environment-configuration)
9. [Future Scaling Recommendations](#9-future-scaling-recommendations)
10. [Known Limitations (v0.1.0)](#10-known-limitations-v010)

---

## 1. System Overview

AMIN is a **multi-agent market intelligence pipeline** that accepts a free-text
research query and returns a structured, validated market intelligence report.
The system is designed around three core principles:

| Principle | Implementation |
|---|---|
| **Separation of concerns** | Each agent has a single, well-defined responsibility |
| **Self-correction** | The Critic node triggers refinement loops before a report is published |
| **Stateless HTTP API** | All graph state is ephemeral per request; no session affinity required |

---

## 2. Technology Stack

| Layer | Technology | Version | Role |
|---|---|---|---|
| API Server | FastAPI + Uvicorn | ≥ 0.111 / ≥ 0.30 | HTTP request handling, OpenAPI docs |
| Orchestration | LangGraph `StateGraph` | ≥ 0.2.0 | Directed agent graph with conditional edges |
| LLM Integration | `langchain-google-genai` | ≥ 2.0.0 | Gemini API client adapter |
| LLM Backend | Google Gemini 2.0 Flash | — | Language model inference |
| Data Validation | Pydantic v2 | ≥ 2.7.0 | Request/response schema enforcement |
| Config Management | `python-dotenv` | ≥ 1.0.0 | `.env` file loading at startup |
| HTTP Client | `httpx` | ≥ 0.27.0 | Async-compatible HTTP for tool integrations |

---

## 3. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT / USER                              │
└───────────────────────────────┬─────────────────────────────────────┘
                                │  HTTP POST /research
                                │  { "query": "...", "context": {} }
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        FASTAPI APPLICATION                          │
│                                                                     │
│   ┌──────────────┐     ┌─────────────────┐     ┌───────────────┐   │
│   │  POST        │     │  Pydantic        │     │  GET          │   │
│   │  /research   │────►│  Validation      │     │  /health      │   │
│   └──────────────┘     └────────┬────────┘     └───────────────┘   │
│                                 │ AgentState (initial)              │
└─────────────────────────────────┼───────────────────────────────────┘
                                  │  compiled_graph.ainvoke()
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH STATEGRAPH ENGINE                      │
│                                                                     │
│   START                                                             │
│     │                                                               │
│     ▼                                                               │
│  ┌──────────────────────────────────────────────────┐              │
│  │              SUPERVISOR NODE                      │              │
│  │                                                   │              │
│  │  Pass 1 (plan==[]):  Decompose query → sub_tasks  │              │
│  │  Pass 2 (plan!=[]):  Assemble final_report        │              │
│  └──────────────────┬───────────────────────────────┘              │
│                     │ sub_tasks[]                                   │
│                     ▼                                               │
│  ┌──────────────────────────────────────────────────┐              │
│  │            WEB RESEARCHER NODE                    │              │
│  │                                                   │              │
│  │  For each sub_task → LLM call → structured        │              │
│  │  finding {summary, sources, sub_task}             │              │
│  │  iteration += 1                                   │              │
│  └──────────────────┬───────────────────────────────┘              │
│                     │ research_findings[]                           │
│                     ▼                                               │
│  ┌──────────────────────────────────────────────────┐              │
│  │              CRITIC NODE                          │              │
│  │                                                   │              │
│  │  For each finding → score relevance + confidence  │              │
│  │  → set approved=True/False per finding            │              │
│  │  → set global approved flag                       │              │
│  └──────────────────┬───────────────────────────────┘              │
│                     │                                               │
│          ┌──────────┴───────────┐                                  │
│          │  route_after_critic  │                                   │
│          └──────────┬───────────┘                                   │
│      approved=False │ AND iteration < MAX_ITERATIONS                │
│      ───────────────┘                                               │
│           │                              approved=True              │
│           │                         OR   iteration >= MAX_ITER      │
│           ▼                              │                          │
│  [web_researcher] ◄──────────────────── ┘                          │
│  (refinement loop)                       │                          │
│                                          ▼                          │
│                                   [supervisor]                      │
│                                   (report assembly)                 │
│                                          │                          │
│                                         END                         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  │  ResearchResponse
                                  ▼
                              CLIENT
```

---

## 4. Component Breakdown

### 4.1 FastAPI Layer

The HTTP surface is intentionally thin — it initialises state and delegates
everything else to the graph engine.

```
POST /research
  │
  ├── Validates request body with Pydantic (ResearchRequest)
  ├── Constructs a blank AgentState (all fields zeroed/empty)
  ├── Calls compiled_graph.ainvoke(state)  ← async, non-blocking
  ├── Unwraps the terminal AgentState
  └── Returns ResearchResponse (query, final_report, plan, iterations, approved)

GET /health
  └── Returns { "status": "ok", "service": "AMIN" }
```

**Key design decisions:**

- `compiled_graph` is compiled once at **module load time** (`build_graph().compile()`),
  not per-request — avoids repeated graph-compilation overhead.
- The endpoint uses `await compiled_graph.ainvoke(...)` so the FastAPI event loop
  is never blocked during LLM calls. Each request is handled concurrently.
- HTTP 400 is raised for empty queries before touching the graph.
- HTTP 500 wraps any graph execution exception with a human-readable message.

---

### 4.2 LangGraph StateGraph Engine

LangGraph models the agent pipeline as a **directed graph** where:

- **Nodes** are Python functions that accept `AgentState` and return a
  (partial) updated `AgentState`.
- **Edges** define mandatory transitions between nodes.
- **Conditional edges** implement the refinement loop routing logic.

```
Graph topology
──────────────
START  ──►  supervisor
            supervisor  ──►  web_researcher      (fixed edge)
            web_researcher  ──►  critic           (fixed edge)
            critic  ──►  web_researcher           (conditional — needs revision)
            critic  ──►  supervisor               (conditional — approved / max iter)
            supervisor  ──►  END                  (fixed edge — second visit)
```

> [!NOTE]
> The `supervisor` node uses a two-pass pattern: the **first visit** (when
> `state["plan"] == []`) handles planning; the **second visit** (triggered
> by the Critic approving findings) handles final report assembly. This avoids
> needing a separate "planner" and "reporter" node, keeping the graph compact.

---

### 4.3 Google Gemini LLM Backend

All three agent nodes call `get_llm()` to obtain a `ChatGoogleGenerativeAI`
instance:

```python
ChatGoogleGenerativeAI(
    model        = MODEL_NAME,          # default: "gemini-3.8-flash"
    google_api_key = GOOGLE_API_KEY or None,
    temperature  = 0.2,                 # low — encourages deterministic output
    max_output_tokens = 4096,
)
```

**Why `temperature=0.2`?**  
Market intelligence reports benefit from factual, consistent output. A low
temperature reduces hallucination variance and makes structured-format parsing
(e.g., `RELEVANCE: 8`) more reliable.

**Why `gemini-3.8-flash` as default?**  
Flash offers a strong balance of speed, context length (1M tokens), and cost —
well-suited for multi-call pipelines where latency compounds across nodes.
Swap to `gemini-1.5-pro` via `MODEL_NAME` env var for deeper reasoning tasks.

---

## 5. Multi-Agent State Data Flow

### 5.1 Shared State Schema

All nodes read from and write to a single `AgentState` TypedDict. LangGraph
passes the full state object between nodes; each node returns only the fields
it mutates (merged via `{**state, ...}`).

```
┌─────────────────────────────────────────────────────────┐
│                      AgentState                         │
├─────────────────┬────────────────┬──────────────────────┤
│ Field           │ Type           │ Owner / Writer       │
├─────────────────┼────────────────┼──────────────────────┤
│ query           │ str            │ API layer (input)    │
│ context         │ dict           │ API layer (input)    │
│ plan            │ list[str]      │ supervisor (pass 1)  │
│ sub_tasks       │ list[str]      │ supervisor (pass 1)  │
│ research_findings│ list[dict]    │ web_researcher       │
│ critique        │ list[dict]     │ critic               │
│ iteration       │ int            │ web_researcher       │
│ approved        │ bool           │ critic               │
│ final_report    │ str            │ supervisor (pass 2)  │
└─────────────────┴────────────────┴──────────────────────┘
```

Each `research_findings` dict element has the shape:
```json
{
  "sub_task":  "Analyse EV battery supply chain risks",
  "summary":   "...",
  "sources":   ["bloomberg.com", "iea.org"]
}
```

Each `critique` dict element has the shape:
```json
{
  "sub_task":   "Analyse EV battery supply chain risks",
  "relevance":  8,
  "confidence": 7,
  "issues":     "None",
  "approved":   true
}
```

---

### 5.2 Agent Pipeline Walkthrough

Below is a step-by-step trace of a single `/research` request:

```
Step 0 ── API receives POST /research
          query = "Key growth drivers in the EV battery market for 2025"
          AgentState initialised with all fields empty / zero

Step 1 ── SUPERVISOR (planning pass)
          Input:  query, context
          Action: LLM call → numbered list of 3-5 sub-tasks
          Output: plan = sub_tasks = [
                    "Analyse global EV adoption trends",
                    "Research lithium and cobalt supply chain",
                    "Identify leading battery manufacturers",
                    "Review government incentive policies"
                  ]
                  iteration = 0

Step 2 ── WEB RESEARCHER (iteration 1)
          Input:  sub_tasks[], iteration=0
          Action: One LLM call per sub-task → structured SUMMARY + SOURCES
          Output: research_findings = [
                    { sub_task, summary, sources },   ← for sub_task[0]
                    { sub_task, summary, sources },   ← for sub_task[1]
                    ...
                  ]
                  iteration = 1

Step 3 ── CRITIC (validation pass)
          Input:  research_findings[], query
          Action: One LLM call per finding → RELEVANCE / CONFIDENCE / VERDICT
          Output: critique = [ { sub_task, relevance, confidence, issues, approved } ... ]
                  approved = True  (if ALL findings pass thresholds)
                          OR False (if ANY finding fails)

Step 4a ─ ROUTE (approved=False, iteration < MAX_ITERATIONS)
          → back to WEB RESEARCHER for refinement (repeat Steps 2-3)

Step 4b ─ ROUTE (approved=True OR iteration >= MAX_ITERATIONS)
          → forward to SUPERVISOR (report assembly pass)

Step 5 ── SUPERVISOR (report assembly pass)
          Input:  research_findings[], query
          Action: LLM call → full structured market intelligence report
          Output: final_report = "## Executive Summary\n..."

Step 6 ── END → FastAPI returns ResearchResponse to client
```

---

### 5.3 Refinement Loop Logic

The `route_after_critic` function implements the guard:

```python
def route_after_critic(state) -> Literal["web_researcher", "supervisor"]:
    if not state["approved"] and state["iteration"] < MAX_ITERATIONS:
        return "web_researcher"   # trigger another research pass
    return "supervisor"           # proceed to final report
```

**Worst-case LLM call budget per request:**

```
MAX_ITERATIONS = 3
sub_tasks      = N  (typically 3–5)

Supervisor planning:   1 call
Web researcher:        N calls × up to 3 iterations  =  3N calls
Critic:                N calls × up to 3 iterations  =  3N calls
Supervisor reporting:  1 call

Total (max):           1 + 3N + 3N + 1  =  6N + 2  calls

With N=4:              6(4) + 2  =  26 LLM calls (absolute worst case)
With N=4, approved 1st pass:      1 + 4 + 4 + 1  =  10 LLM calls (best case)
```

---

## 6. Token Usage Management

### Current Strategy (v0.1.0)

| Control | Setting | Effect |
|---|---|---|
| `max_output_tokens=4096` | Per `get_llm()` call | Hard cap on any single response |
| `temperature=0.2` | Per `get_llm()` call | Discourages verbose/rambling output |
| `MAX_ITERATIONS=3` | `.env` / env var | Caps total graph passes |
| Sub-task count | Supervisor prompt ("3-5") | Bounds total parallel LLM calls |

### Token Estimation Per Request

Using Gemini 2.0 Flash approximate rates:

| Phase | Est. Input Tokens | Est. Output Tokens |
|---|---|---|
| Supervisor (planning) | ~200 | ~150 |
| Web Researcher (per sub-task) | ~300 | ~500 |
| Critic (per finding) | ~600 | ~200 |
| Supervisor (report assembly) | ~3,000 | ~1,500 |
| **Total (N=4, 1 iteration)** | **~7,000** | **~5,300** |

> [!TIP]
> To reduce token consumption in development, set `MODEL_NAME=gemini-3.8-flash`
> (default) and `MAX_ITERATIONS=1` in your `.env`. For production accuracy,
> increase `MAX_ITERATIONS` to 3 and consider `gemini-1.5-pro` for the
> Supervisor's report assembly pass only.

### Recommended Improvements

1. **Prompt compression** — strip redundant whitespace and repeat context from
   Critic prompts; pass only the delta (issues list) back to the researcher
   rather than full findings.
2. **Streaming responses** — use `compiled_graph.astream()` to stream tokens
   back to the client as the Supervisor assembles the report, reducing
   perceived latency.
3. **Output caching** — cache identical `(query_hash, iteration)` tuples in
   Redis with a short TTL (e.g., 1 hour) to avoid reprocessing duplicate
   queries. See [Section 9](#9-future-scaling-recommendations).

---

## 7. API Contract

### `POST /research`

**Request**
```json
{
  "query":   "string — the market research question (required, non-empty)",
  "context": "object — optional metadata, e.g. { \"sector\": \"EV\", \"year\": 2025 }"
}
```

**Response `200 OK`**
```json
{
  "query":        "string — echoed input query",
  "final_report": "string — full markdown-formatted market intelligence report",
  "plan":         ["string", "..."] ,
  "iterations":   2,
  "approved":     true
}
```

**Error Responses**

| Code | Condition |
|---|---|
| `400 Bad Request` | Empty or whitespace-only `query` |
| `500 Internal Server Error` | LangGraph graph execution failure |

---

### `GET /health`

**Response `200 OK`**
```json
{ "status": "ok", "service": "AMIN" }
```

---

## 8. Environment Configuration

All runtime behaviour is controlled via environment variables (loaded from
`.env` by `python-dotenv` at startup):

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | *(empty)* | Google AI Studio / Vertex AI API key. If set in the shell environment (e.g., Antigravity workspace), `.env` entry can be omitted. |
| `MODEL_NAME` | `gemini-3.8-flash` | Gemini model identifier. Options: `gemini-3.8-flash`, `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-3.8-flash-thinking-exp` |
| `MAX_ITERATIONS` | `3` | Maximum Critic→Researcher refinement cycles before forcing final report assembly. |

> [!IMPORTANT]
> `GOOGLE_API_KEY` is resolved with `os.getenv("GOOGLE_API_KEY", "")`. When
> running inside the Antigravity workspace, the key is injected automatically
> into the environment — no `.env` entry needed.

---

## 9. Future Scaling Recommendations

### 9.1 Async Parallel Sub-Task Research

**Current:** The Web Researcher node executes one LLM call per sub-task
**sequentially** in a `for` loop.

**Recommended:** Use `asyncio.gather()` to fan out sub-task research calls in
parallel, reducing wall-clock time from `O(N × latency)` to `O(latency)`:

```python
import asyncio

async def web_researcher_node(state: AgentState) -> AgentState:
    llm = get_llm()

    async def research_one(sub_task: str) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=sub_task)]
        response = await llm.ainvoke(messages)
        ...
        return {"sub_task": sub_task, "summary": summary, "sources": sources}

    findings = await asyncio.gather(*[research_one(t) for t in state["sub_tasks"]])
    return {**state, "research_findings": list(findings), "iteration": state["iteration"] + 1}
```

---

### 9.2 Redis Query Result Cache

Cache the terminal `AgentState` keyed on a hash of the query to serve
repeated questions instantly without re-invoking the graph:

```
Client ──► FastAPI ──► Redis GET(hash(query))
                            │
                     hit ───┘ (return cached ResearchResponse)
                     miss ──► LangGraph Graph ──► Redis SET(hash(query), result, TTL=3600)
```

**Recommended stack:** `redis-py` + `fakeredis` for local development.

---

### 9.3 Message Queue for Long-Running Requests

For enterprise workloads where a single query may take 30–90 seconds, move
graph execution off the HTTP request path:

```
POST /research  ──►  FastAPI  ──►  Enqueue job (job_id)
                                   └──► Return 202 Accepted { "job_id": "..." }

Worker process ──►  Dequeue  ──►  compiled_graph.ainvoke()
                                   └──► Store result in DB keyed by job_id

GET /research/{job_id}  ──►  Poll for result (200 + body) or 202 (still running)
```

**Recommended stack:** Celery + Redis broker, or Google Cloud Tasks.

---

### 9.4 Persistent Checkpointing with LangGraph

LangGraph natively supports **checkpointers** that persist graph state to a
database between steps, enabling:
- **Human-in-the-loop** review at the Critic node before approving findings.
- **Graph replay** — resume a failed run from the last successful node.
- **Audit trails** — full state history per `thread_id`.

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("./amin_checkpoints.db")
compiled_graph = build_graph().compile(checkpointer=checkpointer)

# Pass a thread_id per request to enable state persistence
result = await compiled_graph.ainvoke(state, config={"configurable": {"thread_id": job_id}})
```

---

### 9.5 Real Web Search Tool Integration

Replace the LLM-simulated research in `web_researcher_node` with real-time
web data retrieval:

| Tool | Package | Best For |
|---|---|---|
| Tavily Search | `tavily-python` | General web search with LLM-optimised results |
| SerpAPI | `google-search-results` | Google Search, News, Scholar |
| Exa | `exa-py` | Semantic web search |
| SEC EDGAR | Direct HTTP (EDGAR REST API) | Financial filings |

Each tool call replaces one `llm.invoke(messages)` in the sub-task loop,
and real URLs replace the simulated source domains in `sources`.

---

### 9.6 Observability & Tracing

| Concern | Recommendation |
|---|---|
| LLM call tracing | LangSmith (`LANGCHAIN_TRACING_V2=true`) |
| API metrics | Prometheus + `prometheus-fastapi-instrumentator` |
| Distributed tracing | OpenTelemetry → Jaeger / Google Cloud Trace |
| Structured logging | `structlog` with JSON output for log aggregation |
| Alerting | Alert on `approved=False` rate > 20% (indicates prompt drift) |

---

### 9.7 Deployment Architecture (Production Target)

```
                        ┌─────────────────────────────┐
                        │      Load Balancer           │
                        │   (Cloud Run / GKE Ingress)  │
                        └──────────────┬──────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
     ┌────────▼───────┐      ┌─────────▼──────┐      ┌─────────▼──────┐
     │  AMIN Instance │      │  AMIN Instance │      │  AMIN Instance │
     │  (FastAPI +    │      │  (FastAPI +    │      │  (FastAPI +    │
     │   LangGraph)   │      │   LangGraph)   │      │   LangGraph)   │
     └────────┬───────┘      └─────────┬──────┘      └─────────┬──────┘
              │                        │                        │
              └────────────────────────┼────────────────────────┘
                                       │
                        ┌──────────────┼──────────────┐
                        │              │              │
               ┌────────▼────┐  ┌──────▼─────┐  ┌────▼────────┐
               │  Redis      │  │  Task Queue│  │  PostgreSQL │
               │  (cache +   │  │  (Celery / │  │  (checkpoint│
               │   sessions) │  │   Cloud    │  │   store +   │
               └─────────────┘  │   Tasks)   │  │   results)  │
                                └────────────┘  └─────────────┘
                                       │
                               ┌───────▼────────┐
                               │  Google Gemini │
                               │  API           │
                               └────────────────┘
```

---

## 10. Known Limitations (v0.1.0)

| Limitation | Impact | Mitigation Path |
|---|---|---|
| Sequential sub-task research | Latency scales linearly with sub-task count | See §9.1: `asyncio.gather()` |
| No real web search | Research is bounded by model training data | See §9.5: Tavily / SerpAPI |
| No result caching | Every identical query triggers a full graph run | See §9.2: Redis cache |
| No persistent checkpoints | Failed runs cannot be resumed | See §9.4: LangGraph checkpointer |
| Python 3.9 (EOL warning) | `google-ai-generativelanguage` won't receive 3.9 patches | Upgrade to Python 3.11+ |
| In-process graph execution | Request timeout risk for long pipelines | See §9.3: Message queue |
| No authentication | API is open to all callers | Add OAuth2 / API key middleware |
| No rate limiting | Unbounded concurrent requests to Gemini API | Add `slowapi` middleware |

---

*Last updated: 2026-10-06 | Maintained by the AMIN project team*
