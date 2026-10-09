# AGENTS.md — Autonomous Market Intelligence Network (AMIN)

This document defines the roles, responsibilities, and interaction contracts for
every agent node in the AMIN LangGraph workflow.

---

## 1. Supervisor Agent

**Node name:** `supervisor`

### Role
Orchestrates the overall market-intelligence pipeline. It receives the raw user
query, decomposes it into sub-tasks, routes work to specialist agents, and
assembles their outputs into a final, coherent report.

### Responsibilities
- Parse and clarify the incoming research query.
- Decide which agents to invoke and in what order (sequential or parallel).
- Aggregate intermediate results and detect gaps that require additional passes.
- Produce the final structured response returned to the FastAPI endpoint.

### Inputs
| Field | Type | Description |
|-------|------|-------------|
| `query` | `str` | Raw user query |
| `context` | `dict` | Optional metadata (sector, date range, etc.) |

### Outputs
| Field | Type | Description |
|-------|------|-------------|
| `plan` | `list[str]` | Ordered list of sub-tasks dispatched |
| `final_report` | `str` | Consolidated market intelligence report |

### Decision Logic
```
if critic flags issues  → route back to web_researcher with refined query
else                    → compile final_report and END
```

---

## 2. Web Research Agent

**Node name:** `web_researcher`

### Role
Autonomously gathers raw market data from the web (news, filings, analyst
summaries, pricing feeds) in response to sub-tasks assigned by the Supervisor.

### Responsibilities
- Formulate targeted search queries from the Supervisor's sub-task list.
- Retrieve and parse relevant web sources.
- Deduplicate and summarise raw findings into structured snippets.
- Attach source citations to every claim.

### Inputs
| Field | Type | Description |
|-------|------|-------------|
| `sub_tasks` | `list[str]` | Research sub-tasks from the Supervisor |
| `iteration` | `int` | Current refinement pass (starts at 0) |

### Outputs
| Field | Type | Description |
|-------|------|-------------|
| `research_findings` | `list[dict]` | `{summary, sources, sub_task}` per task |
| `iteration` | `int` | Passed through for downstream tracking |

### Constraints
- Maximum **3 refinement iterations** per query to prevent runaway loops.
- Each finding must include at least one source URL.

---

## 3. Critic / Validator Agent

**Node name:** `critic`

### Role
Independently evaluates the Web Research Agent's findings for factual
consistency, source quality, logical coherence, and relevance before the
Supervisor assembles the final report.

### Responsibilities
- Cross-check claims against cited sources.
- Score each finding on **relevance** (0–10) and **confidence** (0–10).
- Flag contradictions, missing evidence, or low-quality sources.
- Return a structured critique with an explicit `approved` boolean.

### Inputs
| Field | Type | Description |
|-------|------|-------------|
| `research_findings` | `list[dict]` | Output from the Web Research Agent |
| `query` | `str` | Original user query (for relevance grounding) |

### Outputs
| Field | Type | Description |
|-------|------|-------------|
| `critique` | `list[dict]` | `{sub_task, score, issues, approved}` per finding |
| `approved` | `bool` | `True` if all findings pass; `False` triggers refinement |

### Approval Criteria
A finding is **approved** when:
- Relevance score ≥ 7 **and** Confidence score ≥ 6
- At least one primary source (not a blog or social media post) is cited
- No direct contradictions exist between claims and sources

---

## Agent Interaction Flow

```
User Query
    │
    ▼
[Supervisor] ──── plan ────► [Web Researcher]
    ▲                               │
    │                        research_findings
    │                               │
    │                               ▼
    │                          [Critic]
    │                               │
    │          approved=False       │
    └──────────────────────────────-┤
                                    │ approved=True
                                    ▼
                           [Supervisor] ──► Final Report ──► User
```

---

## Shared State Schema

All agents read from and write to a shared `AgentState` TypedDict:

```python
class AgentState(TypedDict):
    query:              str
    context:            dict
    plan:               list[str]
    sub_tasks:          list[str]
    research_findings:  list[dict]
    critique:           list[dict]
    iteration:          int
    approved:           bool
    final_report:       str
```
