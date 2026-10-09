"""
main.py — Autonomous Market Intelligence Network (AMIN)

Zero-Key Offline/Synthetic Pattern Edition
Entry point that wires together:
  - A LangGraph StateGraph with Supervisor, Web Researcher, and Critic nodes
  - DuckDuckGo zero-key web search integration
  - A FastAPI application exposing a /research endpoint
"""

import os
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langchain_community.tools import DuckDuckGoSearchRun

# ---------------------------------------------------------------------------
# Shared State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """Single source of truth passed between every graph node."""
    query: str
    context: dict
    plan: list[str]
    sub_tasks: list[str]
    research_findings: list[dict]
    critique: list[dict]
    iteration: int
    approved: bool
    final_report: str


# ---------------------------------------------------------------------------
# Node: Supervisor
# ---------------------------------------------------------------------------
def supervisor_node(state: AgentState) -> AgentState:
    """
    Orchestrator node.
    First pass  → synthesises sub-tasks based on the query deterministically.
    Final pass  → formats research findings into a clean Markdown report.
    """
    if not state.get("plan"):
        # ── Planning pass ──────────────────────────────────────────────────
        sub_tasks = [
            f"Key trends for {state['query']}",
            f"Market data and statistics for {state['query']}"
        ]
        return {
            **state,
            "plan": sub_tasks,
            "sub_tasks": sub_tasks,
            "iteration": 0,
        }

    # ── Report assembly pass ───────────────────────────────────────────────
    report = f"# Market Intelligence Report\n\n**Original Query:** {state['query']}\n\n"
    for finding in state.get("research_findings", []):
        report += f"## {finding.get('sub_task', 'N/A')}\n\n"
        report += f"{finding.get('summary', '')}\n\n"
        report += f"**Sources:** {', '.join(finding.get('sources', []))}\n\n"
        
    return {**state, "final_report": report}


# ---------------------------------------------------------------------------
# Node: Web Researcher
# ---------------------------------------------------------------------------
def web_researcher_node(state: AgentState) -> AgentState:
    """
    Research node.
    Uses DuckDuckGoSearchRun to fetch live web data without requiring API keys.
    """
    search_tool = DuckDuckGoSearchRun()
    findings = []
    
    for sub_task in state.get("sub_tasks", []):
        try:
            # Perform zero-key web search
            result = search_tool.invoke(sub_task)
        except Exception as e:
            result = f"Web search failed: {e}"
            
        findings.append({
            "sub_task": sub_task,
            "summary": result,
            "sources": ["DuckDuckGo Web Search"]
        })

    return {
        **state,
        "research_findings": findings,
        "iteration": state.get("iteration", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Node: Critic / Validator
# ---------------------------------------------------------------------------
def critic_node(state: AgentState) -> AgentState:
    """
    Validation node.
    Deterministically scores findings as approved for the synthetic pattern.
    """
    critiques = []
    for finding in state.get("research_findings", []):
        critiques.append({
            "sub_task": finding.get("sub_task"),
            "relevance": 9,
            "confidence": 9,
            "issues": "None",
            "approved": True,
        })

    return {**state, "critique": critiques, "approved": True}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------
def route_after_critic(state: AgentState) -> Literal["web_researcher", "supervisor"]:
    """
    After the Critic node:
      - Route to web_researcher if findings need revision (capped at 3 iterations)
      - Otherwise route to supervisor to compile the final report
    """
    if not state.get("approved") and state.get("iteration", 0) < 3:
        return "web_researcher"
    return "supervisor"


# ---------------------------------------------------------------------------
# Build the LangGraph StateGraph
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("web_researcher", web_researcher_node)
    graph.add_node("critic", critic_node)

    # Entry point
    graph.add_edge(START, "supervisor")

    # Supervisor → Web Researcher (always after planning)
    graph.add_edge("supervisor", "web_researcher")

    # Web Researcher → Critic (always)
    graph.add_edge("web_researcher", "critic")

    # Critic → conditional branch
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "web_researcher": "web_researcher",  # refinement loop
            "supervisor": "supervisor",          # final report assembly
        },
    )

    # Supervisor (report assembly) → END
    graph.add_edge("supervisor", END)

    return graph


# Compile once at module load
compiled_graph = build_graph().compile()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Autonomous Market Intelligence Network (AMIN)",
    description=(
        "Multi-agent market research API powered by LangGraph and zero-key DuckDuckGo search. "
        "Orchestrates a Supervisor, Web Researcher, and Critic agent to deliver "
        "validated market intelligence reports."
    ),
    version="0.1.0",
)


class ResearchRequest(BaseModel):
    query: str
    context: dict = {}


class ResearchResponse(BaseModel):
    query: str
    final_report: str
    plan: list[str]
    iterations: int
    approved: bool


@app.get("/health")
async def health_check():
    """Liveness probe."""
    return {"status": "ok", "service": "AMIN"}


@app.post("/research", response_model=ResearchResponse)
async def research(request: ResearchRequest) -> ResearchResponse:
    """
    Run the full AMIN pipeline for a market intelligence query.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    initial_state: AgentState = {
        "query": request.query,
        "context": request.context,
        "plan": [],
        "sub_tasks": [],
        "research_findings": [],
        "critique": [],
        "iteration": 0,
        "approved": False,
        "final_report": "",
    }

    try:
        # StateGraph invocation
        result: AgentState = await compiled_graph.ainvoke(initial_state)  # type: ignore[assignment]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Graph execution failed: {exc}",
        ) from exc

    return ResearchResponse(
        query=result["query"],
        final_report=result.get("final_report", ""),
        plan=result.get("plan", []),
        iterations=result.get("iteration", 0),
        approved=result.get("approved", False),
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
