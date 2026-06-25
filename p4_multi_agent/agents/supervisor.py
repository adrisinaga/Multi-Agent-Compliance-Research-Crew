"""Supervisor agent — routes between Researcher, Analyst, Writer, and FINISH.

Uses structured output (Pydantic RouteDecision) instead of free-text parsing
so routing is deterministic and never breaks on unexpected model phrasing.
"""

from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from p4_multi_agent.llm import fast_llm
from p4_multi_agent.state import AgentState

_SUPERVISOR_SYSTEM = """\
You are a supervisor coordinating a compliance research crew.

Team members:
  - Researcher: searches internal compliance docs and the web to gather data
  - Analyst:    runs financial calculations and summarises key findings
  - Writer:     composes the final structured executive report

Routing rules (follow strictly):
  1. If research_findings is empty -> route to Researcher.
  2. If research_findings is present but analysis_results is empty -> route to Analyst.
  3. If both research_findings and analysis_results are present but final_report is
     empty -> route to Writer.
  4. If final_report is present -> return FINISH.
  5. If iteration_count > 8 and final_report is empty -> route to Writer anyway
     (prevent looping).

Always explain your routing decision in the 'reason' field.
"""


class RouteDecision(BaseModel):
    next: Literal["Researcher", "Analyst", "Writer", "FINISH"] = Field(
        description="Which team member to call next, or FINISH when done."
    )
    reason: str = Field(description="One-sentence explanation of the routing decision.")


_router = fast_llm.with_structured_output(RouteDecision)


def supervisor_node(state: AgentState) -> dict:
    has_research = bool(state.get("research_findings", "").strip())
    has_analysis = bool(state.get("analysis_results", "").strip())
    has_report = bool(state.get("final_report", "").strip())
    iteration = state.get("iteration_count", 0)

    prompt = (
        f"{_SUPERVISOR_SYSTEM}\n\n"
        f"Current state:\n"
        f"  research_findings present: {has_research}\n"
        f"  analysis_results present:  {has_analysis}\n"
        f"  final_report present:      {has_report}\n"
        f"  iteration_count:           {iteration}\n\n"
        f"Original query: {state.get('query', '')}\n\n"
        "What should the team do next?"
    )

    decision: RouteDecision = _router.invoke([HumanMessage(content=prompt)])

    print(f"\n[SUPERVISOR] iter={iteration + 1} -> {decision.next} | {decision.reason}")

    return {
        "next": decision.next,
        "iteration_count": iteration + 1,
    }
