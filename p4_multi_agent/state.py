import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    query: str
    research_findings: str
    analysis_results: str
    final_report: str
    next: str          # "Researcher" | "Analyst" | "Writer" | "FINISH"
    iteration_count: int
