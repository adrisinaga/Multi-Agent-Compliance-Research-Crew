"""Researcher agent — gathers EU compliance data using internal docs + web search."""

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from p4_multi_agent.llm import fast_llm
from p4_multi_agent.state import AgentState
from p4_multi_agent.tools.compliance_search import search_compliance_docs
from p4_multi_agent.tools.web_search import web_search

_RESEARCHER_PROMPT = """\
You are a specialist EU trade compliance researcher.

Your job: gather accurate information about EU import regulations, tariff rates,
HS codes, and rules of origin for the product described in the user's query.

Workflow:
1. Always call search_compliance_docs first — it contains pre-verified tariff data.
2. Then call web_search to check for recent regulatory updates or news.
3. Synthesise the findings into a clear, factual summary that the Analyst can use.

Output format (required):
  PRODUCT: <name and HS code>
  EU TARIFF RATE: <MFN rate> (EVFTA preferential rate if applicable: <rate>)
  RULES OF ORIGIN: <requirement>
  REGULATORY REQUIREMENTS: <bullet list>
  RECENT UPDATES: <web search findings>
  RAW DATA FOR ANALYST: <any numeric figures needed for calculations, labelled clearly>
"""

_agent = create_react_agent(
    model=fast_llm,
    tools=[search_compliance_docs, web_search],
    prompt=_RESEARCHER_PROMPT,
)


def researcher_node(state: AgentState) -> dict:
    print("\n[RESEARCHER] Starting research...")
    query = state.get("query", "")

    result = _agent.invoke({"messages": [HumanMessage(content=query)]})

    findings = result["messages"][-1].content
    print(f"[RESEARCHER] Done. Findings preview: {findings[:300]}...")

    return {
        "research_findings": findings,
        "messages": result["messages"],
    }
