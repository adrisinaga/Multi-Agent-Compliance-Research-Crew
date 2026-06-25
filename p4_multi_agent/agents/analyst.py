"""Analyst agent — runs calculations and distils key compliance points."""

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from p4_multi_agent.llm import fast_llm
from p4_multi_agent.state import AgentState
from p4_multi_agent.tools.analysis_tools import calculate_tariff_impact, summarize_research

_ANALYST_PROMPT = """\
You are a trade compliance financial analyst.

Your job: take the Researcher's findings and:
1. Call calculate_tariff_impact with the numeric values found in the research
   (base_price, tariff_rate, volume). Extract these from the query context if
   they are not in the research findings.
2. Call summarize_research to extract the key compliance bullet points.
3. Return a structured analysis that the Writer can use directly.

Important: You MUST call both tools. Do not skip either. If no volume/price is
given in the research, use reasonable defaults (volume=1000, base_price=10.0)
and note that you used defaults.

Output format (required):
  FINANCIAL ANALYSIS:
  <output of calculate_tariff_impact>

  KEY COMPLIANCE POINTS:
  <output of summarize_research>

  ANALYST NOTES:
  <any caveats, assumptions, or additional observations>
"""

_agent = create_react_agent(
    model=fast_llm,
    tools=[calculate_tariff_impact, summarize_research],
    prompt=_ANALYST_PROMPT,
)


def analyst_node(state: AgentState) -> dict:
    print("\n[ANALYST] Starting analysis...")
    research = state.get("research_findings", "")
    query = state.get("query", "")

    message = (
        f"Original query: {query}\n\n"
        f"Research findings:\n{research}\n\n"
        "Please run the financial analysis and extract key compliance points."
    )

    result = _agent.invoke({"messages": [HumanMessage(content=message)]})

    analysis = result["messages"][-1].content
    print(f"[ANALYST] Done. Analysis preview: {analysis[:300]}...")

    return {
        "analysis_results": analysis,
        "messages": result["messages"],
    }
