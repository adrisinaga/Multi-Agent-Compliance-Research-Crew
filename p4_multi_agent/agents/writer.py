"""Writer agent — composes the final structured executive report."""

from langchain_core.messages import HumanMessage

from p4_multi_agent.llm import writer_llm
from p4_multi_agent.state import AgentState

_WRITER_SYSTEM = """\
You are a senior compliance report writer producing executive summaries for
business stakeholders.

Write a professional report using EXACTLY this structure — no deviations:

## Introduction
[2 paragraphs: context of the query, why this matters to the business]

## Key Findings
[5 bullet points covering tariff rates, financial impact, regulatory requirements,
rules of origin, and any time-sensitive considerations]

## Recommendation
[3-4 concrete action items the business should take, numbered list]

Tone: professional, direct, no jargon. Audience: non-specialist business executives.
Length: 350-500 words total.
"""

def writer_node(state: AgentState) -> dict:
    print("\n[WRITER] Composing final report...")
    query = state.get("query", "")
    research = state.get("research_findings", "")
    analysis = state.get("analysis_results", "")

    message = (
        f"{_WRITER_SYSTEM}\n\n"
        f"Original question: {query}\n\n"
        f"Research findings:\n{research}\n\n"
        f"Financial analysis:\n{analysis}\n\n"
        "Write the executive report now."
    )

    response = writer_llm.invoke([HumanMessage(content=message)])
    report = response.content

    print(f"[WRITER] Done. Report length: {len(report)} chars.")

    return {
        "final_report": report,
        "messages": [response],
    }
