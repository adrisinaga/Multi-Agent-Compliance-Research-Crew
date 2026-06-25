"""Entry point for the Multi-Agent Compliance Research Crew (Project 4).

Run with:
    python -m p4_multi_agent.main

Requires ANTHROPIC_API_KEY in environment (or .env file).
TAVILY_API_KEY is optional — falls back to stub if absent.
"""

import sys
import textwrap
import time

# Force UTF-8 output on Windows so model responses with Unicode don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

from p4_multi_agent.graph import graph  # noqa: E402 (after load_dotenv)
from p4_multi_agent.state import AgentState  # noqa: E402

SAMPLE_QUERY = (
    "Research the EU tariff implications for importing textile products "
    "from Vietnam in 2024. Calculate the cost impact for 10,000 units at "
    "€5 base price per unit. Write a concise executive summary with findings "
    "and recommendations."
)


def run(query: str = SAMPLE_QUERY) -> str:
    print("=" * 60)
    print("Multi-Agent Compliance Research Crew - Project 4")
    print("=" * 60)
    print(f"\nQuery: {textwrap.fill(query, 58)}\n")
    print("-" * 60)

    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "research_findings": "",
        "analysis_results": "",
        "final_report": "",
        "next": "",
        "iteration_count": 0,
    }

    start = time.perf_counter()
    result = graph.invoke(initial_state, config={"recursion_limit": 15})
    elapsed = time.perf_counter() - start

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(result["final_report"])
    print("\n" + "-" * 60)
    print(f"Total wall time: {elapsed:.1f}s | Iterations: {result['iteration_count']}")

    return result["final_report"]


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else SAMPLE_QUERY
    run(query)
