"""Web search tool — wraps TavilySearchResults with a graceful fallback stub
when TAVILY_API_KEY is absent (useful for offline testing).
"""

import os

from langchain_core.tools import tool


def _build_tool():
    if os.getenv("TAVILY_API_KEY"):
        from langchain_community.tools.tavily_search import TavilySearchResults
        return TavilySearchResults(max_results=3, name="web_search")
    return None


_tavily = _build_tool()


@tool
def web_search(query: str) -> str:
    """Search the live web for recent news, regulatory updates, and market data
    related to EU trade policy and compliance requirements.

    Use this after search_compliance_docs when you need current information
    that may not be in the internal database (e.g. recent policy changes,
    news articles, official EU announcements).

    Args:
        query: Search query string.
    """
    if _tavily is None:
        return (
            f"[WEB SEARCH STUB — TAVILY_API_KEY not set]\n"
            f"Query: '{query}'\n"
            "Simulated result: EU-Vietnam Free Trade Agreement (EVFTA) entered into "
            "force August 2020. As of 2024, approximately 99% of EU goods and 65% of "
            "Vietnamese goods are duty-free. Textile sector benefits from preferential "
            "rates under EVFTA with double-transformation rules of origin requirement. "
            "Source: European Commission Trade Policy (stub data)."
        )
    results = _tavily.invoke(query)
    if isinstance(results, list):
        formatted = "\n\n".join(
            f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\n{r.get('content', '')}"
            for r in results
        )
        return formatted
    return str(results)
