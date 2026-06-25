"""Mock P2 RAG retriever — returns EU tariff data from an in-memory dict.

In a real P2 implementation this would query a vector DB (e.g. ChromaDB or
FAISS). The interface is identical so swapping it out requires no changes to
the agent code.
"""

import json

from langchain_core.tools import tool

_MOCK_DB: dict[str, dict] = {
    "textile": {
        "hs_code": "5208",
        "product_description": "Woven fabrics of cotton",
        "eu_tariff_rate_pct": 12.0,
        "preferential_rate_vietnam_pct": 9.6,  # EVFTA GSP rate
        "rules_of_origin": "Double transformation required",
        "regulatory_notes": (
            "Subject to EU textile labelling Regulation (EU) No 1007/2011. "
            "REACH compliance required for chemical finishes. "
            "EVFTA preferential rate applies with valid EUR.1 certificate."
        ),
        "source": "EU TARIC database (mock), EVFTA Annex 2-A",
    },
    "electronics": {
        "hs_code": "8471",
        "product_description": "Automatic data processing machines",
        "eu_tariff_rate_pct": 0.0,
        "preferential_rate_vietnam_pct": 0.0,
        "rules_of_origin": "CTH (Change of Tariff Heading)",
        "regulatory_notes": (
            "Zero-duty under ITA (Information Technology Agreement). "
            "CE marking mandatory. RoHS and WEEE directives apply."
        ),
        "source": "EU TARIC database (mock), ITA",
    },
    "footwear": {
        "hs_code": "6404",
        "product_description": "Footwear with rubber or plastics outer soles",
        "eu_tariff_rate_pct": 17.0,
        "preferential_rate_vietnam_pct": 0.0,
        "rules_of_origin": "Production from materials of any heading",
        "regulatory_notes": (
            "High-duty category. EVFTA reduces to 0% over 7-year schedule ending 2027. "
            "Current EVFTA rate (2024): ~7.5%. "
            "REACH and General Product Safety Regulation apply."
        ),
        "source": "EU TARIC database (mock), EVFTA Annex 2-A",
    },
    "pharmaceutical": {
        "hs_code": "3004",
        "product_description": "Medicaments for retail sale",
        "eu_tariff_rate_pct": 0.0,
        "preferential_rate_vietnam_pct": 0.0,
        "rules_of_origin": "Specific process rule",
        "regulatory_notes": (
            "Zero MFN duty. EMA marketing authorisation required. "
            "GDP (Good Distribution Practice) guidelines mandatory for import."
        ),
        "source": "EU TARIC database (mock)",
    },
    "furniture": {
        "hs_code": "9403",
        "product_description": "Other furniture and parts thereof",
        "eu_tariff_rate_pct": 2.7,
        "preferential_rate_vietnam_pct": 0.0,
        "rules_of_origin": "Production from materials of any heading",
        "regulatory_notes": (
            "EVFTA eliminates duty immediately. "
            "EUTR (EU Timber Regulation) due diligence if wood content >10%. "
            "Flammability standards EN 597 apply to upholstered items."
        ),
        "source": "EU TARIC database (mock), EVFTA Annex 2-A",
    },
}

_KEYWORDS = list(_MOCK_DB.keys())


def _match(query: str) -> list[dict]:
    q = query.lower()
    results = []
    for keyword, data in _MOCK_DB.items():
        if keyword in q or data["hs_code"] in q:
            results.append({"matched_keyword": keyword, **data})
    if not results:
        # return all entries as a fallback so the agent still gets data
        results = [{"matched_keyword": k, **v} for k, v in _MOCK_DB.items()]
    return results


@tool
def search_compliance_docs(query: str) -> str:
    """Search internal EU compliance documentation for tariff rates, HS codes,
    rules of origin, and regulatory requirements.

    Returns JSON-formatted records from the compliance database. Use this tool
    first before doing a web search — it contains pre-verified regulatory data.

    Args:
        query: Natural language query, product name, or HS code.
    """
    matches = _match(query)
    return json.dumps(matches, indent=2)
