"""Pure-Python analysis tools for the Analyst agent.

No LLM calls here — deterministic calculations so the Analyst's output is
verifiable and reproducible.
"""

import re
from langchain_core.tools import tool


@tool
def calculate_tariff_impact(
    base_price: float,
    tariff_rate: float,
    volume: int,
    currency: str = "EUR",
) -> str:
    """Calculate the financial impact of an EU import tariff.

    Computes per-unit duty, total duty cost, and fully-landed cost.

    Args:
        base_price: CIF price per unit (Cost, Insurance, Freight) in the given currency.
        tariff_rate: Tariff rate as a percentage (e.g. 12.0 for 12%).
        volume: Number of units being imported.
        currency: Currency symbol or code (default: EUR).
    """
    duty_per_unit = base_price * (tariff_rate / 100)
    landed_cost_per_unit = base_price + duty_per_unit
    total_duty = duty_per_unit * volume
    total_landed_cost = landed_cost_per_unit * volume

    return (
        f"Tariff Impact Analysis\n"
        f"{'─' * 35}\n"
        f"Base price per unit:      {currency} {base_price:.2f}\n"
        f"Tariff rate:              {tariff_rate:.1f}%\n"
        f"Duty per unit:            {currency} {duty_per_unit:.2f}\n"
        f"Landed cost per unit:     {currency} {landed_cost_per_unit:.2f}\n"
        f"Volume:                   {volume:,} units\n"
        f"Total duty cost:          {currency} {total_duty:,.2f}\n"
        f"Total landed cost:        {currency} {total_landed_cost:,.2f}\n"
        f"Duty as % of total cost:  {(total_duty / total_landed_cost * 100):.1f}%"
    )


@tool
def summarize_research(research_text: str, max_points: int = 5) -> str:
    """Extract the most important facts from raw research text as bullet points.

    This is a deterministic extraction tool — it pulls sentences containing
    key compliance signal words (tariff, rate, duty, regulation, requirement,
    certificate, compliance, HS code) and returns them as structured bullets.

    Args:
        research_text: Raw text from the Researcher agent.
        max_points: Maximum number of bullet points to return (default: 5).
    """
    signal_words = [
        "tariff", "rate", "duty", "regulation", "requirement",
        "certificate", "compliance", "hs code", "evfta", "origin",
        "prohibited", "restricted", "mandatory", "applies",
    ]

    sentences = re.split(r"(?<=[.!?])\s+", research_text.strip())
    scored: list[tuple[int, str]] = []
    for sent in sentences:
        score = sum(1 for w in signal_words if w in sent.lower())
        if score > 0:
            scored.append((score, sent.strip()))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_points]

    if not top:
        return "No specific compliance signals found in research text. Review raw findings."

    bullets = "\n".join(f"• {sent}" for _, sent in top)
    return f"Key Compliance Points ({len(top)} of {len(scored)} extracted):\n{bullets}"
