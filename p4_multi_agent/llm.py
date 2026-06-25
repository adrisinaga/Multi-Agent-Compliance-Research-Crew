"""Central LLM configuration — swap provider here, not in every agent.

Uses SumoPod's OpenAI-compatible endpoint via langchain-openai.
  - fast_llm:   cheap model for Supervisor, Researcher, Analyst (high call volume)
  - writer_llm: better model for Writer (user-facing output)
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

_base_url = os.getenv("SUMOPOD_BASE_URL", "https://ai.sumopod.com/v1")
_api_key = os.getenv("SUMOPOD_API_KEY", "")

if not _api_key:
    raise EnvironmentError(
        "SUMOPOD_API_KEY is not set. "
        "Add it to your .env file or set the environment variable."
    )

fast_llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url=_base_url,
    api_key=_api_key,
    temperature=0,
)

writer_llm = ChatOpenAI(
    model="gpt-4o-mini",   # bump to "gpt-4o" if available on your plan
    base_url=_base_url,
    api_key=_api_key,
    temperature=0.3,
)
