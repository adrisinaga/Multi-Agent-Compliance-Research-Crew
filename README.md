# Project 4 — Multi-Agent Compliance Research Crew

A LangGraph-based multi-agent system for answering complex EU import tariff
questions. Queries like *"research the implications of new EU tariffs for
product X and write an executive summary"* are handled by three specialist
agents coordinated by a single Supervisor.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Shared State](#shared-state)
3. [LLM Config](#llm-config)
4. [Tools](#tools)
5. [Each Agent](#each-agent)
6. [Graph & Routing](#graph--routing)
7. [End-to-End Flow](#end-to-end-flow)
8. [How to Run](#how-to-run)
9. [Real Weaknesses & Trade-offs](#real-weaknesses--trade-offs)

---

## Architecture

```
User Query
    |
    v
+-------------------+
|    graph.invoke() |  <-- entry point in main.py
+-------------------+
    |
    v
+-------------+        RouteDecision (Pydantic)
| SUPERVISOR  | ----> Researcher | Analyst | Writer | FINISH
+-------------+
    ^    |
    |    +-----------> [Researcher] -> returns to Supervisor
    |    +-----------> [Analyst]    -> returns to Supervisor
    |    +-----------> [Writer]     -> returns to Supervisor
    |
    +--- FINISH -> END (graph stops)
```

All nodes share a single `AgentState` object. Each agent writes to its own
fields; the Supervisor reads those fields to decide who to call next.

---

## Shared State

**File:** `p4_multi_agent/state.py`

```python
class AgentState(TypedDict):
    messages:          Annotated[Sequence[BaseMessage], operator.add]
    query:             str   # original user query, unchanged throughout the workflow
    research_findings: str   # written by Researcher
    analysis_results:  str   # written by Analyst
    final_report:      str   # written by Writer
    next:              str   # written by Supervisor: "Researcher"|"Analyst"|"Writer"|"FINISH"
    iteration_count:   int   # incremented by 1 each time Supervisor is called
```

### Why `operator.add` for `messages`?

The `messages` field uses `Annotated[..., operator.add]` — meaning every node
that returns `messages` will **append** to the existing list rather than
overwrite it. This is the LangGraph v1.x pattern for message accumulation.

The other fields (`research_findings`, `analysis_results`, `final_report`) are
plain strings — a node filling them simply does `return {"research_findings": "..."}`,
and LangGraph overwrites the previous value.

---

## LLM Config

**File:** `p4_multi_agent/llm.py`

All agents import their LLM from a single place. To swap providers, edit
this file only — no agent code needs to change.

```python
fast_llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0)    # Supervisor, Researcher, Analyst
writer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)  # Writer
```

Both point to the SumoPod endpoint (`https://ai.sumopod.com/v1`), which is
compatible with the OpenAI SDK. The API key is read from `.env`:

```
SUMOPOD_API_KEY=sk-...
SUMOPOD_BASE_URL=https://ai.sumopod.com/v1
```

`temperature=0` for agents that need deterministic output (routing,
calculations). `temperature=0.3` for the Writer so the report reads
naturally rather than robotically.

---

## Tools

Tools are plain Python functions decorated with `@tool`. The LLM does not
execute them directly — it requests a specific tool with specific arguments,
then LangGraph executes it and returns the result back to the LLM.

### `search_compliance_docs`
**File:** `p4_multi_agent/tools/compliance_search.py`

A simulated RAG (Retrieval-Augmented Generation) from Project 2. In a real
implementation this would query a vector database (ChromaDB/FAISS). Here it
uses an in-memory dict with 5 product categories:

| Keyword        | HS Code | MFN Rate | EVFTA Rate |
|----------------|---------|----------|------------|
| textile        | 5208    | 12.0%    | 9.6%       |
| electronics    | 8471    | 0.0%     | 0.0%       |
| footwear       | 6404    | 17.0%    | ~7.5%      |
| pharmaceutical | 3004    | 0.0%     | 0.0%       |
| furniture      | 9403    | 2.7%     | 0.0%       |

Input: query string (product name or HS code).
Output: JSON string containing an array of matching records.

If no keyword matches, it returns **all** records as a fallback so the
agent still has data to work with.

---

### `web_search`
**File:** `p4_multi_agent/tools/web_search.py`

Wrapper for the Tavily Search API. Returns the top 3 results from the web.

If `TAVILY_API_KEY` is not set, the tool returns a **stub string** with
simulated data — the system keeps running without crashing.

```python
if _tavily is None:
    return "[WEB SEARCH STUB] ..."  # simulated data
```

---

### `calculate_tariff_impact`
**File:** `p4_multi_agent/tools/analysis_tools.py`

Pure Python calculation — no LLM calls. Deterministic and manually verifiable.

```
Input:  base_price=5.0, tariff_rate=12.0, volume=10000, currency="EUR"
Output:
  Base price per unit:      EUR 5.00
  Tariff rate:              12.0%
  Duty per unit:            EUR 0.60
  Landed cost per unit:     EUR 5.60
  Volume:                   10,000 units
  Total duty cost:          EUR 6,000.00
  Total landed cost:        EUR 56,000.00
  Duty as % of total cost:  10.7%
```

---

### `summarize_research`
**File:** `p4_multi_agent/tools/analysis_tools.py`

Deterministic extraction — not an LLM. Splits research text into sentences,
scores each one by the presence of compliance signal words (`tariff`, `rate`,
`duty`, `regulation`, `certificate`, etc.), then returns the top N sentences
as bullet points.

---

## Each Agent

### Supervisor
**File:** `p4_multi_agent/agents/supervisor.py`

**Model:** `fast_llm` (gpt-4o-mini, temperature=0)

**Role:** reads state, decides who gets called next.

**How it works:**

1. Reads 3 booleans from state: `has_research`, `has_analysis`, `has_report`
2. Builds a prompt containing the system prompt + status of those three fields
3. Sends it to the LLM via `.with_structured_output(RouteDecision)`
4. The LLM returns a `RouteDecision` object with `next` and `reason` fields
5. Returns `{"next": "...", "iteration_count": iteration + 1}`

**Why `with_structured_output`?**

The alternative is free-text parsing like `if "Researcher" in response`.
That is fragile — if the model returns "go to Researcher" instead of
"Researcher", parsing fails and the supervisor loops. With a Pydantic schema,
the LLM is forced to return one of 4 valid values (`Literal` type).

**Routing rules (priority order):**
```
research_findings empty            -> Researcher
analysis_results empty             -> Analyst
final_report empty                 -> Writer
final_report present               -> FINISH
iteration_count > 8 & not FINISHED -> Writer (emergency anti-loop)
```

**What the Supervisor does NOT do:** it does not read the *contents* of
`research_findings` or `analysis_results` — it only knows whether those
fields are empty or not. This is a deliberate trade-off: routing stays
simple and cheap, but the Supervisor cannot detect that the research was poor.

---

### Researcher
**File:** `p4_multi_agent/agents/researcher.py`

**Model:** `fast_llm` (gpt-4o-mini, temperature=0)
**Tools:** `search_compliance_docs`, `web_search`
**Pattern:** `create_react_agent` (ReAct loop)

**How it works:**

`create_react_agent` creates a ReAct sub-graph: model thinks → picks a tool →
tool executes → sees result → thinks again → until the model decides it is done.

```
[Researcher receives query]
    |
    v
Thought: "I need to find tariff data for textiles from Vietnam"
    |
    v
Action: search_compliance_docs("textile Vietnam tariff")
    |
    v
Observation: {"hs_code": "5208", "eu_tariff_rate_pct": 12.0, ...}
    |
    v
Thought: "I have base data. I need to check for recent web updates"
    |
    v
Action: web_search("EU textile tariff Vietnam 2024 update")
    |
    v
Observation: "[web results / stub]"
    |
    v
Final Answer: "PRODUCT: Woven fabrics... EU TARIFF RATE: 12.0%..."
```

**Writes to state:** `research_findings` (formatted string)

**Reads from state:** `query` only

---

### Analyst
**File:** `p4_multi_agent/agents/analyst.py`

**Model:** `fast_llm` (gpt-4o-mini, temperature=0)
**Tools:** `calculate_tariff_impact`, `summarize_research`
**Pattern:** `create_react_agent` (ReAct loop)

**How it works:**

Receives `research_findings` + `query` from state (combined into a single
HumanMessage). The ReAct loop runs:

```
[Analyst receives research + query]
    |
    v
Thought: "I see tariff_rate=12.0%, base_price=5 EUR, volume=10000"
    |
    v
Action: calculate_tariff_impact(base_price=5.0, tariff_rate=12.0, volume=10000)
    |
    v
Observation: "Total duty: EUR 6,000.00 | Landed cost: EUR 56,000.00"
    |
    v
Action: summarize_research(research_text="PRODUCT: Woven fabrics...")
    |
    v
Observation: "• EVFTA preferential rate applies with EUR.1 certificate..."
    |
    v
Final Answer: "FINANCIAL ANALYSIS: ... KEY COMPLIANCE POINTS: ..."
```

**Important:** The Analyst prompt requires both tools to be called. If no
explicit numbers are in the research, the Analyst uses defaults
(volume=1000, base_price=10.0) and notes that it did so.

**Writes to state:** `analysis_results`

**Reads from state:** `research_findings`, `query`

---

### Writer
**File:** `p4_multi_agent/agents/writer.py`

**Model:** `writer_llm` (gpt-4o-mini, temperature=0.3)
**Tools:** none
**Pattern:** direct `llm.invoke()` — not ReAct

**Why not `create_react_agent`?**

The Writer has no tools — its job is pure text synthesis. Adding ReAct
overhead (an extra LLM call to "pick a tool") provides no value. Direct
invoke is faster and cheaper.

**How it works:**

Builds a single large prompt containing the system prompt + query + research +
analysis, then sends it to the LLM in one call:

```
[Writer receives research + analysis + query]
    |
    v
LLM invoke (single call, no loop)
    |
    v
Output: structured report with 3 required sections:
  ## Introduction     <- 2 paragraphs of context
  ## Key Findings     <- 5 bullet points
  ## Recommendation   <- 3-4 numbered action items
```

**Writes to state:** `final_report`

**Reads from state:** `research_findings`, `analysis_results`, `query`

---

## Graph & Routing

**File:** `p4_multi_agent/graph.py`

```python
workflow = StateGraph(AgentState)

# Register 4 nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Analyst",    analyst_node)
workflow.add_node("Writer",     writer_node)

# Entry point is always the Supervisor
workflow.set_entry_point("supervisor")

# Conditional edges from Supervisor based on state["next"]
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "Researcher": "Researcher",
        "Analyst":    "Analyst",
        "Writer":     "Writer",
        "FINISH":     END,         # graph stops
    }
)

# All sub-agents return to Supervisor after finishing
workflow.add_edge("Researcher", "supervisor")
workflow.add_edge("Analyst",    "supervisor")
workflow.add_edge("Writer",     "supervisor")
```

**Recursion limit:** `graph.invoke(state, config={"recursion_limit": 15})`
LangGraph raises `RecursionError` after 15 steps. This is the last safety
net — the Supervisor has its own anti-loop mechanism at `iteration_count > 8`.

---

## End-to-End Flow

Real query from the last run:
> *"Research EU new tariff implications for electronics products imported
> from China in 2024, calculate cost impact for 5000 units at 50 EUR base
> price, and write an executive summary"*

```
Step 1: graph.invoke() is called
        Initial state: all fields empty, iteration_count=0

Step 2: SUPERVISOR (iter=1)
        Reads state: has_research=False, has_analysis=False, has_report=False
        LLM decides: next="Researcher"
        Reason: "Research findings are empty"
        State changes: next="Researcher", iteration_count=1

Step 3: RESEARCHER
        Tool call #1: search_compliance_docs("electronics China tariff")
          -> gets HS 8471 record: tariff 0.0%, ITA zero-duty
        Tool call #2: web_search("EU electronics tariff China 2024")
          -> gets stub (Tavily not configured)
        Synthesises -> research_findings is populated
        State changes: research_findings="PRODUCT: electronics (HS 8471)..."

Step 4: SUPERVISOR (iter=2)
        Reads state: has_research=True, has_analysis=False, has_report=False
        LLM decides: next="Analyst"
        Reason: "Research complete, need calculations"
        State changes: next="Analyst", iteration_count=2

Step 5: ANALYST
        Reads research_findings from state
        Tool call #1: calculate_tariff_impact(base_price=50, tariff_rate=0.0, volume=5000)
          -> Total duty: EUR 0.00, Total landed cost: EUR 250,000.00
        Tool call #2: summarize_research(research_findings)
          -> 5 compliance bullet points
        Synthesises -> analysis_results is populated
        State changes: analysis_results="FINANCIAL ANALYSIS:..."

Step 6: SUPERVISOR (iter=3)
        Reads state: has_research=True, has_analysis=True, has_report=False
        LLM decides: next="Writer"
        Reason: "Both research and analysis present, need final report"
        State changes: next="Writer", iteration_count=3

Step 7: WRITER
        Reads research_findings + analysis_results + query from state
        Single LLM call -> report with 3 sections
        State changes: final_report="## Introduction..."

Step 8: SUPERVISOR (iter=4)
        Reads state: has_research=True, has_analysis=True, has_report=True
        LLM decides: next="FINISH"
        State changes: next="FINISH", iteration_count=4

Step 9: Graph stops (FINISH -> END)
        graph.invoke() returns the final state

Total: 4 Supervisor iterations, 2 agent handoffs, wall time ~38 seconds
```

### State Transition Diagram

```
                   [START]
                      |
              state["next"]=""
                      |
               +------v------+
               | SUPERVISOR  |  iter=1 -> next="Researcher"
               +------+------+
                      |
              +-------v--------+
              |   RESEARCHER   |  tool: search_compliance_docs
              |                |  tool: web_search
              +-------+--------+
                      |
               +------v------+
               | SUPERVISOR  |  iter=2 -> next="Analyst"
               +------+------+
                      |
               +------v------+
               |   ANALYST   |  tool: calculate_tariff_impact
               |             |  tool: summarize_research
               +------+------+
                      |
               +------v------+
               | SUPERVISOR  |  iter=3 -> next="Writer"
               +------+------+
                      |
               +------v------+
               |   WRITER    |  direct LLM invoke (no tools)
               +------+------+
                      |
               +------v------+
               | SUPERVISOR  |  iter=4 -> next="FINISH"
               +------+------+
                      |
                    [END]
```

---

## How to Run

### One-time setup

```powershell
# 1. Create .env file
cd A:\Web\agentic-llm
copy .env.example .env
# Edit .env and fill in SUMOPOD_API_KEY with your key

# 2. Install dependencies
pip install langgraph langchain langchain-community langchain-openai python-dotenv pydantic
```

### Run

```powershell
cd A:\Web\agentic-llm
$env:PYTHONIOENCODING="utf-8"
python -m p4_multi_agent.main
```

Custom query (pass as an argument):

```powershell
python -m p4_multi_agent.main "research EU tariff implications for textile products from Vietnam, calculate cost impact for 10000 units at EUR 5, write executive summary"
```

### Products supported by the mock database

`textile`, `electronics`, `footwear`, `pharmaceutical`, `furniture`

Queries containing other keywords return all 5 products as a fallback.

---

## Real Weaknesses & Trade-offs

These are not from the docs — these were found during actual testing.

### Weakness 1: Latency multiplies fast

| Version      | API Calls | Wall Time |
|--------------|-----------|-----------|
| Single-agent | ~3 calls  | ~8-12s    |
| Multi-agent  | ~9 calls  | ~35-40s   |

Every Supervisor call is one API round-trip (~2s). Add the ReAct loop for
each sub-agent (2-3 calls per agent). Total 9+ calls for a query that a
single agent could answer in 3 calls.

### Weakness 2: State passing is lossy

The Supervisor only knows `has_research = True/False` — it does not read
the **contents** of `research_findings`. If the Researcher produces
ambiguous or incorrect data, the Supervisor has no idea. The Analyst then
forwards those bad numbers to the Writer, who confidently reports them.

In a single-agent setup, one model holds all context simultaneously — it is
far easier to catch internal inconsistencies.

### Weakness 3: Debugging multi-hop failures is painful

When the final output is wrong, the question is: where did it break?
- Bad research?
- Analyst misread the numbers?
- Writer synthesised incorrectly?
- Supervisor routed wrong?

Without explicit logging at each node (like `print("[ANALYST] Done...")`)
or a tool like LangSmith, you only receive the wrong final report with no
indication of which step broke.

In a single-agent setup: one chain-of-thought, immediately visible where
the reasoning went wrong.

---

### When is Single-Agent Strictly Better?

| Scenario | Best Choice | Reason |
|----------|-------------|--------|
| Simple query ("what is the tariff for HS 8471?") | Single-agent | Multi-agent is 4x slower for identical output |
| Latency matters (real-time chatbot) | Single-agent | 38s is not acceptable UX |
| Tight budget | Single-agent | Multi-agent uses 3x more tokens |
| Development / prompt iteration phase | Single-agent | Far easier to debug |
| Complex multi-domain query | Multi-agent | Specialist toolsets per agent make sense |
| High scale (hundreds of parallel queries) | Multi-agent | Different models can be assigned per task |

**Honest conclusion:** For this project, a single-agent with all 4 tools
would produce equivalent output 3x faster and at 2-3x lower cost.
Multi-agent is worth it only when:
1. Each agent genuinely needs a different model (e.g. Researcher needs web
   browsing, Writer needs a fine-tuned report model)
2. Agents can run in parallel (here everything is sequential)
3. Different dev teams maintain different agents
