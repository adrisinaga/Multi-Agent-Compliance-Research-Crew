"""Assembles and compiles the LangGraph StateGraph for the multi-agent crew."""

from langgraph.graph import END, StateGraph

from p4_multi_agent.agents.analyst import analyst_node
from p4_multi_agent.agents.researcher import researcher_node
from p4_multi_agent.agents.supervisor import supervisor_node
from p4_multi_agent.agents.writer import writer_node
from p4_multi_agent.state import AgentState


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("Researcher", researcher_node)
    workflow.add_node("Analyst", analyst_node)
    workflow.add_node("Writer", writer_node)

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "Researcher": "Researcher",
            "Analyst": "Analyst",
            "Writer": "Writer",
            "FINISH": END,
        },
    )

    workflow.add_edge("Researcher", "supervisor")
    workflow.add_edge("Analyst", "supervisor")
    workflow.add_edge("Writer", "supervisor")

    return workflow.compile()


graph = build_graph()
