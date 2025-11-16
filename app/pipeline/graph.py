from langgraph.graph import StateGraph, END
from app.pipeline.state import ComplaintState
from app.pipeline.nodes import (
    normalize_node,
    language_cleanup_node,
    category_node,
)


def build_complaint_graph():
    graph = StateGraph(ComplaintState)

    graph.add_node("normalize", normalize_node)
    graph.add_node("language_cleanup", language_cleanup_node)
    graph.add_node("category", category_node)

    graph.set_entry_point("normalize")

    graph.add_edge("normalize", "language_cleanup")
    graph.add_edge("language_cleanup", "category")
    graph.add_edge("category", END)

    return graph.compile()
