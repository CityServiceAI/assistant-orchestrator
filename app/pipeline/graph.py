from langgraph.graph import StateGraph, END

from app.pipeline.conversation_graph_state import ConversationGraphState
from app.pipeline.nodes import (
    normalize_node,
    category_node,
    search_service_node,
    ask_clarification_node,
    route_after_classification,
    route_after_service_search,
    handle_failure_node,
    generate_appeal_node,
)


def build_complaint_graph():
    graph = StateGraph(ConversationGraphState)

    graph.add_node("normalize", normalize_node)
    graph.add_node("category", category_node)
    graph.add_node("service_search", search_service_node)
    graph.add_node("ask_clarification", ask_clarification_node)
    graph.add_node("handle_failure", handle_failure_node)
    graph.add_node("generate_appeal", generate_appeal_node)

    graph.set_entry_point("normalize")

    graph.add_edge("normalize", "category")

    graph.add_conditional_edges(
        "category",
        route_after_classification,
        {
            "handle_failure": "handle_failure",
            "ask_clarification": "ask_clarification",
            "service_search": "service_search",
        },
    )

    graph.add_conditional_edges(
        "service_search",
        route_after_service_search,
        {"handle_failure": "handle_failure", "generate_appeal": "generate_appeal"},
    )

    graph.add_edge("ask_clarification", END)
    graph.add_edge("generate_appeal", END)
    graph.add_edge("handle_failure", END)

    return graph.compile()
