from langgraph.graph import StateGraph, END

from app.pipeline.state import ConversationGraphState

from app.pipeline.nodes import (
    normalize_node,
    category_node,
    search_service_node,
    ask_clarification_node,
    route_after_normalize,
    route_after_classification,
    route_after_service_search,
    handle_failure_node,
    generate_appeal_node,
    classifier_node,
    emergency_node,
    location,
    route_after_location,
)


def build_complaint_graph():
    graph = StateGraph(ConversationGraphState)

    graph.add_node("normalize", normalize_node)
    graph.add_node("category", classifier_node)
    graph.add_node("service_search", search_service_node)
    graph.add_node("ask_clarification", ask_clarification_node)
    graph.add_node("handle_failure", handle_failure_node)
    graph.add_node("generate_appeal", generate_appeal_node)
    graph.add_node("emergency", emergency_node)
    graph.add_node("location", location)

    graph.set_entry_point("normalize")

    graph.add_conditional_edges(
        "normalize",
        route_after_normalize,
        {
            "end": END,
            "category": "category",
        },
    )

    graph.add_conditional_edges(
        "category",
        route_after_classification,
        {
            "handle_failure": "handle_failure",
            "ask_clarification": "ask_clarification",
            "location": "location",
            "emergency": "emergency"
        },
    )

    graph.add_conditional_edges(
        "location",
        route_after_location,
        {
            "service_search": "service_search",
            "ask_clarification": "ask_clarification",
            "handle_failure": "handle_failure",
        }
    )

    graph.add_conditional_edges(
        "service_search",
        route_after_service_search,
        {
            "handle_failure": "handle_failure",
            "generate_appeal": "generate_appeal"
        },
    )

    graph.add_edge("ask_clarification", END)
    graph.add_edge("generate_appeal", END)
    graph.add_edge("handle_failure", END)
    graph.add_edge("emergency", END)

    return graph.compile()
