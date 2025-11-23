from langgraph.graph import StateGraph, END

from app.pipeline import nodes
from app.pipeline.state import ConversationGraphState


def build_complaint_graph():
    graph = StateGraph(ConversationGraphState)

    graph.add_node("normalize", nodes.normalize_node)
    graph.add_node("category", nodes.classifier_node_3)
    graph.add_node("service_search", nodes.search_service_node)
    graph.add_node("ask_clarification", nodes.ask_clarification_node)
    graph.add_node("handle_failure", nodes.handle_failure_node)
    graph.add_node("generate_appeal", nodes.generate_appeal_node)
    graph.add_node("emergency", nodes.emergency_node)
    graph.add_node("location", nodes.location)
    graph.add_node("out_of_scope", nodes.out_of_scope)
    graph.add_node("no_services", nodes.no_service)

    graph.set_entry_point("normalize")

    graph.add_conditional_edges(
        "normalize",
        nodes.route_after_normalize,
        {
            "end": END,
            "category": "category",
        },
    )

    graph.add_conditional_edges(
        "category",
        nodes.route_after_classification,
        {
            "handle_failure": "handle_failure",
            "ask_clarification": "ask_clarification",
            "location": "location",
            "emergency": "emergency",
            "out_of_scope": "out_of_scope"
        },
    )

    graph.add_conditional_edges(
        "location",
        nodes.route_after_location,
        {
            "service_search": "service_search",
            "ask_clarification": "ask_clarification",
            "handle_failure": "handle_failure",
        }
    )

    graph.add_conditional_edges(
        "service_search",
        nodes.route_after_service_search,
        {
            "handle_failure": "handle_failure",
            "generate_appeal": "generate_appeal",
            "no_services": "no_services"
        },
    )

    graph.add_edge("ask_clarification", END)
    graph.add_edge("generate_appeal", END)
    graph.add_edge("handle_failure", END)
    graph.add_edge("emergency", END)
    graph.add_edge("out_of_scope", END)
    graph.add_edge("no_services", END)

    return graph.compile()
