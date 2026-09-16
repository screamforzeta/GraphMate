"""Canonical graph representation metadata."""

NODE_FEATURES = [
    "piece_pawn",
    "piece_knight",
    "piece_bishop",
    "piece_rook",
    "piece_queen",
    "piece_king",
    "color_white",
    "occupied",
    "normalized_row",
    "normalized_column",
    "attacked_by_white",
    "attacked_by_black",
    "legal_mobility",
    "is_pinned",
    "piece_value",
]

EDGE_FEATURES = ["legal_move", "attack", "defend", "pin", "check_line"]

GLOBAL_FEATURES = [
    "side_to_move_white",
    "is_check",
    "fullmove_number_clipped_200_div_200",
    "halfmove_clock_clipped_100_div_100",
]


def graph_representation_metadata():
    """Return canonical graph dimensions and feature names."""

    return {
        "nodes": 64,
        "node_dim": len(NODE_FEATURES),
        "edge_dim": len(EDGE_FEATURES),
        "global_dim": len(GLOBAL_FEATURES),
        "node_features": NODE_FEATURES,
        "edge_features": EDGE_FEATURES,
        "global_features": GLOBAL_FEATURES,
    }
