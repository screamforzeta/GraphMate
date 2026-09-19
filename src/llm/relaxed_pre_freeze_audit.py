"""Pre-freeze diagnostics for relaxed LLM chess move parsing.

This module does not change the relaxed parser. It audits unrecoverable and
text-wrapper cases using only raw model text plus the current board position.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import chess

from src.llm.relaxed_parsing import normalize_outer_text, relaxed_parse_move


PIECE_SRC_DST_RE = re.compile(r"^(?P<piece>[KQRBN])(?P<src>[a-h][1-8])(?P<dst>[a-h][1-8])(?P<suffix>[+#])?$")
PIECE_SRC_SEP_DST_RE = re.compile(
    r"^(?P<piece>[KQRBN])(?P<src>[a-h][1-8])(?P<sep>[-x])(?P<dst>[a-h][1-8])(?P<suffix>[+#])?$"
)
SRC_SEP_DST_RE = re.compile(r"^(?P<src>[a-h][1-8])(?P<sep>[-x])(?P<dst>[a-h][1-8])(?P<suffix>[+#])?$")
PROMOTION_RE = re.compile(
    r"^(?:(?P<piece>[KQRBN])?)"
    r"(?P<src>[a-h][1-8])(?P<sep>[-x]?)"
    r"(?P<dst>[a-h][1-8])=?"
    r"(?P<promo>[QRBNqrbn])(?P<suffix>[+#])?$"
)
UCI_CHECK_SUFFIX_RE = re.compile(r"^(?P<src>[a-h][1-8])(?P<dst>[a-h][1-8])(?P<suffix>[+#])$")
PLAIN_COORD_RE = re.compile(r"^(?P<src>[a-h][1-8])(?P<dst>[a-h][1-8])$")
SAN_LIKE_RE = re.compile(r"^[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=?[QRBN])?[+#]?[!?]*$")
INVALID_SQUARE_OR_PIECE_RE = re.compile(r"[a-h][09]|[i-z][1-8]|^[kqrbnp]")

PIECE_SYMBOLS = {
    "K": chess.KING,
    "Q": chess.QUEEN,
    "R": chess.ROOK,
    "B": chess.BISHOP,
    "N": chess.KNIGHT,
}
PROMOTION_SYMBOLS = {
    "q": chess.QUEEN,
    "r": chess.ROOK,
    "b": chess.BISHOP,
    "n": chess.KNIGHT,
}


@dataclass(frozen=True)
class RecoverabilityResult:
    """Deterministic recoverability audit result."""

    raw_content: str
    normalized_content: str
    family: str
    recoverable: bool
    candidate_uci: str | None
    proposed_rule: str | None
    candidate_count: int
    reason: str
    ambiguous: bool = False
    impossible: bool = False
    annotation_inconsistent: bool = False
    candidate_uci_list: list[str] | None = None

    def to_dict(self) -> dict:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


def move_gives_suffix(board: chess.Board, move: chess.Move, suffix: str | None) -> bool:
    """Return whether a check/checkmate suffix is semantically consistent."""

    if not suffix:
        return True
    copied = board.copy(stack=False)
    copied.push(move)
    if suffix == "#":
        return copied.is_checkmate()
    if suffix == "+":
        return copied.is_check()
    return True


def explicit_move_candidates(
    board: chess.Board,
    source: str,
    destination: str,
    promotion: str | None = None,
) -> list[chess.Move]:
    """Return legal moves matching explicit source, destination, and promotion."""

    source_square = chess.parse_square(source)
    destination_square = chess.parse_square(destination)
    promotion_piece = PROMOTION_SYMBOLS.get(promotion.lower()) if promotion else None
    candidates = []
    for move in board.legal_moves:
        if move.from_square != source_square or move.to_square != destination_square:
            continue
        if promotion_piece is not None and move.promotion != promotion_piece:
            continue
        if promotion_piece is None and move.promotion is not None:
            continue
        candidates.append(move)
    return candidates


def piece_at_source_matches(board: chess.Board, source: str, piece_symbol: str | None) -> bool:
    """Check whether an optional piece prefix matches the side-to-move piece."""

    if piece_symbol is None:
        return True
    piece = board.piece_at(chess.parse_square(source))
    return bool(
        piece
        and piece.color == board.turn
        and piece.piece_type == PIECE_SYMBOLS[piece_symbol]
    )


def classify_family(text: str) -> tuple[str, re.Match[str] | None]:
    """Classify one normalized string into an audit family."""

    for family, pattern in (
        ("promotion_coordinate", PROMOTION_RE),
        ("piece_src_sep_dst", PIECE_SRC_SEP_DST_RE),
        ("piece_src_dst", PIECE_SRC_DST_RE),
        ("src_sep_dst", SRC_SEP_DST_RE),
        ("uci_with_check_suffix", UCI_CHECK_SUFFIX_RE),
        ("plain_coordinate", PLAIN_COORD_RE),
    ):
        match = pattern.fullmatch(text)
        if match:
            return family, match
    if SAN_LIKE_RE.fullmatch(text):
        return "san_like", None
    if "x" in text and re.search(r"[a-h][1-8].*x.*[a-h][1-8]", text):
        return "malformed_capture", None
    if INVALID_SQUARE_OR_PIECE_RE.search(text):
        return "invalid_square_or_piece", None
    if re.search(r"[KQRBNO0a-h][x=+#\-]?[a-h1-8]", text):
        return "other_chess_like", None
    return "non_chess", None


def evaluate_explicit_match(
    raw: str,
    normalized: str,
    board: chess.Board,
    family: str,
    match: re.Match[str],
    proposed_rule: str,
) -> RecoverabilityResult:
    """Evaluate explicit source/destination notation against legal moves."""

    source = match.group("src")
    destination = match.group("dst")
    suffix = match.groupdict().get("suffix")
    separator = match.groupdict().get("sep") or ""
    promotion = match.groupdict().get("promo")
    piece = match.groupdict().get("piece")
    if piece and not piece_at_source_matches(board, source, piece):
        return RecoverabilityResult(raw, normalized, family, False, None, proposed_rule, 0, "piece_prefix_mismatch", impossible=True)
    candidates = explicit_move_candidates(board, source, destination, promotion)
    if separator == "x":
        candidates = [move for move in candidates if board.is_capture(move)]
    if len(candidates) > 1:
        return RecoverabilityResult(
            raw,
            normalized,
            family,
            False,
            None,
            proposed_rule,
            len(candidates),
            "multiple_legal_moves_match_explicit_notation",
            ambiguous=True,
            candidate_uci_list=[move.uci() for move in candidates],
        )
    if not candidates:
        return RecoverabilityResult(raw, normalized, family, False, None, proposed_rule, 0, "no_legal_move_matches_explicit_notation", impossible=True)
    move = candidates[0]
    annotation_inconsistent = not move_gives_suffix(board, move, suffix)
    if annotation_inconsistent:
        return RecoverabilityResult(
            raw,
            normalized,
            family,
            False,
            move.uci(),
            proposed_rule,
            1,
            "annotation_semantically_inconsistent",
            annotation_inconsistent=True,
            candidate_uci_list=[move.uci()],
        )
    return RecoverabilityResult(raw, normalized, family, True, move.uci(), proposed_rule, 1, "unique_legal_move", candidate_uci_list=[move.uci()])


def analyze_deterministic_recoverability(raw: str, fen: str) -> RecoverabilityResult:
    """Audit whether raw text can identify one legal move from string plus board."""

    normalized, _ = normalize_outer_text(raw)
    if not normalized:
        return RecoverabilityResult(raw or "", normalized, "empty", False, None, None, 0, "empty", impossible=True)
    board = chess.Board(fen)
    family, match = classify_family(normalized)
    if match and family in {
        "piece_src_dst",
        "piece_src_sep_dst",
        "src_sep_dst",
        "promotion_coordinate",
        "uci_with_check_suffix",
        "plain_coordinate",
    }:
        rule = {
            "piece_src_dst": "piece_source_destination",
            "piece_src_sep_dst": "piece_source_separator_destination",
            "src_sep_dst": "source_separator_destination",
            "promotion_coordinate": "promotion_coordinate_variant",
            "uci_with_check_suffix": "uci_with_check_suffix",
            "plain_coordinate": "plain_source_destination",
        }[family]
        return evaluate_explicit_match(raw, normalized, board, family, match, rule)
    if family == "san_like":
        try:
            move = board.parse_san(normalized)
        except ValueError:
            return RecoverabilityResult(raw, normalized, family, False, None, "exact_san_recheck", 0, "valid_looking_san_no_legal_interpretation", impossible=True)
        return RecoverabilityResult(raw, normalized, family, True, move.uci(), "exact_san_recheck", 1, "valid_san_legal", candidate_uci_list=[move.uci()])
    return RecoverabilityResult(raw, normalized, family, False, None, None, 0, "no_deterministic_rule_applies", impossible=True)


def text_wrapper_safety(record: dict) -> dict:
    """Audit one current TEXT_WRAPPED_MOVE parse without using target data."""

    raw = record.get("raw_final_content") or ""
    fen = record["fen"]
    parsed = relaxed_parse_move(raw, fen)
    board = chess.Board(fen)
    extracted = parsed.candidate_notation
    legal_relevant = []
    if extracted:
        for move in board.legal_moves:
            san = board.san(move)
            if extracted in {move.uci(), san} or extracted.replace("-", "") == move.uci():
                legal_relevant.append({"uci": move.uci(), "san": san})
    candidate_count = len(parsed.diagnostics.get("text_candidates", []))
    if candidate_count == 1 and parsed.parsed_uci:
        safety = "SAFE"
    elif candidate_count > 1:
        safety = "UNSAFE"
    else:
        safety = "QUESTIONABLE"
    if raw == extracted:
        safety = "QUESTIONABLE"
    return {
        "puzzle_id": record.get("puzzle_id"),
        "fen": fen,
        "raw_final_content": raw,
        "legal_moves_relevant": legal_relevant,
        "parsed_uci": parsed.parsed_uci,
        "parsed_san": parsed.parsed_san,
        "extracted_token": extracted,
        "parser_rule": parsed.parse_method,
        "text_candidates": parsed.diagnostics.get("text_candidates", []),
        "target_used": False,
        "uniqueness_explanation": (
            "Current parser recovered exactly one legal candidate from extracted text candidates."
            if parsed.parsed_uci
            else "Current parser did not recover a legal unique move."
        ),
        "safety": safety,
    }


def summarize_recoverability(results: list[RecoverabilityResult]) -> dict:
    """Aggregate recoverability results by family and proposed rule."""

    families = defaultdict(lambda: Counter())
    examples = defaultdict(list)
    rules = defaultdict(lambda: Counter())
    for result in results:
        family_counter = families[result.family]
        family_counter["total"] += 1
        family_counter["recoverable_unique_move"] += int(result.recoverable)
        family_counter["ambiguous"] += int(result.ambiguous)
        family_counter["illegal"] += int(result.impossible)
        family_counter["annotation_inconsistent"] += int(result.annotation_inconsistent)
        family_counter["unique_raw_forms"] += 0
        if len(examples[result.family]) < 20:
            examples[result.family].append(result.to_dict())
        if result.proposed_rule:
            rules[result.proposed_rule]["total"] += 1
            rules[result.proposed_rule]["recoverable"] += int(result.recoverable)
            rules[result.proposed_rule]["ambiguous"] += int(result.ambiguous)
            rules[result.proposed_rule]["annotation_inconsistent"] += int(result.annotation_inconsistent)
    unique_by_family = defaultdict(set)
    for result in results:
        unique_by_family[result.family].add(result.normalized_content)
    family_rows = []
    for family, counts in sorted(families.items()):
        row = dict(counts)
        row["family"] = family
        row["unique_raw_forms"] = len(unique_by_family[family])
        family_rows.append(row)
    return {
        "families": family_rows,
        "family_examples": dict(examples),
        "rule_counts": {rule: dict(counter) for rule, counter in sorted(rules.items())},
    }


def proposed_rules_from_audits(model_summaries: dict) -> list[dict]:
    """Build proposed rule metadata from per-model audit summaries."""

    rule_descriptions = {
        "piece_source_destination": "Recover KQRBN + explicit source square + destination square.",
        "piece_source_separator_destination": "Recover KQRBN + explicit source + '-'/'x' + destination.",
        "source_separator_destination": "Recover explicit source + '-'/'x' + destination.",
        "promotion_coordinate_variant": "Recover explicit coordinate promotion variants with optional separator/capture/check marker.",
        "uci_with_check_suffix": "Recover UCI coordinate notation followed by +/# only when annotation is semantically consistent.",
        "plain_source_destination": "Recover explicit source + destination coordinate notation if legal.",
        "exact_san_recheck": "Re-check valid-looking SAN strings that current parser marked unrecoverable.",
    }
    rules = []
    for rule_id, description in rule_descriptions.items():
        recoverable_by_model = {
            model: summary["rule_counts"].get(rule_id, {}).get("recoverable", 0)
            for model, summary in model_summaries.items()
        }
        total_recoverable = sum(recoverable_by_model.values())
        if not total_recoverable:
            recommendation = "DO_NOT_ADD"
        elif rule_id in {"plain_source_destination", "exact_san_recheck"}:
            recommendation = "NEEDS_REVIEW"
        else:
            recommendation = "SAFE_TO_ADD"
        rules.append(
            {
                "rule_id": rule_id,
                "description": description,
                "syntax_regex_or_logic": "See src.llm.relaxed_pre_freeze_audit classify/evaluate logic.",
                "semantic_checks": [
                    "raw string plus board only",
                    "exactly one legal move",
                    "piece prefix matches source piece when present",
                    "x separator requires an actual capture",
                    "+/# suffix must match check/checkmate when present",
                ],
                "recoverable_count_4b": recoverable_by_model.get("qwen_3_5_4b", 0),
                "recoverable_count_9b": recoverable_by_model.get("qwen_3_5_9b", 0),
                "ambiguity_count": sum(
                    summary["rule_counts"].get(rule_id, {}).get("ambiguous", 0)
                    for summary in model_summaries.values()
                ),
                "known_risks": (
                    "Plain coordinate and SAN-like rules can overlap with current parser semantics; review examples before changing parser."
                    if rule_id in {"plain_source_destination", "exact_san_recheck"}
                    else "Low if all semantic checks remain mandatory."
                ),
                "recommendation": recommendation,
            }
        )
    return rules
