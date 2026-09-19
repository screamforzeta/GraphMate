"""Position-aware relaxed parsing for post-hoc LLM chess move analysis.

The parser is a secondary analysis tool. It uses only the raw model text and
the current FEN to recover unambiguous chess moves when notation is valid or
safely normalizable. It never uses target moves or future puzzle metadata.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import chess

from src.llm.parsing import UCI_PATTERN, parse_uci_response


RELAXED_PARSER_VERSION = "relaxed_chess_move_v1"
TRAILING_TEXT_PUNCTUATION = ".。;:"
ANNOTATION_PATTERN = re.compile(r"([!?]+)$")
UCI_HYPHEN_PATTERN = re.compile(r"^([a-h][1-8])[-\s]([a-h][1-8])(?:=?([qrbnQRBN]))?$")
MALFORMED_COORD_CAPTURE_PATTERN = re.compile(r"^([a-h][1-8])x([a-h][1-8])(?:=?([qrbnQRBN]))?[+#]?$")
TEXT_CANDIDATE_PATTERN = re.compile(
    r"(?:[a-h][1-8][-\s]?[a-h][1-8](?:=?[qrbnQRBN])?|"
    r"[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=?[QRBN])?[+#]?[!?]*|"
    r"O-O-O|O-O|0-0-0|0-0)"
)


@dataclass(frozen=True)
class RelaxedParseResult:
    """Structured result for relaxed chess move parsing."""

    raw_content: str
    normalized_content: str
    parse_status: str
    parse_method: str
    candidate_notation: str | None = None
    parsed_uci: str | None = None
    parsed_san: str | None = None
    is_legal: bool = False
    ambiguity_count: int = 0
    candidate_count: int = 0
    failure_reason: str | None = None
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self):
        """Return a JSON-serializable representation."""

        return asdict(self)


def normalize_outer_text(raw_content):
    """Apply safe outer-text normalization without changing move semantics."""

    raw = "" if raw_content is None else str(raw_content)
    text = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    diagnostics = []
    if (
        (text.startswith("```") and text.endswith("```"))
        or (text.startswith("`") and text.endswith("`") and text.count("`") == 2)
    ):
        text = text.strip("`").strip()
        diagnostics.append("removed_surrounding_backticks")
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
        diagnostics.append("removed_surrounding_quotes")
    stripped = text.rstrip(TRAILING_TEXT_PUNCTUATION).strip()
    if stripped != text:
        text = stripped
        diagnostics.append("removed_trailing_text_punctuation")
    return text, diagnostics


def safe_san_variants(text):
    """Yield deterministic SAN notation variants for one normalized token."""

    variants = []
    castle = text.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    if castle != text:
        variants.append((castle, "zero_castling_to_letter_o"))
    annotation_stripped = ANNOTATION_PATTERN.sub("", text)
    if annotation_stripped != text:
        variants.append((annotation_stripped, "removed_san_annotation_glyphs"))
    castle_then_annotation = ANNOTATION_PATTERN.sub("", castle)
    if castle_then_annotation != text and castle_then_annotation != castle:
        variants.append((castle_then_annotation, "normalized_castling_and_annotations"))
    seen = set()
    for variant, rule in variants:
        if variant and variant not in seen:
            seen.add(variant)
            yield variant, rule


def safe_uci_variants(text):
    """Yield deterministic UCI-equivalent coordinate variants."""

    match = UCI_HYPHEN_PATTERN.fullmatch(text)
    if match:
        promotion = (match.group(3) or "").lower()
        yield f"{match.group(1)}{match.group(2)}{promotion}", "coordinate_separator_removed"
    match = MALFORMED_COORD_CAPTURE_PATTERN.fullmatch(text)
    if match:
        promotion = (match.group(3) or "").lower()
        yield f"{match.group(1)}{match.group(2)}{promotion}", "coordinate_capture_source_destination"


def legal_move_result(raw_content, normalized_content, board, move, method, candidate, diagnostics=None):
    """Build a successful parse result for one legal move."""

    return RelaxedParseResult(
        raw_content="" if raw_content is None else str(raw_content),
        normalized_content=normalized_content,
        parse_status="PARSED",
        parse_method=method,
        candidate_notation=candidate,
        parsed_uci=move.uci(),
        parsed_san=board.san(move),
        is_legal=True,
        ambiguity_count=0,
        candidate_count=1,
        diagnostics=diagnostics or {},
    )


def parse_san_candidate(raw_content, normalized_content, board, candidate, method, diagnostics=None):
    """Parse one candidate as exact SAN and return a result or None."""

    try:
        move = board.parse_san(candidate)
    except ValueError:
        return None
    if method == "SAN" and board.san(move) != candidate:
        return None
    return legal_move_result(raw_content, normalized_content, board, move, method, candidate, diagnostics)


def parse_uci_candidate(raw_content, normalized_content, board, candidate, method, diagnostics=None):
    """Parse one candidate as UCI and preserve illegal-UCI diagnostics."""

    if not UCI_PATTERN.fullmatch(candidate):
        return None
    strict = parse_uci_response(candidate, board.fen())
    if strict.legal:
        return legal_move_result(
            raw_content,
            normalized_content,
            board,
            chess.Move.from_uci(candidate),
            method,
            candidate,
            diagnostics,
        )
    return RelaxedParseResult(
        raw_content="" if raw_content is None else str(raw_content),
        normalized_content=normalized_content,
        parse_status="ILLEGAL_MOVE",
        parse_method=method,
        candidate_notation=candidate,
        parsed_uci=candidate,
        parsed_san=None,
        is_legal=False,
        ambiguity_count=0,
        candidate_count=1,
        failure_reason="syntactically_valid_uci_illegal",
        diagnostics=diagnostics or {},
    )


def parse_atomic_notation(raw_content, normalized_content, board, text):
    """Parse one isolated move-like string without text-wrapper extraction."""

    result = parse_uci_candidate(raw_content, normalized_content, board, text, "STRICT_UCI")
    if result is not None:
        return result
    result = parse_san_candidate(raw_content, normalized_content, board, text, "SAN")
    if result is not None:
        return result
    for variant, rule in safe_san_variants(text):
        result = parse_san_candidate(
            raw_content,
            normalized_content,
            board,
            variant,
            "NORMALIZED_SAN",
            {"recovery_rule": rule, "source_notation": text},
        )
        if result is not None:
            return result
    for variant, rule in safe_uci_variants(text):
        result = parse_uci_candidate(
            raw_content,
            normalized_content,
            board,
            variant,
            "NORMALIZED_UCI",
            {"recovery_rule": rule, "source_notation": text},
        )
        if result is not None and result.is_legal:
            return result
    return None


def unique_legal_candidates(raw_content, normalized_content, board, candidates):
    """Return unique legal parse results for candidate strings."""

    results = []
    seen_uci = set()
    for candidate in candidates:
        parsed = parse_atomic_notation(raw_content, normalized_content, board, candidate)
        if parsed and parsed.is_legal and parsed.parsed_uci not in seen_uci:
            seen_uci.add(parsed.parsed_uci)
            results.append(parsed)
    return results


def extract_text_candidates(text):
    """Extract plausible move expressions from text without fuzzy ranking."""

    candidates = []
    seen = set()
    for match in TEXT_CANDIDATE_PATTERN.finditer(text):
        candidate = match.group(0).strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    return candidates


def relaxed_parse_move(raw_final_content, fen):
    """Parse an LLM chess move response using relaxed, target-independent rules."""

    raw = "" if raw_final_content is None else str(raw_final_content)
    normalized, normalization_steps = normalize_outer_text(raw)
    diagnostics = {"normalization_steps": normalization_steps}
    if not normalized:
        return RelaxedParseResult(
            raw_content=raw,
            normalized_content=normalized,
            parse_status="EMPTY",
            parse_method="EMPTY",
            failure_reason="empty_after_normalization",
            diagnostics=diagnostics,
        )
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        return RelaxedParseResult(
            raw_content=raw,
            normalized_content=normalized,
            parse_status="UNRECOVERABLE",
            parse_method="UNRECOVERABLE",
            failure_reason=f"invalid_fen:{exc}",
            diagnostics=diagnostics,
        )
    direct = parse_atomic_notation(raw, normalized, board, normalized)
    if direct is not None:
        return direct
    candidates = extract_text_candidates(normalized)
    legal_candidates = unique_legal_candidates(raw, normalized, board, candidates)
    if len(legal_candidates) == 1:
        parsed = legal_candidates[0]
        return RelaxedParseResult(
            raw_content=raw,
            normalized_content=normalized,
            parse_status="PARSED",
            parse_method="TEXT_WRAPPED_MOVE",
            candidate_notation=parsed.candidate_notation,
            parsed_uci=parsed.parsed_uci,
            parsed_san=parsed.parsed_san,
            is_legal=True,
            ambiguity_count=0,
            candidate_count=len(candidates),
            diagnostics={"text_candidates": candidates, **diagnostics},
        )
    if len(legal_candidates) > 1:
        return RelaxedParseResult(
            raw_content=raw,
            normalized_content=normalized,
            parse_status="AMBIGUOUS",
            parse_method="AMBIGUOUS",
            ambiguity_count=len(legal_candidates),
            candidate_count=len(candidates),
            failure_reason="multiple_legal_move_candidates",
            diagnostics={
                "text_candidates": candidates,
                "legal_candidate_uci": [candidate.parsed_uci for candidate in legal_candidates],
                **diagnostics,
            },
        )
    return RelaxedParseResult(
        raw_content=raw,
        normalized_content=normalized,
        parse_status="UNRECOVERABLE",
        parse_method="UNRECOVERABLE",
        candidate_count=len(candidates),
        failure_reason="no_unique_legal_move",
        diagnostics={"text_candidates": candidates, **diagnostics},
    )
