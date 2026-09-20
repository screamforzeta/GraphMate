"""Conservative YACPDB solution-key extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

import chess


MOVE_NUMBER_RE = re.compile(r"^\s*1\.(?!\.)\s*(?P<rest>.+)$")
DEFENSE_RE = re.compile(r"^\s*1\.\.\.")
COMMENT_RE = re.compile(r"\{[^}]*\}")
POPEYE_MOVE_RE = re.compile(
    r"^(?P<piece>[KQRBSN])?(?P<src>[a-h][1-8])(?P<sep>[-x*])(?P<dst>[a-h][1-8])(?:=(?P<promo>[QRBSN]))?$"
)
PROMOTION_MAP = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "S": chess.KNIGHT, "N": chess.KNIGHT}
PIECE_MAP = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "S": chess.KNIGHT, "N": chess.KNIGHT}


@dataclass(frozen=True)
class KeyExtraction:
    """Structurally identified and legally resolved YACPDB key."""

    key_token: str
    key_move_uci: str
    parse_method: str
    candidate_line: str
    skipped_try_count: int
    skipped_setplay_count: int


def strip_comments(text: str) -> str:
    """Remove source comments for token classification."""

    return COMMENT_RE.sub(" ", text)


def first_token_after_move_number(line: str) -> str | None:
    """Return the first key-token candidate from a `1.` line."""

    match = MOVE_NUMBER_RE.match(strip_comments(line))
    if not match:
        return None
    rest = match.group("rest").strip()
    if not rest:
        return None
    return rest.split()[0].rstrip("!+#")


def structurally_identify_key(solution: str) -> tuple[str | None, str | None, dict[str, int | str]]:
    """Identify exactly one actual key token without using board legality."""

    if not solution or not str(solution).strip():
        return None, "NO_SOLUTION", {}
    actual_candidates: list[tuple[str, str]] = []
    skipped_try_count = 0
    skipped_setplay_count = 0
    for raw_line in str(solution).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if DEFENSE_RE.match(line):
            skipped_setplay_count += 1
            continue
        token = first_token_after_move_number(line)
        if token is None:
            continue
        if "?" in strip_comments(line).split(token, 1)[-1][:8] or token.endswith("?"):
            skipped_try_count += 1
            continue
        if re.search(r"\bbut\b", line, flags=re.IGNORECASE):
            skipped_try_count += 1
            continue
        actual_candidates.append((token.rstrip("?"), raw_line))
    if not actual_candidates:
        if skipped_try_count:
            return None, "TRY_ONLY", {"skipped_try_count": skipped_try_count, "skipped_setplay_count": skipped_setplay_count}
        return None, "NO_ACTUAL_KEY", {"skipped_try_count": skipped_try_count, "skipped_setplay_count": skipped_setplay_count}
    unique_tokens = {token for token, _line in actual_candidates}
    if len(unique_tokens) != 1:
        return None, "AMBIGUOUS_KEY", {"candidate_count": len(unique_tokens), "skipped_try_count": skipped_try_count, "skipped_setplay_count": skipped_setplay_count}
    token, line = actual_candidates[0]
    return token, None, {"candidate_line": line, "skipped_try_count": skipped_try_count, "skipped_setplay_count": skipped_setplay_count}


def resolve_key_token(token: str, board: chess.Board) -> tuple[str | None, str | None, str | None]:
    """Resolve one structurally identified key token against the starting board."""

    clean = token.strip().rstrip("!+#")
    if clean in {"O-O", "0-0", "O-O-O", "0-0-0"}:
        if not board.castling_rights:
            return None, "UNKNOWN_CASTLING_RIGHTS", None
        try:
            move = board.parse_san(clean.replace("0", "O"))
        except ValueError:
            return None, "KEY_NOT_LEGAL", None
        return move.uci(), None, "castling"
    match = POPEYE_MOVE_RE.fullmatch(clean)
    if match:
        src = chess.parse_square(match.group("src"))
        dst = chess.parse_square(match.group("dst"))
        promotion = PROMOTION_MAP.get(match.group("promo")) if match.group("promo") else None
        piece_prefix = match.group("piece")
        piece = board.piece_at(src)
        if piece_prefix and (not piece or piece.piece_type != PIECE_MAP[piece_prefix] or piece.color != board.turn):
            return None, "KEY_PIECE_PREFIX_MISMATCH", None
        move = chess.Move(src, dst, promotion=promotion)
        if move not in board.legal_moves:
            return None, "KEY_NOT_LEGAL", None
        if match.group("sep") in {"x", "*"} and not board.is_capture(move):
            return None, "CAPTURE_MARKER_MISMATCH", None
        if match.group("sep") == "-" and board.is_capture(move):
            return None, "CAPTURE_MARKER_MISMATCH", None
        return move.uci(), None, "popeye_coordinate"
    try:
        move = board.parse_san(clean)
    except ValueError:
        return None, "UNSUPPORTED_MOVE_NOTATION", None
    if move not in board.legal_moves:
        return None, "KEY_NOT_LEGAL", None
    return move.uci(), None, "san"


def extract_structural_key(solution: str, board: chess.Board) -> tuple[KeyExtraction | None, str | None]:
    """Extract and legally validate the actual solution key."""

    token, reason, metadata = structurally_identify_key(solution)
    if reason or token is None:
        return None, reason
    uci, resolve_error, method = resolve_key_token(token, board)
    if resolve_error or uci is None or method is None:
        return None, resolve_error
    return (
        KeyExtraction(
            key_token=token,
            key_move_uci=uci,
            parse_method=method,
            candidate_line=str(metadata.get("candidate_line", "")),
            skipped_try_count=int(metadata.get("skipped_try_count", 0)),
            skipped_setplay_count=int(metadata.get("skipped_setplay_count", 0)),
        ),
        None,
    )
