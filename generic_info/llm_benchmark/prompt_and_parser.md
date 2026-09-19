# Prompt And Parser

## Frozen Prompt

System:

```text
You are solving a chess position.
Return exactly one legal chess move for the side to move.
Do not provide explanation or analysis.
```

User:

```text
FEN:
{fen}

Return your move in UCI notation only.
```

Prompt version: `llm_chess_uci_v1`

## Parser Policy

Parser version: `strict_uci_v1`

Accepted:

- `e2e4`
- leading/trailing whitespace around one UCI token
- promotion UCI such as `e7e8q`

Rejected:

- explanations
- SAN such as `Nf3`
- coordinate hyphens such as `e2-e4`
- multiple moves
- markdown/code fences
- malformed coordinates

Every parsed move is validated against `chess.Board(fen).legal_moves`.

