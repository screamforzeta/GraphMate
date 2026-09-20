# YACPDB Held-Out Import

Status: `NORMALIZATION_READY`

This note documents the YACPDB support added for the external classic Mate-in-N held-out dataset. It is an import and validation boundary only: no GNN inference, no LLM inference, no engine proof, and no modification of frozen model artifacts.

## Source Discovery

YACPDB is a public chess problem database available at:

```text
https://www.yacpdb.org/
```

The browsed public pages expose a web search interface and the documented Py2Web embedding format:

```text
https://www.yacpdb.org/static/py2web.html
```

The first pass did not establish a stable documented REST download API. A later technical smoke pass inspected FEN Tool, Olive GUI, and the live YACPDB web UI. It found a public but undocumented query endpoint that can support controlled query/fetch/raw-cache acquisition.

## YACPDB Acquisition Mechanism

### Open-Source Clients Inspected

| Project | Files inspected | Relevant function/code |
|---|---|---|
| FEN Tool | `/tmp/fen-tool/src/app/tools/yacpdb.ts` | `YACPDB.fetch()` builds `https://yacpdb.org/gateway/ql?q=Id('<id>')`; `YACPDB.search()` opens `https://yacpdb.org/#q/<query>/1`; `createQuery()` builds `MatrixExtended(...)` queries. |
| FEN Tool | `/tmp/fen-tool/src/app/meta/fen.ts` | YACPDB-style FEN/piece notation conversion helpers. |
| Olive GUI | `/tmp/olive-gui/yacpdb/indexer/indexer.md` | Documents the YACPDB query language, including `Id(INTEGER id)`, `Stip(STRING regex)`, `Option(STRING option)`, `Fairy`, `Twins`, and pagination examples in hash URLs. |
| Olive GUI | `/tmp/olive-gui/yacpdb/schemas/yacpdb-entry.md` | Documents the YACPDB YAML entry format. |
| Olive GUI | `/tmp/olive-gui/yacpdb/schemas/yacpdb-entry.schema.json` | Formal schema: required `stipulation`, `algebraic`, `solution`; optional `authors`, `source`, `award`, `twins`, `options`, `keywords`, `reprints`, `foreignids`. |
| Olive GUI | `/tmp/olive-gui/gui.py` | `YamlView.onValidate()` posts YAML to `json.php` for validation; this is not a retrieval endpoint. |
| YACPDB web UI | `https://www.yacpdb.org/static/js/assets.js` | Route `#q/:query/:page` calls `gateway/ql?q=<query>&p=<page>`; search form calls `json.php?search&query=<encoded-form>&page=<page>`. |

### Endpoint Summary

| Operation | Method | Endpoint | Parameters | Response |
|---|---|---|---|---|
| QL search | `GET` | `https://yacpdb.org/gateway/ql` | `q=<query>`, optional `p=<page>` | JSON object with `success`, `result.entries`, `result.count` |
| Fetch by ID | `GET` | `https://yacpdb.org/gateway/ql` | `q=Id(<id>)` or FEN Tool style `Id('<id>')` | Same JSON structure, normally one entry |
| Website search form | `GET` | `https://yacpdb.org/json.php` | `search`, `query=<base64-form>`, `page=<page>` | JSON rendered by web UI; less suitable for scripted acquisition |
| YAML validation | `POST` | `https://yacpdb.org/json.php` | `checkedit=1`, `id=-1`, `yamlText=<yaml>`, `readonly=1` | JSON validation result; not used for acquisition |

Endpoint confidence:

| Endpoint | Classification | Evidence |
|---|---|---|
| `gateway/ql` | `PUBLIC_BUT_UNDOCUMENTED` | Used by FEN Tool and current YACPDB web UI; QL itself is documented in Olive/YACPDB docs, but the HTTP endpoint is not presented as a stable public API. |
| `json.php?search` | `CLIENT_INTERNAL` | Used by YACPDB web UI search form; request format is UI-specific encoded form state. |
| `json.php` YAML validation | `CLIENT_INTERNAL` | Used by Olive GUI and YACPDB edit/preview flows; not a retrieval API. |

Authentication:

- `gateway/ql` smoke requests worked without login or cookies.
- Login appears required only for edit/auth flows.

Pagination:

- YACPDB web UI route `#q/:query/:page` calls `gateway/ql?q=<query>&p=<page>`.
- The current web UI sets `searchResultsPerPage = 100`.
- Responses include `result.count`; total pages are derived as `ceil(count / 100)`.
- Result ordering should be treated as server-defined and not assumed immutable unless cached with IDs and manifest.

### Query Syntax

The query language supports predicates and Boolean operators:

```text
Predicate
Predicate CMP INT
NOT Expression
Expression AND Expression
Expression OR Expression
```

Useful predicates for this project:

| Need | QL expression |
|---|---|
| Fetch exact problem | `Id(26026)` |
| Directmate Mate-in-1 | `Stip("^#1$")` |
| Directmate Mate-in-N | `Stip("^#N$")` |
| No options/conditions | `Option("*") = 0` |
| Exclude fairy entries | `NOT Fairy` |
| Exclude twins | `Twins = 1` |

Candidate future acquisition query shape:

```text
Stip("^#N$") AND NOT Fairy AND Option("*") = 0 AND Twins = 1
```

This query shape has not yet been run as a full #1-#10 availability scan.

### Minimal Live Smoke

Live YACPDB contact was performed only for bounded technical validation:

| Request | Records | Raw file |
|---|---:|---|
| `GET /gateway/ql?q=Id(26026)` | 1 | `data/heldout_classic/raw/yacpdb/smoke/fetch_id_26026.json` |
| `GET /gateway/ql?q=Stip("^#2$") AND Id(26026)&p=1` | 1 | `data/heldout_classic/raw/yacpdb/smoke/query_stip_2_id_26026_page_1.json` |

No broad stipulation scan was run.

### Real Record Structure Observed

The smoke record `26026` contains:

| Category | Observed fields |
|---|---|
| Identity | `id`, `ash` |
| Stipulation | `stipulation: "#2"` |
| Position | `algebraic.white`, `algebraic.black`; no ready-to-use FEN field |
| Solution | `solution` as Popeye-style multiline text |
| Provenance | `authors`, `source.name`, `source.date.year`, `source.issue`, `source.problemid` |
| Keywords | `keywords` |
| Extra | `legend`, `transliterations` |

Position representation is `B`: piece-list / algebraic YACPDB structure. The current local adapter expects `fen` or `FEN`, so it cannot directly consume raw live YACPDB records yet.

The observed solution includes set/play or try-like lines before the actual key:

```text
1...d4-d3 ...
...
1.Qf3-d1 ! zugzwang.
```

The current key extractor assumes the first solution token is the actual key. That assumption does not hold for this smoke record. Parser broadening should be a separate targeted pass, not an opportunistic change from one example.

### Acquisition Client

Implemented:

```text
src/data/heldout_sources/yacpdb_client.py
```

Responsibilities:

- build deterministic QL URLs;
- fetch by ID;
- run bounded QL search;
- parse JSON response envelope;
- enforce `max_records`;
- cache raw responses plus metadata;
- timeout, limited retry, and User-Agent.

Non-responsibilities:

- chess normalization;
- FEN conversion;
- solution parsing;
- sampling;
- contamination checks;
- model inference.

Diagnostic smoke command:

```bash
./venv/bin/python -m src.data.heldout_sources.yacpdb_client \
  --mode fetch-id \
  --problem-id 26026 \
  --max-records 3 \
  --output-dir data/heldout_classic/raw/yacpdb/smoke
```

Bounded QL smoke command:

```bash
./venv/bin/python -m src.data.heldout_sources.yacpdb_client \
  --mode query \
  --query 'Stip("^#2$") AND Id(26026)' \
  --page 1 \
  --max-records 3 \
  --output-dir data/heldout_classic/raw/yacpdb/smoke
```

Future full acquisition must use explicit page/range limits and should cache raw responses before normalization.

## Supported Problem Class

The importer accepts only:

- orthodox directmates;
- stipulations `#1` through `#10`;
- standard legal FEN positions;
- one main starting position;
- one legal, unambiguous key move extracted from the source solution.

Rejected classes:

- helpmates such as `h#`;
- selfmates such as `s#`;
- series problems;
- fairy pieces or fairy conditions;
- twins;
- malformed FEN;
- missing source ID;
- missing or unsupported solution key notation;
- illegal key move;
- capture marker mismatch.

## Input Record Format

The input file may be `.json`, `.jsonl`, or `.csv`. Each record should contain as many of these fields as available:

| Field | Meaning |
|---|---|
| `id`, `yacpdb_id`, `source_problem_id` | Original YACPDB problem ID |
| `stipulation` or `stip` | Expected as `#1`..`#10` |
| `fen` or `FEN` | Solver-facing starting FEN |
| `solution` or `source_solution_raw` | Raw YACPDB/Popeye-like solution text |
| `composer` or `author` | Composer metadata |
| `source` or `publication` | Publication/reference |
| `date` or `year` | Publication date when available |
| `source_url` | Optional original URL |
| `license` or `provenance_license` | License/provenance string |

The raw source record is preserved in `source_metadata_json`.

For direct live YACPDB `gateway/ql` records, the source adapter now converts:

- `algebraic.white` / `algebraic.black` to standard FEN for orthodox positions;
- `authors` to composer metadata;
- nested `source` to publication/reference metadata;
- source `id` to `source_problem_id`;
- multiline `solution` to a key candidate only after set-play/try handling is specified.

The focused conversion components are:

```text
src/data/heldout_sources/yacpdb_position.py
src/data/heldout_sources/yacpdb_solution.py
```

## Algebraic Position Normalization

Real YACPDB records encode orthodox positions as piece lists:

```yaml
algebraic:
  white: [Kc6, Qf3, Sb5, Pa3]
  black: [Kc4, Se4, Pd4]
```

Supported grammar:

- standard 8x8 board only;
- `K`, `Q`, `R`, `B`, `S`/`N`, `P`;
- square syntax `[a-h][1-8]`;
- one list for `white`, one list for `black`.

Rejected:

- fairy pieces or specifications;
- neutral pieces;
- malformed squares;
- duplicate occupancy;
- missing kings;
- unsupported non-directmate stipulations.

Historical-state policy:

- side to move is White for orthodox directmates `#1`..`#10`;
- castling rights are set to none because YACPDB algebraic lists do not provide them;
- en-passant is set to none because it is not represented;
- halfmove clock is `0`;
- fullmove number is `1`.

If a key requires castling or en-passant state that the source does not provide, the record is rejected.

## Structural Key Extraction

The unsafe rule "first move-like token = key" has been removed for real YACPDB records.

The structural parser:

- ignores `1...` set-play/defense lines;
- skips solver-side tries marked with `?` or refuted with `but`;
- identifies exactly one remaining solver-side `1.` key line;
- strips source comments in `{...}` for classification only;
- resolves the identified token against the starting board;
- validates legal move membership.

Supported key notation:

- Popeye coordinate moves, for example `Qf3-d1`;
- captures with `x` or `*`;
- explicit promotions such as `a7-a8=Q`;
- castling only when legal historical rights are present in an existing FEN source;
- SAN fallback for unambiguous orthodox moves.

Unsupported or rejected:

- ambiguous multiple actual keys;
- try-only solutions;
- unsupported notation;
- illegal structurally identified keys;
- capture marker mismatch;
- castling without source-supported castling rights.

## Rejection Taxonomy

Important deterministic rejection reasons include:

- `UNSUPPORTED_NON_DIRECTMATE`
- `UNSUPPORTED_STIPULATION`
- `UNSUPPORTED_FAIRY_CONDITION`
- `UNSUPPORTED_TWIN`
- `UNSUPPORTED_ALGEBRAIC_POSITION`
- `FAIRY_PIECE`
- `DUPLICATE_SQUARE`
- `MISSING_KING`
- `SIDE_TO_MOVE_UNKNOWN`
- `MALFORMED_POSITION`
- `NO_SOLUTION`
- `NO_ACTUAL_KEY`
- `TRY_ONLY`
- `AMBIGUOUS_KEY`
- `UNSUPPORTED_MOVE_NOTATION`
- `KEY_NOT_LEGAL`
- `CAPTURE_MARKER_MISMATCH`
- `UNKNOWN_CASTLING_RIGHTS`

## Normalization Smoke Report

See:

```text
generic_info/yacpdb_normalization_smoke.md
```

The pass inspected 8 unique real YACPDB IDs and accepted directmate examples `26026`, `297`, `36411`, `4`, and `49265`. It did not run the full `#1`..`#10` availability scan.

## Key Move Extraction

The YACPDB importer extracts only the first move of the source solution, called the key move.

Supported key syntaxes:

- Popeye coordinate style such as `1.Qg7-f8`;
- capture forms such as `Qh1*f3` or `Qh1xf3`;
- promotions such as `a7-a8=Q`;
- castling `O-O`, `O-O-O`, `0-0`, `0-0-0`;
- SAN accepted by `python-chess` when unambiguous and legal.

The key is validated against `chess.Board(fen).legal_moves`. The extractor does not use target moves, model outputs, engines, or chess correctness beyond board legality.

## Solution Tree Handling

YACPDB solutions can contain variations, threats, tries, and prose. Version 1 does not normalize the full solution tree.

Canonical output therefore uses:

- `key_move_uci`: validated first move;
- `source_solution_raw`: original source solution text;
- `solution_tree`: empty/null until a reviewed tree parser exists;
- `principal_line_uci`: empty/null until a reviewed principal-line normalizer exists;
- `solution_uci`: empty for YACPDB key-only records.

This prevents accidental overclaiming: key validation is not the same as full forced-mate verification.

## Validation Status

YACPDB rows built by this importer use:

| Field | Value |
|---|---|
| `validation_status` | `KEY_VALIDATED` |
| `key_validation_status` | `KEY_VALIDATED` |
| `line_validation_status` | `NOT_NORMALIZED` |
| `forced_mate_verification_status` | `NOT_VERIFIED_ENGINE_NOT_USED` |
| `lifecycle_status` in manifest | `VALIDATED` |
| `license_status` in manifest | `UNCLEAR` unless externally resolved |

The manifest is not a freeze certificate. Freezing requires separate review of provenance, license, counts, contamination, and fingerprints.

## Commands

Discovery over a local export/cache:

```bash
./venv/bin/python -m src.data.heldout_sources.yacpdb \
  --mode discover \
  --source data/heldout_classic/raw/yacpdb/<export>.jsonl \
  --output-root data/heldout_classic
```

Availability scan:

```bash
./venv/bin/python -m src.data.heldout_sources.yacpdb \
  --mode availability \
  --source data/heldout_classic/raw/yacpdb/<export>.jsonl \
  --output-root data/heldout_classic
```

Build a validated-not-frozen candidate CSV:

```bash
./venv/bin/python -m src.data.heldout_sources.yacpdb \
  --mode build \
  --source data/heldout_classic/raw/yacpdb/<export>.jsonl \
  --output-root data/heldout_classic \
  --dataset-version yacpdb_v1 \
  --per-depth 20 \
  --seed 42
```

## Outputs

```text
data/heldout_classic/raw/yacpdb/discovery_manifest.json
data/heldout_classic/raw/yacpdb/smoke/*.json
data/heldout_classic/raw/yacpdb/smoke/*.metadata.json
data/heldout_classic/processed/yacpdb_availability.json
data/heldout_classic/processed/yacpdb_rejected_records.json
data/heldout_classic/processed/yacpdb_manual_review.json
data/heldout_classic/final/heldout_classic.csv
data/heldout_classic/final/manifest.json
```

## Manual Review

`yacpdb_manual_review.json` provides deterministic examples per MateDepth with:

- source problem ID;
- composer/reference;
- FEN;
- ASCII board;
- extracted key move;
- raw source solution;
- validation statuses.

These examples are intended for human provenance and notation review before any freeze.
