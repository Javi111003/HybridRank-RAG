"""
Phase 2: Norma Linker — Extract relationships between normas.

Reads the Phase 1 SQLite DB, scans norma texts for cross-references
(especially in DISPOSICIONES FINALES sections), resolves them against
the norma index, and stores relationships + unresolved references.
"""

import argparse
import json
import logging
import re
import sqlite3
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from src.data_preparation.norma_models import (
    LinkingResult,
    NormaReference,
    NormaRelationship,
    normalize_numero,
    normalize_tipo,
)

logger = logging.getLogger(__name__)


# ── Regex patterns ───────────────────────────────────────────────────────────

DISPOSICIONES_PATTERN = re.compile(
    r'DISPOSICION(?:ES)?\s+FINAL(?:ES)?',
    re.IGNORECASE,
)

# Reference pattern: captures norma type, optional "No.", number, optional date, year, optional organismo
REFERENCE_PATTERN = re.compile(
    r'(?P<tipo>'
    r'Resoluci[oó]n\s+Conjunta'
    r'|Instrucci[oó]n\s+Conjunta'
    r'|Decreto[\s-]*Presidencial'
    r'|Decreto[\s-]*Ley'
    r'|Resoluci[oó]n'
    r'|Instrucci[oó]n'
    r'|Acuerdo'
    r'|Decreto'
    r'|Ley'
    r')\s+'
    r'(?:No\.?\s*)?(?P<numero>[A-Za-z0-9/._-]+)\s+'
    r'(?:de\s+(?:\d{1,2})\s+de\s+\w+\s+)?'
    r'de\s+(?P<year>\d{4})'
    r'(?:\s*,?\s*(?:del?|emitid[ao]\s+por)\s+(?P<organismo>[^,;.()\n]+?))?'
    r'(?=\s+y\s+(?:el|la|los|las)\s+|[,;.]|\s+(?:Resoluci|Decreto|Ley\s|Acuerdo|Instrucci)|$)',
    re.IGNORECASE,
)

# Verb patterns for relation type classification
DEROGA_VERBS = re.compile(
    r'se\s+derog[aó]|queda[n]?\s+derogad[oa]s?|dejar\s+sin\s+efecto|'
    r'deja[n]?\s+sin\s+efecto|queda\s+sin\s+efecto|sin\s+vigor',
    re.IGNORECASE,
)

MODIFICA_VERBS = re.compile(
    r'se\s+modific[aó]|queda[n]?\s+modificad[oa]s?|se\s+adicion[aó]n?|'
    r'se\s+sustituye[n]?|se\s+reemplaz[aó]|se\s+suprim[eó]|'
    r'queda[n]?\s+redactad[oa]s?',
    re.IGNORECASE,
)

COMPLEMENTA_VERBS = re.compile(
    r'complement[aó]r?|en\s+lo\s+que\s+complementa',
    re.IGNORECASE,
)

# Action-verb window: finds an action verb followed (within 200 chars) by a norma
# type + number, but WITHOUT requiring "de YEAR".
# Used only for loose references.
ACTION_VERB_PATTERN = re.compile(
    r'(?P<verb>'
    r'se\s+derog[aó]n?|queda[n]?\s+derogad[oa]s?|dejar?\s+sin\s+efecto[s]?|'
    r'se\s+modific[aó]n?|queda[n]?\s+modificad[oa]s?|se\s+adicion[aó]n?|'
    r'se\s+sustituye[n]?|se\s+suprim[eó]n?|queda[n]?\s+redactad[oa]s?'
    r')',
    re.IGNORECASE,
)

# Loose reference: tipo + numero, no year required.
# Anchored after an action verb via the extraction function.
LOOSE_REFERENCE_PATTERN = re.compile(
    r'(?P<tipo>'
    r'Resoluci[oó]n(?:es)?\s+Conjunta(?:s)?'
    r'|Instrucci[oó]n(?:es)?\s+Conjunta(?:s)?'
    r'|Decreto(?:s)?[\s-]*Presidencial(?:es)?'
    r'|Decreto(?:s)?[\s-]*Ley(?:es)?'
    r'|Resoluci[oó]n(?:es)?'
    r'|Instrucci[oó]n(?:es)?'
    r'|Acuerdo(?:s)?'
    r'|Decreto(?:s)?'
    r'|Ley(?:es)?'
    r')\s+'
    r'(?:No\.?\s*)?(?P<numero>[A-Za-z0-9/_-]+)',
    re.IGNORECASE,
)


# ── Organismo normalization ──────────────────────────────────────────────────

STOP_WORDS = frozenset({
    'de', 'del', 'la', 'las', 'los', 'el', 'y', 'e', 'a', 'en', 'por',
    'para', 'con', 'su', 'sus', 'al',
})


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def normalize_organismo(organismo: str) -> str:
    """Normalize an organismo string for comparison."""
    if not organismo:
        return ''
    text = _strip_accents(organismo.lower().strip())
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def fuzzy_organismo_match(ref_organismo: str, db_organismo: str) -> bool:
    """Check if two organismo strings refer to the same entity.

    Handles cases like 'ministra de Finanzas y Precios' matching
    'Ministerio de Finanzas y Precios'.
    """
    ref = normalize_organismo(ref_organismo)
    db = normalize_organismo(db_organismo)
    if not ref or not db:
        return False
    ref_tokens = set(ref.split()) - STOP_WORDS
    db_tokens = set(db.split()) - STOP_WORDS
    if not ref_tokens or not db_tokens:
        return False
    overlap = len(ref_tokens & db_tokens) / min(len(ref_tokens), len(db_tokens))
    return overlap >= 0.6


# ── Section extraction ───────────────────────────────────────────────────────

def extract_disposiciones_section(text: str) -> Optional[str]:
    """Extract the DISPOSICIONES FINALES section from norma text.

    Returns the text from the DISPOSICIONES header to the end,
    or None if no such section exists.
    """
    if not text:
        return None
    m = DISPOSICIONES_PATTERN.search(text)
    if not m:
        return None
    return text[m.start():]


# ── Relation type classification ─────────────────────────────────────────────

def classify_relation_type(context: str) -> str:
    """Classify the relation type from surrounding context text.

    Priority: deroga > modifica > complementa > menciona
    """
    if DEROGA_VERBS.search(context):
        return 'deroga'
    if MODIFICA_VERBS.search(context):
        return 'modifica'
    if COMPLEMENTA_VERBS.search(context):
        return 'complementa'
    return 'menciona'


# ── Reference extraction ─────────────────────────────────────────────────────

def _get_context_window(text: str, match_start: int, match_end: int,
                        window: int = 200) -> str:
    """Get surrounding text for a regex match."""
    start = max(0, match_start - window)
    end = min(len(text), match_end + window)
    return text[start:end]


def _build_ref_key(tipo: str, numero: str, year: int) -> Tuple[str, str, int]:
    """Build a deduplication key for a reference."""
    return (normalize_tipo(tipo), normalize_numero(numero), year)


def extract_references(source_norma_id: str, text: str) -> List[NormaReference]:
    """Extract all cross-references from a norma's text.

    Scans the full text. References within the DISPOSICIONES FINALES section
    get their relation type from verb analysis. References elsewhere get
    verb analysis too but default to 'menciona' when no action verb is found.
    """
    if not text:
        return []

    # Determine DISPOSICIONES section boundary (if any)
    disp_match = DISPOSICIONES_PATTERN.search(text)
    disp_start = disp_match.start() if disp_match else None

    refs = []
    seen_keys: Set[Tuple[str, str, int]] = set()

    for m in REFERENCE_PATTERN.finditer(text):
        tipo_raw = re.sub(r'[\s-]+', ' ', m.group('tipo')).strip()
        numero_raw = m.group('numero').strip()
        year = int(m.group('year'))
        organismo = m.group('organismo')
        if organismo:
            organismo = organismo.strip()

        # Normalize numero for dedup and matching
        numero_norm = normalize_numero(numero_raw)

        # Deduplication
        ref_key = _build_ref_key(tipo_raw, numero_norm, year)
        if ref_key in seen_keys:
            continue
        seen_keys.add(ref_key)

        # Self-reference check: build a tentative norma_id and compare
        tipo_norm = normalize_tipo(tipo_raw)
        tentative_id_prefix = f"{tipo_norm.replace(' ', '_')}_{numero_norm}_{year}_"
        if source_norma_id.startswith(tentative_id_prefix):
            continue

        # Classify relation type from context
        context = _get_context_window(text, m.start(), m.end(), window=200)
        in_disposiciones = disp_start is not None and m.start() >= disp_start
        if in_disposiciones:
            relation = classify_relation_type(context)
        else:
            # Outside DISPOSICIONES: still check for action verbs
            relation = classify_relation_type(context)
            # If no verb found, it's just a mention
            # classify_relation_type already returns 'menciona' as default

        refs.append(NormaReference(
            source_norma_id=source_norma_id,
            ref_tipo=tipo_raw,
            ref_numero=numero_norm,
            ref_year=year,
            ref_organismo=organismo,
            relation_type=relation,
            raw_text=m.group(0),
            context_text=context,
        ))

    return refs


def extract_loose_references(
    source_norma_id: str,
    text: str,
    already_found: Set[Tuple[str, str]] = None,
) -> List[NormaReference]:
    """Extract references WITHOUT year, gated by nearby action verbs.

    Only captures tipo+numero when an action verb (deroga/modifica/etc.) appears
    within 200 chars before the reference. These are stored with ref_year=None.

    `already_found` is a set of (tipo_norm, numero_norm) already captured by
    extract_references (with year) to avoid duplicates.
    """
    if not text:
        return []

    already = already_found or set()
    refs = []
    seen_keys: Set[Tuple[str, str]] = set()

    for verb_match in ACTION_VERB_PATTERN.finditer(text):
        # Search for norma references in the 200 chars after the verb
        search_start = verb_match.start()
        search_end = min(len(text), verb_match.end() + 200)
        window = text[search_start:search_end]

        for m in LOOSE_REFERENCE_PATTERN.finditer(window):
            tipo_raw = re.sub(r'[\s-]+', ' ', m.group('tipo')).strip()
            numero_raw = m.group('numero').strip()

            # Skip if numero looks like a non-reference (articles, etc.)
            # Valid numeros: digits, X-prefix, V-prefix, s/n
            if not re.match(r'^[A-Za-z]?-?\d+(?:/\d{4})?$', numero_raw):
                continue

            numero_norm = normalize_numero(numero_raw)
            tipo_norm = normalize_tipo(tipo_raw)

            # Collect all numeros: the primary + any "y NUMERO" / ", NUMERO" continuations
            # e.g., "Acuerdos 5204 y 5209" → [('5204', raw1), ('5209', raw2)]
            all_numeros = [(numero_norm, m.group(0))]
            continuation_text = window[m.end():]
            for cont in re.finditer(
                r'\s*(?:,\s*|y\s+)(?:No\.?\s*)?(\d+[A-Za-z0-9/_-]*)',
                continuation_text,
            ):
                cont_num = normalize_numero(cont.group(1).strip())
                all_numeros.append((cont_num, f"{tipo_raw} {cont.group(1).strip()}"))
                # Stop if there's a gap (non-continuation text) after this match
                if cont.end() < len(continuation_text) and not re.match(
                    r'\s*(?:,|y\s)', continuation_text[cont.end():]
                ):
                    break

            for num, raw in all_numeros:
                # Skip if already captured with year by extract_references
                if (tipo_norm, num) in already:
                    continue

                # Dedup within loose refs
                loose_key = (tipo_norm, num)
                if loose_key in seen_keys:
                    continue
                seen_keys.add(loose_key)

                # Self-reference check
                if source_norma_id.startswith(f"{tipo_norm.replace(' ', '_')}_{num}_"):
                    continue

                # Classify relation from verb context
                context = _get_context_window(text, search_start + m.start(),
                                              search_start + m.end(), window=200)
                relation = classify_relation_type(context)

                refs.append(NormaReference(
                    source_norma_id=source_norma_id,
                    ref_tipo=tipo_raw,
                    ref_numero=num,
                    ref_year=None,
                    ref_organismo=None,
                    relation_type=relation,
                    raw_text=raw,
                    context_text=context,
                ))

    return refs


# ── Index building ───────────────────────────────────────────────────────────

def build_norma_index(conn: sqlite3.Connection) -> Dict:
    """Build lookup indices from Phase 1 SQLite DB.

    Returns a dict with:
        'lookup': {(tipo_norm, numero_norm, year): [norma_id, ...]}
        'loose_lookup': {(tipo_norm, numero_norm): [norma_id, ...]}  # ignores year
        'organismos': {norma_id: organismo_emisor}
        'goc_codes': {norma_id: goc_code}
    """
    cursor = conn.execute(
        "SELECT norma_id, tipo, numero, year, organismo_emisor, goc_code FROM normas"
    )
    lookup = defaultdict(list)
    loose_lookup = defaultdict(list)
    organismos = {}
    goc_codes = {}

    for norma_id, tipo, numero, year, organismo, goc_code in cursor:
        key = (normalize_tipo(tipo), normalize_numero(numero), year)
        lookup[key].append(norma_id)
        loose_key = (normalize_tipo(tipo), normalize_numero(numero))
        loose_lookup[loose_key].append(norma_id)
        organismos[norma_id] = organismo
        goc_codes[norma_id] = goc_code

    return {
        'lookup': dict(lookup),
        'loose_lookup': dict(loose_lookup),
        'organismos': organismos,
        'goc_codes': goc_codes,
    }


# ── Resolution ───────────────────────────────────────────────────────────────

def resolve_references(
    refs: List[NormaReference],
    index: Dict,
    source_organismos: Optional[Dict[str, str]] = None,
) -> List[NormaReference]:
    """Resolve references against the norma index.

    Resolution tiers:
    1. Exact: single match for (tipo, numero, year) → 'exact'
    2. Fuzzy organismo: multiple matches, ref_organismo disambiguates → 'fuzzy'
    3. Source organismo: multiple matches, source norma's own organismo
       matches exactly one candidate → 'source_org'
    4. Ambiguous: multiple matches, cannot disambiguate → 'ambiguous'
    5. Unresolved: no match in DB → 'unresolved'

    source_organismos: {norma_id: organismo_emisor} for the source normas,
    used for the source-organismo heuristic.
    """
    lookup = index['lookup']
    organismos = index['organismos']
    src_orgs = source_organismos or {}

    for ref in refs:
        key = (normalize_tipo(ref.ref_tipo), ref.ref_numero, ref.ref_year)
        candidates = lookup.get(key, [])

        if len(candidates) == 0:
            ref.confidence = 'unresolved'
            ref.resolved_norma_id = None

        elif len(candidates) == 1:
            ref.confidence = 'exact'
            ref.resolved_norma_id = candidates[0]

        else:
            # Multiple candidates — try organismo disambiguation
            resolved = False

            # Tier 2: explicit ref_organismo
            if ref.ref_organismo:
                for cand in candidates:
                    db_org = organismos.get(cand, '')
                    if fuzzy_organismo_match(ref.ref_organismo, db_org):
                        ref.confidence = 'fuzzy'
                        ref.resolved_norma_id = cand
                        resolved = True
                        break

            # Tier 3: source norma's own organismo
            if not resolved:
                src_org = src_orgs.get(ref.source_norma_id, '')
                if src_org:
                    matching = [c for c in candidates
                                if fuzzy_organismo_match(src_org, organismos.get(c, ''))]
                    if len(matching) == 1:
                        ref.confidence = 'source_org'
                        ref.resolved_norma_id = matching[0]
                        resolved = True

            if not resolved:
                ref.confidence = 'ambiguous'
                ref.resolved_norma_id = None

    return refs


def resolve_loose_references(
    refs: List[NormaReference],
    index: Dict,
    source_organismos: Optional[Dict[str, str]] = None,
) -> List[NormaReference]:
    """Resolve loose references (no year) against the norma index.

    Uses loose_lookup (tipo, numero) ignoring year. Same resolution tiers
    as resolve_references but searching across all years.
    """
    loose_lookup = index.get('loose_lookup', {})
    organismos = index['organismos']
    src_orgs = source_organismos or {}

    for ref in refs:
        key = (normalize_tipo(ref.ref_tipo), ref.ref_numero)
        candidates = loose_lookup.get(key, [])

        # Filter out self-references
        candidates = [c for c in candidates
                      if not ref.source_norma_id == c]

        if len(candidates) == 0:
            ref.confidence = 'unresolved'
            ref.resolved_norma_id = None

        elif len(candidates) == 1:
            ref.confidence = 'exact_loose'
            ref.resolved_norma_id = candidates[0]

        else:
            # Try source organismo heuristic
            resolved = False
            src_org = src_orgs.get(ref.source_norma_id, '')
            if src_org:
                matching = [c for c in candidates
                            if fuzzy_organismo_match(src_org, organismos.get(c, ''))]
                if len(matching) == 1:
                    ref.confidence = 'source_org_loose'
                    ref.resolved_norma_id = matching[0]
                    resolved = True

            if not resolved:
                ref.confidence = 'ambiguous'
                ref.resolved_norma_id = None

    return refs


# ── Main pipeline ────────────────────────────────────────────────────────────

def link_all_normas(conn: sqlite3.Connection) -> LinkingResult:
    """Run the full linking pipeline on all normas in the DB."""
    index = build_norma_index(conn)

    # Load all norma texts + build source organismo map
    cursor = conn.execute("SELECT norma_id, raw_text, goc_code, organismo_emisor FROM normas")
    norma_rows = cursor.fetchall()
    source_organismos = {nid: org for nid, _, _, org in norma_rows}

    # Phase A: extract references WITH year
    all_refs: List[NormaReference] = []
    per_norma_found: Dict[str, Set[Tuple[str, str]]] = {}  # track for loose dedup

    for norma_id, raw_text, goc_code, _ in norma_rows:
        refs = extract_references(norma_id, raw_text)
        all_refs.extend(refs)
        # Track (tipo_norm, numero_norm) found per norma for loose dedup
        per_norma_found[norma_id] = {
            (normalize_tipo(r.ref_tipo), r.ref_numero) for r in refs
        }

    logger.info("Extracted %d year-refs from %d normas", len(all_refs), len(norma_rows))

    # Phase B: extract LOOSE references (no year, action-verb gated)
    loose_refs: List[NormaReference] = []
    for norma_id, raw_text, goc_code, _ in norma_rows:
        lrefs = extract_loose_references(
            norma_id, raw_text,
            already_found=per_norma_found.get(norma_id, set()),
        )
        loose_refs.extend(lrefs)

    logger.info("Extracted %d loose refs (no year)", len(loose_refs))

    # Resolve both sets
    all_refs = resolve_references(all_refs, index, source_organismos)
    loose_refs = resolve_loose_references(loose_refs, index, source_organismos)

    # Combine
    combined = all_refs + loose_refs

    # Build relationships from resolved refs
    RESOLVED_CONFIDENCES = ('exact', 'fuzzy', 'source_org', 'exact_loose', 'source_org_loose')

    relationships = []
    unresolved = []
    counts: Dict[str, int] = defaultdict(int)

    for ref in combined:
        if ref.confidence in RESOLVED_CONFIDENCES:
            rel = NormaRelationship(
                source_norma_id=ref.source_norma_id,
                target_norma_id=ref.resolved_norma_id,
                relation_type=ref.relation_type,
                source_goc_code=index['goc_codes'].get(ref.source_norma_id, ''),
                target_goc_code=index['goc_codes'].get(ref.resolved_norma_id, ''),
                raw_reference_text=ref.raw_text,
                resolution_method=ref.confidence,
            )
            relationships.append(rel)
        else:
            unresolved.append(ref)
        counts[ref.confidence] += 1

    n_resolved = sum(counts[c] for c in RESOLVED_CONFIDENCES)
    n_ambiguous = counts.get('ambiguous', 0)
    n_unresolved = counts.get('unresolved', 0)

    logger.info(
        "Resolution: %d resolved (%s), %d ambiguous, %d unresolved",
        n_resolved,
        ", ".join(f"{c}={counts[c]}" for c in RESOLVED_CONFIDENCES if counts.get(c)),
        n_ambiguous, n_unresolved,
    )

    return LinkingResult(
        relationships=relationships,
        unresolved_references=unresolved,
        total_references_found=len(combined),
        total_resolved=n_resolved,
        total_ambiguous=n_ambiguous,
        total_unresolved=n_unresolved,
    )


# ── SQLite output ────────────────────────────────────────────────────────────

_LINKING_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS norma_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_norma_id TEXT NOT NULL,
    target_norma_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    source_goc_code TEXT,
    target_goc_code TEXT,
    raw_reference_text TEXT,
    resolution_method TEXT DEFAULT 'exact',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS norma_unresolved_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_norma_id TEXT NOT NULL,
    ref_tipo TEXT NOT NULL,
    ref_numero TEXT NOT NULL,
    ref_year INTEGER,
    ref_organismo TEXT,
    relation_type TEXT NOT NULL,
    raw_text TEXT,
    context_text TEXT,
    confidence TEXT DEFAULT 'unresolved'
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON norma_relationships(source_norma_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON norma_relationships(target_norma_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON norma_relationships(relation_type);
CREATE INDEX IF NOT EXISTS idx_unresolved_source ON norma_unresolved_references(source_norma_id);
"""


def save_relationships_sqlite(result: LinkingResult, conn: sqlite3.Connection) -> None:
    """Save linking results to SQLite (extends existing Phase 1 DB)."""
    conn.executescript(_LINKING_SCHEMA_SQL)

    # Clear previous results (idempotent re-runs)
    conn.execute("DELETE FROM norma_relationships")
    conn.execute("DELETE FROM norma_unresolved_references")

    for rel in result.relationships:
        conn.execute(
            "INSERT INTO norma_relationships "
            "(source_norma_id, target_norma_id, relation_type, source_goc_code, "
            "target_goc_code, raw_reference_text, resolution_method) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (rel.source_norma_id, rel.target_norma_id, rel.relation_type,
             rel.source_goc_code, rel.target_goc_code, rel.raw_reference_text,
             rel.resolution_method),
        )

    for ref in result.unresolved_references:
        conn.execute(
            "INSERT INTO norma_unresolved_references "
            "(source_norma_id, ref_tipo, ref_numero, ref_year, ref_organismo, "
            "relation_type, raw_text, context_text, confidence) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ref.source_norma_id, ref.ref_tipo, ref.ref_numero, ref.ref_year,
             ref.ref_organismo, ref.relation_type, ref.raw_text,
             ref.context_text, ref.confidence),
        )

    conn.commit()
    logger.info(
        "Saved %d relationships and %d unresolved references to SQLite",
        len(result.relationships), len(result.unresolved_references),
    )


# ── JSON report ──────────────────────────────────────────────────────────────

def save_linking_report(result: LinkingResult, output_path: str) -> None:
    """Save a linking report as JSON."""
    # Compute distributions
    rel_type_dist = defaultdict(int)
    for rel in result.relationships:
        rel_type_dist[rel.relation_type] += 1

    method_dist = defaultdict(int)
    for rel in result.relationships:
        method_dist[rel.resolution_method] += 1

    unresolved_confidence_dist = defaultdict(int)
    for ref in result.unresolved_references:
        unresolved_confidence_dist[ref.confidence] += 1

    report = {
        'stats': result.to_dict()['stats'],
        'relation_type_distribution': dict(rel_type_dist),
        'resolution_method_distribution': dict(method_dist),
        'unresolved_confidence_distribution': dict(unresolved_confidence_dist),
        'unresolved_references': [r.to_dict() for r in result.unresolved_references],
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("Linking report written to %s", output_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

import os


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Extract norma relationships from Phase 1 DB."
    )
    parser.add_argument(
        '--db-input',
        default='.data/norma_output/normas.db',
        help='Path to Phase 1 SQLite DB (default: .data/norma_output/normas.db)',
    )
    parser.add_argument(
        '--report-output',
        default='.data/norma_output/linking_report.json',
        help='Path for JSON linking report (default: .data/norma_output/linking_report.json)',
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    if not os.path.exists(args.db_input):
        logger.error("DB file not found: %s", args.db_input)
        return

    logger.info("Opening DB: %s", args.db_input)
    conn = sqlite3.connect(args.db_input)

    try:
        result = link_all_normas(conn)
        save_relationships_sqlite(result, conn)
        save_linking_report(result, args.report_output)

        # Print summary
        method_counts = defaultdict(int)
        for r in result.relationships:
            method_counts[r.resolution_method] += 1

        print(f"References found:    {result.total_references_found}")
        print(f"Resolved:            {result.total_resolved}")
        for method, count in sorted(method_counts.items()):
            print(f"  - {method:18s} {count}")
        print(f"Ambiguous:           {result.total_ambiguous}")
        print(f"Unresolved:          {result.total_unresolved}")
        print(f"Relationships:       {len(result.relationships)}")

    finally:
        conn.close()


if __name__ == '__main__':
    main()
