"""
Norma Processor: segments and structures legal norms from Gaceta Oficial chunks.

Reads the output of text_cleaner (cleaned_elements.json), groups chunks by gaceta,
reconstructs full text, segments by norma using GOC codes, and outputs structured
JSON + SQLite.
"""

import bisect
import json
import logging
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import ijson
except ImportError:
    ijson = None

try:
    from src.data_preparation.norma_models import (
        GOC_CODE_PATTERN,
        Gaceta,
        Norma,
        NormaIdentity,
        ProcessingResult,
        extract_header_identity,
        normalize_numero,
        normalize_tipo,
        parse_norma_name,
    )
except ModuleNotFoundError:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from src.data_preparation.norma_models import (
        GOC_CODE_PATTERN,
        Gaceta,
        Norma,
        NormaIdentity,
        ProcessingResult,
        extract_header_identity,
        normalize_numero,
        normalize_tipo,
        parse_norma_name,
    )

logger = logging.getLogger(__name__)

# Spanish month names for date parsing
_MESES = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
    'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
    'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12,
}


# Loading

def _load_elements(path: str) -> List[Dict[str, Any]]:
    """Load elements from JSON, streaming via ijson if available."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    if ijson is not None:
        logger.info("Loading elements via streaming from: %s", path)
        elements = []
        with open(path, 'rb') as f:
            for item in ijson.items(f, 'item'):
                if isinstance(item, dict):
                    elements.append(item)
        logger.info("Loaded %d elements (streaming)", len(elements))
        return elements

    logger.info("Loading elements from: %s", path)
    with open(path, 'r', encoding='utf-8') as f:
        elements = json.load(f)
    logger.info("Loaded %d elements", len(elements))
    return elements


# Step 1: Group chunks by gaceta

def group_chunks_by_gaceta(
    elements: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group chunks by their gaceta, keyed by checksum or numero_fecha."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    skipped = 0

    for element in elements:
        meta = element.get('metadata', {})
        checksum = meta.get('gaceta_checksum', '')
        if checksum:
            key = checksum
        else:
            numero = meta.get('gaceta_numero', '')
            fecha = meta.get('gaceta_fecha', '')
            if not numero and not fecha:
                skipped += 1
                continue
            key = f"{numero}_{fecha}"
        groups[key].append(element)

    if skipped:
        logger.warning("Skipped %d chunks with no gaceta metadata", skipped)
    logger.info("Grouped %d chunks into %d gacetas", len(elements) - skipped, len(groups))
    return dict(groups)


# Step 2: Concatenate chunks into full gaceta text

def concatenate_gaceta_text(
    chunks: List[Dict[str, Any]],
) -> Tuple[str, List[Tuple[int, int]]]:
    """Sort chunks by page number and concatenate raw content.

    Returns (full_text, page_boundaries) where page_boundaries is a list of
    (char_offset, page_number) tuples for mapping offsets back to pages.
    """
    sorted_chunks = sorted(
        chunks,
        key=lambda c: c.get('metadata', {}).get('page_number') or 0,
    )

    parts: List[str] = []
    page_boundaries: List[Tuple[int, int]] = []
    offset = 0

    for chunk in sorted_chunks:
        page = chunk.get('metadata', {}).get('page_number') or 0
        content = chunk.get('content', '')
        page_boundaries.append((offset, page))
        parts.append(content)
        offset += len(content) + 1  # +1 for the \n separator

    full_text = '\n'.join(parts)
    return full_text, page_boundaries


def _offset_to_page(offset: int, page_boundaries: List[Tuple[int, int]]) -> int:
    """Map a character offset to a page number using bisect."""
    if not page_boundaries:
        return 0
    offsets = [pb[0] for pb in page_boundaries]
    idx = bisect.bisect_right(offsets, offset) - 1
    if idx < 0:
        return page_boundaries[0][1] if page_boundaries else 0
    return page_boundaries[idx][1]


# Step 3: Segment by GOC codes

def _is_sumario_goc_match(full_text: str, match: re.Match) -> bool:
    """Check if a GOC code match is inside a SUMARIO (table of contents).

    SUMARIO GOC codes are typically followed by ')' (in parenthetical references)
    or preceded by '(' e.g., "(GOC-2026-215-O24)...........page_number".
    """
    end_pos = match.end()
    # Check characters after the GOC code
    after = full_text[end_pos:end_pos + 5].lstrip()
    if after.startswith(')'):
        return True
    # Check if preceded by '('
    start_pos = match.start()
    before = full_text[max(0, start_pos - 3):start_pos].rstrip()
    if before.endswith('('):
        return True
    return False


def segment_by_goc_codes(
    full_text: str,
    page_boundaries: List[Tuple[int, int]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Split text at GOC code positions.

    Filters out GOC codes that appear inside SUMARIO (table of contents)
    sections, identified by being inside parentheses.

    Returns (sumario_text, segments) where each segment is a dict with
    goc_code, text, page_range, ordinal_position.
    """
    all_matches = list(GOC_CODE_PATTERN.finditer(full_text))

    # Filter out SUMARIO GOC codes
    matches = [m for m in all_matches if not _is_sumario_goc_match(full_text, m)]

    # Deduplicate: if same GOC code appears multiple times, keep only the first
    # body occurrence (not the SUMARIO one)
    seen_codes: Dict[str, int] = {}
    unique_matches: List[re.Match] = []
    for m in matches:
        code = m.group(1).upper()
        if code not in seen_codes:
            seen_codes[code] = len(unique_matches)
            unique_matches.append(m)
    matches = unique_matches

    if not matches:
        return full_text, []

    sumario_text = full_text[:matches[0].start()].strip()
    segments: List[Dict[str, Any]] = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        text = full_text[start:end].strip()
        goc_code = match.group(1)

        page_start = _offset_to_page(start, page_boundaries)
        page_end = _offset_to_page(end - 1, page_boundaries)

        segments.append({
            'goc_code': goc_code,
            'text': text,
            'page_range': (page_start, page_end),
            'ordinal_position': i,
        })

    return sumario_text, segments


# Step 4-5: Match segments to norma metadata

def match_segments_to_normas(
    segments: List[Dict[str, Any]],
    parsed_normas: List[NormaIdentity],
) -> Tuple[List[Norma], List[Dict[str, Any]]]:
    """Match GOC-delimited text segments to parsed norma identities.

    Returns (matched_normas, unmatched_segments).
    """
    available = list(parsed_normas)  # mutable copy
    matched: List[Norma] = []
    unmatched: List[Dict[str, Any]] = []

    # Phase A: Match by tipo + numero (high confidence)
    remaining_segments: List[Dict[str, Any]] = []
    for seg in segments:
        header = extract_header_identity(seg['text'])
        if header is None:
            remaining_segments.append(seg)
            continue

        seg_tipo, seg_numero = header
        best_idx = None
        for i, identity in enumerate(available):
            if (normalize_tipo(seg_tipo) == normalize_tipo(identity.tipo)
                    and normalize_numero(seg_numero) == normalize_numero(identity.numero)):
                best_idx = i
                break

        if best_idx is not None:
            identity = available.pop(best_idx)
            matched.append(Norma(
                identity=identity,
                goc_code=seg['goc_code'],
                raw_text=seg['text'],
                page_range=seg['page_range'],
                ordinal_position=seg['ordinal_position'],
                match_confidence='high',
            ))
        else:
            remaining_segments.append(seg)

    # Phase B: Positional matching for s/n normas (medium confidence)
    still_remaining: List[Dict[str, Any]] = []
    if remaining_segments and available:
        # Group remaining available normas by tipo
        sn_normas_by_tipo: Dict[str, List[int]] = defaultdict(list)
        for i, identity in enumerate(available):
            if normalize_numero(identity.numero) == 's/n':
                sn_normas_by_tipo[normalize_tipo(identity.tipo)].append(i)

        matched_indices = set()
        for seg in remaining_segments:
            header = extract_header_identity(seg['text'])
            if header is None:
                still_remaining.append(seg)
                continue

            seg_tipo, _ = header
            tipo_key = normalize_tipo(seg_tipo)
            candidates = sn_normas_by_tipo.get(tipo_key, [])
            # Find first unused candidate
            picked = None
            for idx in candidates:
                if idx not in matched_indices:
                    picked = idx
                    matched_indices.add(idx)
                    break

            if picked is not None:
                identity = available[picked]
                matched.append(Norma(
                    identity=identity,
                    goc_code=seg['goc_code'],
                    raw_text=seg['text'],
                    page_range=seg['page_range'],
                    ordinal_position=seg['ordinal_position'],
                    match_confidence='medium',
                ))
            else:
                still_remaining.append(seg)

        # Remove matched from available (in reverse to preserve indices)
        for idx in sorted(matched_indices, reverse=True):
            available.pop(idx)
        remaining_segments = still_remaining

    # Phase C: Positional fallback — 1:1 if counts match (low confidence)
    if remaining_segments and available and len(remaining_segments) == len(available):
        for seg, identity in zip(remaining_segments, available):
            matched.append(Norma(
                identity=identity,
                goc_code=seg['goc_code'],
                raw_text=seg['text'],
                page_range=seg['page_range'],
                ordinal_position=seg['ordinal_position'],
                match_confidence='low',
            ))
        remaining_segments = []
        available = []

    # Phase D: Unmatched bucket
    for seg in remaining_segments:
        unmatched.append({
            'goc_code': seg['goc_code'],
            'header_extract': seg['text'][:200],
            'text_length': len(seg['text']),
            'page_range': seg['page_range'],
        })

    if available:
        logger.warning(
            "%d norma metadata entries could not be matched to text segments: %s",
            len(available),
            [n.raw_string for n in available],
        )
    if unmatched:
        logger.warning(
            "%d text segments could not be matched to norma metadata",
            len(unmatched),
        )

    # Sort by ordinal position
    matched.sort(key=lambda n: n.ordinal_position)
    return matched, unmatched


# Step 5: Process a single gaceta

def process_single_gaceta(
    grouping_key: str,
    chunks: List[Dict[str, Any]],
) -> Gaceta:
    """Orchestrate processing for one gaceta's chunks into a structured Gaceta."""
    # Extract gaceta-level metadata from first chunk
    first_meta = chunks[0].get('metadata', {})
    norma_strings = first_meta.get('gaceta_normas', []) or []

    gaceta = Gaceta(
        numero=first_meta.get('gaceta_numero', ''),
        fecha=first_meta.get('gaceta_fecha', ''),
        tipo_edicion=first_meta.get('gaceta_tipo_edicion', ''),
        pdf_url=first_meta.get('gaceta_pdf_url', ''),
        checksum=first_meta.get('gaceta_checksum', ''),
        norma_metadata_strings=norma_strings,
    )

    # Parse norma identity from metadata strings
    parsed_normas: List[NormaIdentity] = []
    for s in norma_strings:
        ni = parse_norma_name(s)
        if ni:
            parsed_normas.append(ni)
        else:
            logger.warning("Could not parse norma name: '%s' (gaceta %s/%s)", s, gaceta.numero, gaceta.fecha)

    # Concatenate text
    full_text, page_boundaries = concatenate_gaceta_text(chunks)

    # Segment by GOC codes
    sumario_text, segments = segment_by_goc_codes(full_text, page_boundaries)
    gaceta.sumario_text = sumario_text

    # Special case: no GOC codes found
    if not segments:
        if len(parsed_normas) == 1:
            # Single norma gaceta with no GOC codes — treat full text as that norma
            norma = Norma(
                identity=parsed_normas[0],
                goc_code='',
                raw_text=full_text,
                page_range=(
                    _offset_to_page(0, page_boundaries),
                    _offset_to_page(len(full_text) - 1, page_boundaries) if full_text else 0,
                ),
                ordinal_position=0,
                match_confidence='medium',
            )
            gaceta.normas = [norma]
        elif parsed_normas:
            logger.warning(
                "Gaceta %s/%s: no GOC codes found but %d normas in metadata",
                gaceta.numero, gaceta.fecha, len(parsed_normas),
            )
            gaceta.unmatched_segments = [{
                'goc_code': '',
                'header_extract': full_text[:200],
                'text_length': len(full_text),
                'page_range': (0, 0),
            }]
        return gaceta

    # Match segments to norma metadata
    matched_normas, unmatched_segs = match_segments_to_normas(segments, parsed_normas)
    gaceta.normas = matched_normas
    gaceta.unmatched_segments = unmatched_segs

    return gaceta


# Step 6: Detect cross-gaceta duplicates

_FECHA_PATTERN = re.compile(r'(\d{1,2})\s+(\w+),?\s+(\d{4})')


def _parse_gaceta_fecha(fecha: str) -> Optional[Tuple[int, int, int]]:
    """Parse 'DD Mes, YYYY' into (year, month, day) or None."""
    m = _FECHA_PATTERN.match(fecha.strip())
    if not m:
        return None
    day = int(m.group(1))
    month_name = m.group(2).lower()
    year = int(m.group(3))
    month = _MESES.get(month_name, 0)
    if month == 0:
        return None
    return (year, month, day)


def detect_cross_gaceta_duplicates(
    gacetas: List[Gaceta],
) -> List[Dict[str, Any]]:
    """Find normas that appear in multiple gacetas."""
    norma_occurrences: Dict[str, List[Tuple[Gaceta, Norma]]] = defaultdict(list)

    for gaceta in gacetas:
        for norma in gaceta.normas:
            nid = norma.identity.norma_id
            norma_occurrences[nid].append((gaceta, norma))

    duplicates: List[Dict[str, Any]] = []
    for nid, occurrences in norma_occurrences.items():
        if len(occurrences) < 2:
            continue

        # Sort by fecha to determine which is latest
        dated = []
        for gaceta, norma in occurrences:
            parsed = _parse_gaceta_fecha(gaceta.fecha)
            dated.append((parsed or (0, 0, 0), gaceta, norma))
        dated.sort(key=lambda x: x[0])

        kept = dated[-1]
        superseded = dated[:-1]

        duplicates.append({
            'norma_id': nid,
            'occurrences': [
                {
                    'gaceta_checksum': g.checksum,
                    'gaceta_fecha': g.fecha,
                    'gaceta_numero': g.numero,
                    'goc_code': n.goc_code,
                }
                for _, g, n in dated
            ],
            'kept': {
                'gaceta_checksum': kept[1].checksum,
                'gaceta_fecha': kept[1].fecha,
            },
            'superseded': [
                {
                    'gaceta_checksum': g.checksum,
                    'gaceta_fecha': g.fecha,
                }
                for _, g, _ in superseded
            ],
        })

    if duplicates:
        logger.info("Found %d duplicate norma(s) across gacetas", len(duplicates))
    return duplicates


# Step 7: Full pipeline

def process_all_normas(
    elements: List[Dict[str, Any]],
) -> ProcessingResult:
    """Main pipeline: group, segment, match, dedup."""
    start = time.time()

    groups = group_chunks_by_gaceta(elements)

    gacetas: List[Gaceta] = []
    errors = 0
    for i, (key, chunks) in enumerate(groups.items()):
        try:
            gaceta = process_single_gaceta(key, chunks)
            gacetas.append(gaceta)
        except Exception:
            errors += 1
            logger.exception("Error processing gaceta %s", key)

        if (i + 1) % 100 == 0:
            logger.info("Progress: %d/%d gacetas processed", i + 1, len(groups))

    duplicates = detect_cross_gaceta_duplicates(gacetas)

    total_normas = sum(len(g.normas) for g in gacetas)
    total_unmatched = sum(len(g.unmatched_segments) for g in gacetas)

    elapsed = time.time() - start
    logger.info(
        "Pipeline completed in %.2fs: %d gacetas, %d normas, %d unmatched, %d errors",
        elapsed, len(gacetas), total_normas, total_unmatched, errors,
    )

    return ProcessingResult(
        gacetas=gacetas,
        total_chunks_processed=len(elements),
        total_normas_extracted=total_normas,
        total_unmatched_segments=total_unmatched,
        duplicate_normas=duplicates,
    )


# Output: JSON

def save_json_output(result: ProcessingResult, output_path: str) -> None:
    """Save result as hierarchical JSON with incremental writing."""
    logger.info("Saving JSON output to: %s", output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data = result.to_dict()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info("JSON saved: %d gacetas", len(result.gacetas))


# Output: SQLite

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS gacetas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    numero TEXT NOT NULL,
    fecha TEXT NOT NULL,
    tipo_edicion TEXT NOT NULL,
    pdf_url TEXT,
    checksum TEXT,
    sumario_text TEXT,
    norma_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS normas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    norma_id TEXT NOT NULL,
    tipo TEXT NOT NULL,
    numero TEXT NOT NULL,
    year INTEGER NOT NULL,
    organismo_emisor TEXT NOT NULL,
    goc_code TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    ordinal_position INTEGER DEFAULT 0,
    match_confidence TEXT DEFAULT 'high',
    gaceta_id INTEGER NOT NULL REFERENCES gacetas(id),
    raw_metadata_string TEXT
);

CREATE TABLE IF NOT EXISTS norma_duplicates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    norma_id TEXT NOT NULL,
    kept_gaceta_id INTEGER REFERENCES gacetas(id),
    superseded_gaceta_id INTEGER REFERENCES gacetas(id),
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_normas_norma_id ON normas(norma_id);
CREATE INDEX IF NOT EXISTS idx_normas_tipo ON normas(tipo);
CREATE INDEX IF NOT EXISTS idx_normas_gaceta_id ON normas(gaceta_id);
CREATE INDEX IF NOT EXISTS idx_normas_year ON normas(year);
CREATE INDEX IF NOT EXISTS idx_gacetas_checksum ON gacetas(checksum);
"""


def save_sqlite_output(result: ProcessingResult, db_path: str) -> None:
    """Save result to SQLite database."""
    logger.info("Saving SQLite output to: %s", db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Remove existing DB to start fresh
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_SCHEMA_SQL)

        # Build checksum → row id mapping for duplicates
        checksum_to_gaceta_id: Dict[str, int] = {}

        for gaceta in result.gacetas:
            cursor = conn.execute(
                "INSERT INTO gacetas (numero, fecha, tipo_edicion, pdf_url, checksum, sumario_text, norma_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    gaceta.numero, gaceta.fecha, gaceta.tipo_edicion,
                    gaceta.pdf_url, gaceta.checksum, gaceta.sumario_text,
                    len(gaceta.normas),
                ),
            )
            gaceta_row_id = cursor.lastrowid
            if gaceta.checksum:
                checksum_to_gaceta_id[gaceta.checksum] = gaceta_row_id

            for norma in gaceta.normas:
                conn.execute(
                    "INSERT INTO normas "
                    "(norma_id, tipo, numero, year, organismo_emisor, goc_code, "
                    "raw_text, page_start, page_end, ordinal_position, match_confidence, "
                    "gaceta_id, raw_metadata_string) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        norma.identity.norma_id,
                        norma.identity.tipo,
                        norma.identity.numero,
                        norma.identity.year,
                        norma.identity.organismo_emisor,
                        norma.goc_code,
                        norma.raw_text,
                        norma.page_range[0],
                        norma.page_range[1],
                        norma.ordinal_position,
                        norma.match_confidence,
                        gaceta_row_id,
                        norma.identity.raw_string,
                    ),
                )

        # Insert duplicate records
        for dup in result.duplicate_normas:
            kept_checksum = dup.get('kept', {}).get('gaceta_checksum', '')
            kept_id = checksum_to_gaceta_id.get(kept_checksum)
            for sup in dup.get('superseded', []):
                sup_checksum = sup.get('gaceta_checksum', '')
                sup_id = checksum_to_gaceta_id.get(sup_checksum)
                conn.execute(
                    "INSERT INTO norma_duplicates (norma_id, kept_gaceta_id, superseded_gaceta_id, notes) "
                    "VALUES (?, ?, ?, ?)",
                    (dup['norma_id'], kept_id, sup_id, 'auto-detected'),
                )

        conn.commit()
        logger.info("SQLite saved: %d gacetas, %d normas", len(result.gacetas), result.total_normas_extracted)
    finally:
        conn.close()


# Output: Processing report

def save_processing_report(result: ProcessingResult, report_path: str) -> None:
    """Save a lightweight diagnostic report."""
    logger.info("Saving processing report to: %s", report_path)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    # Collect gacetas with issues
    gacetas_with_unmatched = []
    confidence_counts = defaultdict(int)
    tipo_counts = defaultdict(int)

    for gaceta in result.gacetas:
        for norma in gaceta.normas:
            confidence_counts[norma.match_confidence] += 1
            tipo_counts[norma.identity.tipo] += 1
        if gaceta.unmatched_segments:
            gacetas_with_unmatched.append({
                'numero': gaceta.numero,
                'fecha': gaceta.fecha,
                'unmatched_count': len(gaceta.unmatched_segments),
                'segments': gaceta.unmatched_segments,
            })

    report = {
        'stats': {
            'total_gacetas': len(result.gacetas),
            'total_normas_extracted': result.total_normas_extracted,
            'total_unmatched_segments': result.total_unmatched_segments,
            'total_chunks_processed': result.total_chunks_processed,
            'total_duplicate_normas': len(result.duplicate_normas),
        },
        'match_confidence_distribution': dict(confidence_counts),
        'norma_tipo_distribution': dict(sorted(tipo_counts.items(), key=lambda x: -x[1])),
        'gacetas_with_unmatched_segments': gacetas_with_unmatched,
        'duplicate_normas': result.duplicate_normas,
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("Report saved with %d issue entries", len(gacetas_with_unmatched))


# CLI

def main() -> None:
    import argparse

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

    default_input = os.path.join(
        project_root, '.data', 'cleaned_content', 'cleaned_elements.json',
    )
    default_json_output = os.path.join(
        project_root, '.data', 'norma_output', 'normas_by_gaceta.json',
    )
    default_db_output = os.path.join(
        project_root, '.data', 'norma_output', 'normas.db',
    )
    default_report_output = os.path.join(
        project_root, '.data', 'norma_output', 'processing_report.json',
    )

    parser = argparse.ArgumentParser(
        description="Segmenta y estructura normas juridicas desde chunks de Gaceta Oficial",
    )
    parser.add_argument('--input-file', default=default_input, help="JSON con elementos limpios")
    parser.add_argument('--json-output', default=default_json_output, help="JSON de salida jerárquico")
    parser.add_argument('--db-output', default=default_db_output, help="SQLite de salida")
    parser.add_argument('--report-output', default=default_report_output, help="Reporte de procesamiento")
    parser.add_argument('--skip-sqlite', action='store_true', help="No generar SQLite")
    parser.add_argument('--skip-json', action='store_true', help="No generar JSON")
    parser.add_argument(
        '--log-level', default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help="Nivel de logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    try:
        elements = _load_elements(args.input_file)
    except FileNotFoundError:
        logger.error("Input file not found: %s", args.input_file)
        sys.exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error("Error reading JSON: %s", e)
        sys.exit(1)

    if not elements:
        logger.error("No elements loaded")
        sys.exit(1)

    result = process_all_normas(elements)

    if not args.skip_json:
        save_json_output(result, args.json_output)

    if not args.skip_sqlite:
        save_sqlite_output(result, args.db_output)

    save_processing_report(result, args.report_output)

    stats = result.to_dict()['stats']
    print(f"\nProcessing complete:")
    print(f"  Gacetas:    {stats['total_gacetas']}")
    print(f"  Normas:     {stats['total_normas_extracted']}")
    print(f"  Unmatched:  {stats['total_unmatched_segments']}")
    print(f"  Duplicates: {stats['total_duplicate_normas']}")


if __name__ == '__main__':
    main()
