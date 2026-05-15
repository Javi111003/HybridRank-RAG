"""
Build indexable retrieval fragments from structured normas.

Reads the SQLite output produced by norma_processor.py and exports a JSON array
compatible with embedding_generator_e5.py and index_builder.py.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    from src.data_preparation.norma_models import NormaIndexFragment
except ModuleNotFoundError:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from src.data_preparation.norma_models import NormaIndexFragment


logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, '.data', 'norma_output', 'normas.db')
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, '.data', 'norma_output', 'norma_fragments.json')

WHOLE_NORMA_MAX_WORDS = 450
WINDOW_WORDS = 380
WINDOW_OVERLAP_WORDS = 60
SMALL_BLOCK_MIN_WORDS = 80

_WHITESPACE_PATTERN = re.compile(r'\s+')
_WORD_PATTERN = re.compile(r'\S+')

_STRUCTURAL_HEADING_PATTERN = re.compile(
    r'(?is)(?:^|(?<=\n)|(?<=\.\s)|(?<=:\s))\s*'
    r'(?P<label>'
    r'CAP[IÍ]TULO\s+[IVXLCDM0-9]+'
    r'|SECCI[OÓ]N\s+(?:PRIMERA|SEGUNDA|TERCERA|CUARTA|QUINTA|SEXTA|'
    r'S[EÉ]PTIMA|OCTAVA|NOVENA|D[EÉ]CIMA|[IVXLCDM0-9]+)'
    r'|T[IÍ]TULO\s+[IVXLCDM0-9]+'
    r'|DISPOSICIONES\s+(?:GENERALES|ESPECIALES|FINALES|TRANSITORIAS)'
    r'|ANEXO(?:\s+[A-Z0-9IVXLCDM]+)?'
    r'|ART[ÍI]CULO\s+\d+[A-Z]?(?:\.\d+)*\.?'
    r'|Art[íi]culo\s+\d+[a-z]?(?:\.\d+)*\.?'
    r'|PRIMERO\s*:|SEGUNDO\s*:|TERCERO\s*:|CUARTO\s*:|QUINTO\s*:'
    r'|SEXTO\s*:|S[EÉ]PTIMO\s*:|OCTAVO\s*:|NOVENO\s*:|D[EÉ]CIMO\s*:'
    r'|UND[EÉ]CIMO\s*:|DUOD[EÉ]CIMO\s*:'
    r')'
)


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(' ', text or '').strip()


def _word_count(text: str) -> int:
    return len(_WORD_PATTERN.findall(text or ''))


def _clean_for_index(text: str) -> str:
    return _normalize_whitespace(text).lower()


def _safe_id_part(value: Any, fallback: str) -> str:
    text = str(value or '').strip()
    if not text:
        text = fallback
    text = re.sub(r'[^A-Za-z0-9._-]+', '_', text)
    text = text.strip('_.-')
    return text or fallback


def _gaceta_checksum_12(row: Dict[str, Any]) -> str:
    checksum = row.get('gaceta_checksum') or ''
    if checksum:
        return _safe_id_part(checksum[:12], 'nochecksum')

    stable_source = '|'.join([
        str(row.get('gaceta_numero') or ''),
        str(row.get('gaceta_fecha') or ''),
        str(row.get('gaceta_tipo_edicion') or ''),
    ])
    return hashlib.sha1(stable_source.encode('utf-8')).hexdigest()[:12]


def build_fragment_id(row: Dict[str, Any], fragment_index: int) -> str:
    """Build the deterministic ID used by both BM25 and Chroma."""
    norma_id = _safe_id_part(row.get('norma_id'), 'norma')
    checksum = _gaceta_checksum_12(row)
    goc_or_ordinal = row.get('goc_code') or f"ord{int(row.get('ordinal_position') or 0):03d}"
    goc_or_ordinal = _safe_id_part(goc_or_ordinal, 'ord000')
    return f"{norma_id}__{checksum}__{goc_or_ordinal}__f{fragment_index:03d}"


def split_normative_blocks(text: str) -> List[Tuple[str, str]]:
    """Split a norma into legal structure blocks when headings are present."""
    normalized = _normalize_whitespace(text)
    matches = list(_STRUCTURAL_HEADING_PATTERN.finditer(normalized))
    if not matches:
        return [('Norma completa', normalized)] if normalized else []

    blocks: List[Tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = normalized[:matches[0].start()].strip()
        if preamble:
            blocks.append(('Preambulo', preamble))

    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        block_text = normalized[start:end].strip()
        if not block_text:
            continue
        label = _normalize_whitespace(match.group('label')).rstrip('.:')
        blocks.append((label, block_text))

    return _merge_small_blocks(blocks)


def _merge_small_blocks(blocks: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Attach tiny structural headings to neighboring text before windowing."""
    merged: List[Tuple[str, str]] = []
    pending_label = ''
    pending_text = ''

    for label, text in blocks:
        if not text:
            continue
        if pending_text:
            text = f"{pending_text} {text}"
            label = pending_label if pending_label != 'Preambulo' else label
            pending_label = ''
            pending_text = ''

        if _word_count(text) < SMALL_BLOCK_MIN_WORDS:
            pending_label = label
            pending_text = text
            continue

        merged.append((label, text))

    if pending_text:
        if merged and _word_count(merged[-1][1]) + _word_count(pending_text) <= WHOLE_NORMA_MAX_WORDS:
            last_label, last_text = merged[-1]
            merged[-1] = (last_label, f"{last_text} {pending_text}")
        else:
            merged.append((pending_label or 'Fragmento', pending_text))

    return merged


def _window_block(label: str, text: str, window_words: int, overlap_words: int) -> List[Tuple[str, str]]:
    words = _WORD_PATTERN.findall(text)
    if len(words) <= WHOLE_NORMA_MAX_WORDS:
        return [(label, text)]

    step = max(1, window_words - overlap_words)
    windows: List[Tuple[str, str]] = []
    start = 0
    part = 1
    while start < len(words):
        end = min(start + window_words, len(words))
        chunk_words = words[start:end]
        windows.append((f"{label} parte {part}", ' '.join(chunk_words)))
        if end >= len(words):
            break
        start += step
        part += 1
    return windows


def split_norma_text(
    text: str,
    whole_norma_max_words: int = WHOLE_NORMA_MAX_WORDS,
    window_words: int = WINDOW_WORDS,
    overlap_words: int = WINDOW_OVERLAP_WORDS,
) -> List[Tuple[str, str]]:
    """Return labeled fragments for one norma raw_text."""
    text = _normalize_whitespace(text)
    if not text:
        return []
    if _word_count(text) <= whole_norma_max_words:
        return [('Norma completa', text)]

    fragments: List[Tuple[str, str]] = []
    for label, block_text in split_normative_blocks(text):
        if _word_count(block_text) <= whole_norma_max_words:
            fragments.append((label, block_text))
        else:
            fragments.extend(_window_block(label, block_text, window_words, overlap_words))
    return fragments


def _get_superseded_keys(conn: sqlite3.Connection) -> set:
    cursor = conn.execute(
        "SELECT norma_id, superseded_gaceta_id "
        "FROM norma_duplicates "
        "WHERE superseded_gaceta_id IS NOT NULL"
    )
    return {
        (row['norma_id'], row['superseded_gaceta_id'])
        for row in cursor.fetchall()
    }


def _iter_norma_rows(conn: sqlite3.Connection, include_superseded: bool = False) -> Iterable[Dict[str, Any]]:
    superseded_keys = set() if include_superseded else _get_superseded_keys(conn)
    cursor = conn.execute(
        """
        SELECT
            n.id AS norma_row_id,
            n.norma_id,
            n.tipo,
            n.numero,
            n.year,
            n.organismo_emisor,
            n.goc_code,
            n.raw_text,
            n.page_start,
            n.page_end,
            n.ordinal_position,
            n.match_confidence,
            n.raw_metadata_string,
            g.id AS gaceta_id,
            g.numero AS gaceta_numero,
            g.fecha AS gaceta_fecha,
            g.tipo_edicion AS gaceta_tipo_edicion,
            g.pdf_url AS gaceta_pdf_url,
            g.checksum AS gaceta_checksum
        FROM normas n
        JOIN gacetas g ON g.id = n.gaceta_id
        ORDER BY g.id, n.ordinal_position, n.id
        """
    )

    for row in cursor.fetchall():
        data = dict(row)
        if (data['norma_id'], data['gaceta_id']) in superseded_keys:
            continue
        yield data


def _build_metadata(row: Dict[str, Any], fragment_id: str, fragment_index: int,
                    fragment_total: int, fragment_label: str) -> Dict[str, Any]:
    page_start = row.get('page_start')
    checksum_12 = _gaceta_checksum_12(row)
    return {
        'chunk_id': fragment_id,
        'source': 'normas.db',
        'type': 'NormaIndexFragment',
        'filename': 'normas.db',
        'document_type': 'norma',
        'filetype': 'sqlite',
        'page_number': page_start if page_start is not None else -1,
        'corpus_type': 'normas',
        'norma_id': row.get('norma_id') or '',
        'fragment_id': fragment_id,
        'fragment_index': fragment_index,
        'fragment_total': fragment_total,
        'fragment_label': fragment_label,
        'tipo': row.get('tipo') or '',
        'numero': row.get('numero') or '',
        'year': int(row.get('year') or 0),
        'organismo_emisor': row.get('organismo_emisor') or '',
        'goc_code': row.get('goc_code') or '',
        'gaceta_id': int(row.get('gaceta_id') or 0),
        'gaceta_numero': row.get('gaceta_numero') or '',
        'gaceta_fecha': row.get('gaceta_fecha') or '',
        'gaceta_tipo_edicion': row.get('gaceta_tipo_edicion') or '',
        'gaceta_pdf_url': row.get('gaceta_pdf_url') or '',
        'gaceta_checksum': row.get('gaceta_checksum') or '',
        'gaceta_checksum_12': checksum_12,
        'page_start': page_start if page_start is not None else -1,
        'page_end': row.get('page_end') if row.get('page_end') is not None else -1,
        'match_confidence': row.get('match_confidence') or '',
        'ordinal_position': int(row.get('ordinal_position') or 0),
        'raw_metadata_string': row.get('raw_metadata_string') or '',
    }


def build_fragments_for_norma(row: Dict[str, Any]) -> List[NormaIndexFragment]:
    labeled_texts = split_norma_text(row.get('raw_text') or '')
    total = len(labeled_texts)
    fragments: List[NormaIndexFragment] = []

    title = f"{row.get('tipo') or ''} {row.get('numero') or ''} de {row.get('year') or ''}"
    issuer = row.get('organismo_emisor') or ''
    context_prefix = _normalize_whitespace(f"{title} - {issuer}")

    for index, (label, fragment_text) in enumerate(labeled_texts):
        fragment_id = build_fragment_id(row, index)
        content = _normalize_whitespace(f"{context_prefix}. {label}. {fragment_text}")
        metadata = _build_metadata(row, fragment_id, index, total, label)
        fragments.append(NormaIndexFragment(
            fragment_id=fragment_id,
            content=content,
            cleaned_content=_clean_for_index(content),
            metadata=metadata,
        ))

    return fragments


def build_norma_fragments(db_path: str, include_superseded: bool = False) -> List[NormaIndexFragment]:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"No se encontro la base de normas: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        fragments: List[NormaIndexFragment] = []
        for row in _iter_norma_rows(conn, include_superseded=include_superseded):
            fragments.extend(build_fragments_for_norma(row))
        return fragments
    finally:
        conn.close()


def save_fragments(fragments: Sequence[NormaIndexFragment], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump([fragment.to_dict() for fragment in fragments], f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exporta fragmentos normativos indexables desde normas.db",
    )
    parser.add_argument('--db-path', default=DEFAULT_DB_PATH, help="Ruta a .data/norma_output/normas.db")
    parser.add_argument('--output-file', default=DEFAULT_OUTPUT_PATH, help="JSON de fragmentos de salida")
    parser.add_argument(
        '--include-superseded',
        action='store_true',
        help="Incluye ocurrencias de normas marcadas como duplicadas/superseded",
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help="Nivel de logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    fragments = build_norma_fragments(args.db_path, include_superseded=args.include_superseded)
    save_fragments(fragments, args.output_file)

    norma_ids = {fragment.metadata['norma_id'] for fragment in fragments}
    logger.info("Exported %d fragments from %d normas to %s", len(fragments), len(norma_ids), args.output_file)


if __name__ == '__main__':
    main()
