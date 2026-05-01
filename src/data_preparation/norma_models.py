"""
Data models for structured representation of Cuban Gaceta Oficial normas.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


# --- Regex patterns ---

# GOC codes in PDF text (e.g., GOC-2026-179-EX30, goc-2026-ext.32)
GOC_CODE_PATTERN = re.compile(
    r'(GOC-\d{4}-\d+-(?:EXE|EX|O)\d+)',
    re.IGNORECASE,
)

# Parse norma name strings from scraper metadata
# e.g., "Resolución 41 de 2026 de Ministerio de Finanzas y Precios"
# Longest alternatives first to prevent partial matches.
NORMA_NAME_PATTERN = re.compile(
    r'(?P<tipo>'
    r'Nota\s+de\s+Presentaci[oó]n\s+de\s+Cartas\s+Credenciales'
    r'|Resoluci[oó]n\s+Conjunta'
    r'|Instrucci[oó]n\s+Conjunta'
    r'|Instrucci[oó]n\s+Especial'
    r'|Decreto\s+Presidencial'
    r'|Decreto[\s-]+Ley'
    r'|Constituci[oó]n'
    r'|Convocatoria'
    r'|Instrucci[oó]n'
    r'|Resoluci[oó]n'
    r'|Directiva'
    r'|Dictamen'
    r'|Proclama'
    r'|Acuerdo'
    r'|Decreto'
    r'|Lista'
    r'|Aviso'
    r'|Ley'
    r')\s+'
    r'(?P<numero>[A-Za-z0-9/._-]+(?:\s*-\s*[A-Za-z0-9/._]+)*|[Ss]/[Nn])\s+'
    r'de\s+(?P<year>\d{4})\s+'
    r'de\s+(?P<organismo>.+)$',
    re.IGNORECASE,
)

# Special case: "Constitución de 2019 de Asamblea Nacional..." (no numero field)
NORMA_NAME_NO_NUMERO_PATTERN = re.compile(
    r'(?P<tipo>Constituci[oó]n)\s+'
    r'de\s+(?P<year>\d{4})\s+'
    r'de\s+(?P<organismo>.+)$',
    re.IGNORECASE,
)

# Extract norma type + number from ALL-CAPS headers in PDF body text
# e.g., "RESOLUCION 41/2026", "DECRETO-LEY 114 DE LA ASOCIACION..."
HEADER_TYPE_PATTERN = re.compile(
    r'(?P<tipo>'
    r'NOTA\s+DE\s+PRESENTACI[OÓ]N\s+DE\s+CARTAS\s+CREDENCIALES'
    r'|RESOLUCI[OÓ]N\s+CONJUNTA'
    r'|INSTRUCCI[OÓ]N\s+CONJUNTA'
    r'|DECRETO[\s-]*PRESIDENCIAL'
    r'|DECRETO[\s-]*LEY'
    r'|CONSTITUCI[OÓ]N'
    r'|CONVOCATORIA'
    r'|INSTRUCCI[OÓ]N'
    r'|RESOLUCI[OÓ]N'
    r'|DIRECTIVA'
    r'|DICTAMEN'
    r'|PROCLAMA'
    r'|ACUERDO'
    r'|DECRETO'
    r'|LISTA'
    r'|AVISO'
    r'|LEY'
    r')\s+'
    r'(?:No\.?\s*)?'
    r'(?P<numero>[A-Za-z0-9/._-]+?)'
    r'(?:\s+DE\s+\d{4}|\s+DE\s+[A-Z]|\s*"|/\d{4}|\s)',
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r'\s+', ' ', text).strip()


def _strip_accents(text: str) -> str:
    """Remove accents for comparison purposes only."""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def normalize_tipo(tipo: str) -> str:
    """Normalize a norma type string for comparison."""
    t = _strip_accents(tipo.lower())
    t = t.replace('-', ' ')
    return re.sub(r'\s+', ' ', t).strip()


def normalize_numero(numero: str) -> str:
    """Normalize a norma number for comparison.

    '8/2026' → '8', '08' → '8', 'X-144' → 'x-144', 's/n' → 's/n',
    'X - 6' → 'x-6'
    """
    n = numero.strip().lower()
    # Collapse spaces around hyphens: "X - 6" → "X-6"
    n = re.sub(r'\s*-\s*', '-', n)
    # Strip year suffix: "8/2026" → "8"
    n = re.sub(r'/\d{4}$', '', n)
    # Strip leading zeros for pure numeric: "08" → "8"
    if n.isdigit():
        n = str(int(n))
    return n


def parse_norma_name(norma_string: str) -> Optional['NormaIdentity']:
    """Parse a norma name string from gaceta_normas metadata into a NormaIdentity.

    Examples:
        "Resolución 41 de 2026 de Ministerio de Finanzas y Precios"
        "Decreto Ley 114 de 2025 de Consejo de Estado"
        "Proclama S/N de 2026 de Consejo de Estado"
        "Constitución de 2019 de Asamblea Nacional del Poder Popular"
    """
    if not norma_string or not norma_string.strip():
        return None

    text = _normalize_text(norma_string)

    m = NORMA_NAME_PATTERN.match(text)
    if m:
        tipo_raw = _normalize_text(m.group('tipo'))
        # Normalize "Decreto-Ley" variants to "Decreto Ley"
        tipo_raw = re.sub(r'[\s-]+', ' ', tipo_raw)
        return NormaIdentity(
            tipo=tipo_raw,
            numero=m.group('numero').strip(),
            year=int(m.group('year')),
            organismo_emisor=m.group('organismo').strip(),
            raw_string=norma_string.strip(),
        )

    # Special case: no numero
    m = NORMA_NAME_NO_NUMERO_PATTERN.match(text)
    if m:
        return NormaIdentity(
            tipo=_normalize_text(m.group('tipo')),
            numero='s/n',
            year=int(m.group('year')),
            organismo_emisor=m.group('organismo').strip(),
            raw_string=norma_string.strip(),
        )

    return None


def extract_header_identity(segment_text: str) -> Optional[Tuple[str, str]]:
    """Extract norma type and number from the beginning of a text segment.

    Searches the first 1500 characters for a norma header in ALL-CAPS format.
    Skips past preamble text (signer names, "HAGO SABER:", etc.).
    Returns (normalized_tipo, numero) or None.
    """
    if not segment_text:
        return None

    search_area = segment_text[:1500]
    m = HEADER_TYPE_PATTERN.search(search_area)
    if not m:
        return None

    tipo_raw = _normalize_text(m.group('tipo'))
    # Convert ALL-CAPS to title case, normalize spacing
    tipo_raw = re.sub(r'[\s-]+', ' ', tipo_raw).title()
    numero = m.group('numero').strip()
    return (tipo_raw, numero)


# --- Dataclasses ---

@dataclass
class NormaIdentity:
    """Structured identity of a Cuban legal norm."""
    tipo: str
    numero: str
    year: int
    organismo_emisor: str
    raw_string: str = ''

    @property
    def norma_id(self) -> str:
        """Unique key: e.g., 'decreto_ley_114_2025_consejo_de_estado'."""
        t = normalize_tipo(self.tipo)
        t = t.replace(' ', '_')
        n = self.numero.lower().replace('/', '-')
        org = _strip_accents(self.organismo_emisor.lower())
        org = re.sub(r'[^a-z0-9\s]', '', org)
        org = re.sub(r'\s+', '_', org.strip())
        if len(org) > 50:
            org = org[:50]
        return f"{t}_{n}_{self.year}_{org}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tipo': self.tipo,
            'numero': self.numero,
            'year': self.year,
            'organismo_emisor': self.organismo_emisor,
            'norma_id': self.norma_id,
            'raw_string': self.raw_string,
        }


@dataclass
class Norma:
    """A single legal norm extracted from a Gaceta Oficial."""
    identity: NormaIdentity
    goc_code: str
    raw_text: str
    page_range: Tuple[int, int] = (0, 0)
    ordinal_position: int = 0
    match_confidence: str = 'high'  # high, medium, low, unmatched

    def to_dict(self) -> Dict[str, Any]:
        return {
            'identity': self.identity.to_dict(),
            'goc_code': self.goc_code,
            'raw_text': self.raw_text,
            'page_range': list(self.page_range),
            'ordinal_position': self.ordinal_position,
            'match_confidence': self.match_confidence,
        }


@dataclass
class Gaceta:
    """A Gaceta Oficial issue containing one or more normas."""
    numero: str
    fecha: str
    tipo_edicion: str
    pdf_url: str = ''
    checksum: str = ''
    sumario_text: str = ''
    normas: List[Norma] = field(default_factory=list)
    unmatched_segments: List[Dict[str, Any]] = field(default_factory=list)
    norma_metadata_strings: List[str] = field(default_factory=list)

    @property
    def grouping_key(self) -> str:
        if self.checksum:
            return self.checksum
        return f"{self.numero}_{self.fecha}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'numero': self.numero,
            'fecha': self.fecha,
            'tipo_edicion': self.tipo_edicion,
            'pdf_url': self.pdf_url,
            'checksum': self.checksum,
            'sumario_text': self.sumario_text,
            'norma_count': len(self.normas),
            'normas': [n.to_dict() for n in self.normas],
            'unmatched_segments': self.unmatched_segments,
            'norma_metadata_strings': self.norma_metadata_strings,
        }


@dataclass
class ProcessingResult:
    """Result of processing all gacetas."""
    gacetas: List[Gaceta] = field(default_factory=list)
    total_chunks_processed: int = 0
    total_normas_extracted: int = 0
    total_unmatched_segments: int = 0
    duplicate_normas: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stats': {
                'total_gacetas': len(self.gacetas),
                'total_normas_extracted': self.total_normas_extracted,
                'total_unmatched_segments': self.total_unmatched_segments,
                'total_chunks_processed': self.total_chunks_processed,
                'total_duplicate_normas': len(self.duplicate_normas),
            },
            'gacetas': [g.to_dict() for g in self.gacetas],
            'duplicate_normas': self.duplicate_normas,
        }


# --- Phase 2: Linking dataclasses ---

@dataclass
class NormaReference:
    """A reference to another norma found in text."""
    source_norma_id: str
    ref_tipo: str
    ref_numero: str
    ref_year: Optional[int] = None
    ref_organismo: Optional[str] = None
    relation_type: str = 'menciona'
    raw_text: str = ''
    context_text: str = ''
    resolved_norma_id: Optional[str] = None
    confidence: str = 'unresolved'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_norma_id': self.source_norma_id,
            'ref_tipo': self.ref_tipo,
            'ref_numero': self.ref_numero,
            'ref_year': self.ref_year,
            'ref_organismo': self.ref_organismo,
            'relation_type': self.relation_type,
            'raw_text': self.raw_text,
            'context_text': self.context_text,
            'resolved_norma_id': self.resolved_norma_id,
            'confidence': self.confidence,
        }


@dataclass
class NormaRelationship:
    """A resolved relationship between two normas in the DB."""
    source_norma_id: str
    target_norma_id: str
    relation_type: str
    source_goc_code: str = ''
    target_goc_code: str = ''
    raw_reference_text: str = ''
    resolution_method: str = 'exact'

    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_norma_id': self.source_norma_id,
            'target_norma_id': self.target_norma_id,
            'relation_type': self.relation_type,
            'source_goc_code': self.source_goc_code,
            'target_goc_code': self.target_goc_code,
            'raw_reference_text': self.raw_reference_text,
            'resolution_method': self.resolution_method,
        }


@dataclass
class LinkingResult:
    """Result of the linking pipeline."""
    relationships: List[NormaRelationship] = field(default_factory=list)
    unresolved_references: List[NormaReference] = field(default_factory=list)
    total_references_found: int = 0
    total_resolved: int = 0
    total_ambiguous: int = 0
    total_unresolved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'stats': {
                'total_references_found': self.total_references_found,
                'total_resolved': self.total_resolved,
                'total_ambiguous': self.total_ambiguous,
                'total_unresolved': self.total_unresolved,
                'relationships_created': len(self.relationships),
            },
            'relationships': [r.to_dict() for r in self.relationships],
            'unresolved_references': [r.to_dict() for r in self.unresolved_references],
        }
