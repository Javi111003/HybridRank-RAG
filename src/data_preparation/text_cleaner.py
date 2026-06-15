import re
import unicodedata
import uuid
import os
import sys
import time
import json
import math
from typing import List, Dict, Any, Iterator, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import spacy

try:
    import ijson
except ImportError:
    ijson = None

try:
    from src.shared.legal_terms import CUSTOM_STOPWORDS, LEGAL_TERMS_TO_KEEP
except ModuleNotFoundError:
    # Permite ejecutar este archivo directamente como script.
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from src.shared.legal_terms import CUSTOM_STOPWORDS, LEGAL_TERMS_TO_KEEP

try:
    nlp_model = spacy.load("es_core_news_md")
    nlp_model.Defaults.stop_words.update(CUSTOM_STOPWORDS)
except Exception as e:
    nlp_model = None
    print(f"No se pudo cargar el modelo de SpaCy 'es_core_news_md': {e}")
    print("La lematización y el filtrado de stopwords no se aplicarán en TextCleaner.")


class TextCleaner:
    """
    Clase para limpiar y normalizar texto extraído de documentos legales cubanos (Gaceta Oficial).
    Aplica una serie de operaciones de pre-procesamiento adaptadas para textos jurídicos
    para su posterior segmentación y generación de embeddings.
    Trabaja con la salida estructurada del DocumentLoader (lista de diccionarios).
    """
    def __init__(self, min_cleaned_length: int = 10, batch_size: int = 500, max_workers: Optional[int] = None):
        """
        Inicializa el limpiador de texto.

        :param min_cleaned_length: Longitud mínima de cleaned_content para conservar un elemento.
        :param batch_size: Tamaño de lote para procesamiento paralelo.
        :param max_workers: Número de workers para ProcessPoolExecutor (None = cpu_count).
        """
        self.nlp = nlp_model
        self.min_cleaned_length = min_cleaned_length
        self.batch_size = batch_size
        self.max_workers = max_workers

    def extract_legal_references(self, text: str):
        """
        Extrae referencias legales (artículos, incisos, etc.) para preservarlas durante la limpieza.
        """
        pattern = r"""
            (                                           # Grupo principal
            (?:Art[íi]culo|Art\.|ARTÍCULO)\s*\d+[a-z]?  # Artículo 123, Art. 45a
            |(?:Art[íi]culo|Art\.|ARTÍCULO)\s*\d+\s*(?:bis|ter|quater|quinquies)\b  # Artículo 5 bis
            |(?:Art[íi]culo|Art\.|ARTÍCULO)\s*\d+[º°]   # Artículo 5º
            |(?:Inciso|INCISO)\s*[a-z]\)               # Inciso a), INCISO b)
            |(?:Apartado|APARTADO)\s*\d+               # Apartado 1, APARTADO 2
            |(?:Ley|LEY)\s*(?:No\.|Número|N[úu]mero)?\s*\d+(?:/\d+)?  # Ley No. 123/2019
            |(?:Decreto|DECRETO)(?:-Ley|-LEY)?\s*(?:No\.|Número)?\s*\d+(?:/\d+)?  # Decreto-Ley 15/2020
            |(?:Resolución|RESOLUCIÓN)\s*(?:No\.|Número)?\s*\d+(?:/\d+)?  # Resolución No. 45/2021
            |GOC-\d+-\w+-EX\d+                        # Código de Gaceta: GOC-2020-48-EX5
            )
        """
        references = re.findall(pattern, text, re.VERBOSE | re.IGNORECASE)

        text_sin_referencias = text
        placeholders = []
        for i, ref in enumerate(references):
            placeholder = f"__legal_ref_{uuid.uuid4().hex[:10]}_{i}__"
            placeholders.append(placeholder)
            text_sin_referencias = text_sin_referencias.replace(ref, placeholder, 1)
        return text_sin_referencias, list(zip(placeholders, references))
    
    def reincorporate_legal_references(self, text: str, references: List[tuple]) -> str:
        """Reincorpora las referencias legales al texto limpio."""
        for placeholder, ref in references:
            text = text.replace(placeholder, ref)
        return text

    def remove_extra_whitespace(self, text: str) -> str:
        """Elimina espacios en blanco adicionales."""
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def normalize_unicode(self, text: str) -> str:
        """Normaliza caracteres Unicode manteniendo acentos españoles."""
        return unicodedata.normalize('NFKC', text)

    def remove_gaceta_headers_footers(self, text: str) -> str:
        """
        Elimina patrones comunes de encabezados y pies de página de la Gaceta Oficial.
        Soporta ediciones Ordinaria, Extraordinaria y Extraordinaria Especial.
        """
        patterns = [
            # Información de Gaceta (ordinaria, extraordinaria, extraordinaria especial)
            r'Gaceta Oficial No\.\s*\d+.*?de.*?\d{4}',
            r'EXTRAORDINARIA(?:\s+ESPECIAL)?\s+AÑO\s+C\w+',
            r'LA HABANA,.*?\d{4}',
            r'Sitio Web:\s*https?://www\.gacetaoficial\.gob\.cu/?',
            r'Calle Zanja No\.\s*\d+.*?Centro Habana',
            r'Teléfonos:\s*[\d\-\s]+',
            # Números de página
            r'^\s*Página\s+\d+\s*$',
            r'^\s*\d+\s*$',  # Solo números
            r'^\s*\[\s*\d+\s*\]\s*$',  # [123]
            # Códigos de referencia – cubre GOC-2020-48-EX5, goc-2026-exe5, goc-2026-ext.32, goc-2026-o24
            r'^\s*(?:GOC|goc)-\d{4}-(?:o|O|ext?\.?|EX|exe|EXE)\d+[_\d]*\s*$',
            # Formato legacy
            r'^\s*GOC-\d+-\w+-EX\d+\s*$',
            # Separadores y líneas
            r'^_{3,}$',
            r'^-+$'
        ]

        cleaned_text = text
        for pattern in patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.MULTILINE | re.IGNORECASE)
        return cleaned_text

    def remove_urls(self, text: str) -> str:
        """Elimina URLs del texto."""
        return re.sub(r'(https?://[^\s]+)|(www\.[^\s]+)', '', text)

    def remove_emails(self, text: str) -> str:
        """Elimina direcciones de email del texto."""
        return re.sub(r'\S*@\S*\s?', '', text)

    def normalize_legal_formatting(self, text: str) -> str:
        """
        Normaliza el formato de textos legales cubanos.
        """
        # Normalizar "POR CUANTO:" y similares
        text = re.sub(r'POR\s+CUANTO\s*:', 'POR CUANTO:', text, flags=re.IGNORECASE)
        text = re.sub(r'CONSIDERANDO\s*:', 'CONSIDERANDO:', text, flags=re.IGNORECASE)
        text = re.sub(r'RESUELVO\s*:', 'RESUELVO:', text, flags=re.IGNORECASE)
        text = re.sub(r'DISPONGO\s*:', 'DISPONGO:', text, flags=re.IGNORECASE)
        
        # Normalizar referencias a artículos
        text = re.sub(r'Art[íi]culo\s+(\d+)', r'Artículo \1', text, flags=re.IGNORECASE)
        text = re.sub(r'Art\.\s*(\d+)', r'Artículo \1', text, flags=re.IGNORECASE)
        
        return text

    def remove_non_alphanumeric(self, text: str, keep_punctuation: str = r'.,!?;:()[]{}%-/°ª_') -> str:
        """
        Elimina caracteres no alfanuméricos excepto puntuación específica para textos legales.
        """
        pattern = r'[^\w\s' + re.escape(keep_punctuation) + r']'
        text = re.sub(pattern, '', text)
        return text

    def convert_to_lowercase(self, text: str) -> str:
        """Convierte texto a minúsculas."""
        return text.lower()

    def _extract_lemmas_from_doc(self, doc, include_stopwords: bool = False) -> str:
        """Extrae lemas filtrados de un Doc de spaCy ya procesado."""
        tokens = []
        for token in doc:
            if not token.is_space and not token.is_punct:
                token_lower = token.text.lower()
                if (include_stopwords
                        or not token.is_stop
                        or token_lower in LEGAL_TERMS_TO_KEEP
                        or token.pos_ in ['PROPN', 'NUM']):
                    tokens.append(token.lemma_.lower())
        return " ".join(tokens)

    def remove_stopwords_and_lemmatize(self, text: str, include_stopwords: bool = False) -> str:
        """
        Elimina stopwords y aplica lematización usando spaCy.
        Para textos legales, conserva términos técnicos importantes.
        """
        if not self.nlp:
            return text

        doc = self.nlp(text)
        return self._extract_lemmas_from_doc(doc, include_stopwords)

    def clean_element_content(self, content: str, element_type: str, apply_lemmatization: bool = False) -> str:
        """
        Aplica limpieza específica según el tipo de elemento de documentos legales.

        :param content: El texto en bruto del elemento a limpiar.
        :param element_type: El tipo de elemento (ej., 'CompositeElement', 'Table', 'Title').
        :param apply_lemmatization: Si es True, aplica lematización y elimina stopwords.
        """
        if not content or not content.strip():
            return ""
            
        legal_references = []
        cleaned_content = content
        
        cleaned_content = self.remove_urls(cleaned_content)
        cleaned_content = self.remove_emails(cleaned_content)
        
        # Extraer referencias legales antes de limpiar
        cleaned_content, legal_references = self.extract_legal_references(cleaned_content)
        if element_type == 'Table':
            cleaned_content = self.remove_extra_whitespace(cleaned_content)
        else:
            cleaned_content = self.remove_gaceta_headers_footers(cleaned_content)
            cleaned_content = self.normalize_legal_formatting(cleaned_content)
            cleaned_content = self.normalize_unicode(cleaned_content)
            cleaned_content = self.remove_extra_whitespace(cleaned_content)
            
            if not self._is_important_header(cleaned_content):
                cleaned_content = self.remove_non_alphanumeric(
                    cleaned_content, 
                    keep_punctuation=r'.,!?;:()[]%-/°ªº_'
                )
            
            if legal_references:
                cleaned_content = self.reincorporate_legal_references(cleaned_content, legal_references)
                
            if apply_lemmatization:
                cleaned_content = self.remove_stopwords_and_lemmatize(cleaned_content)
            else:
                cleaned_content = self.convert_to_lowercase(cleaned_content)
        
        cleaned_content = self.remove_extra_whitespace(cleaned_content)
        
        return cleaned_content

    def _is_important_header(self, text: str) -> bool:
        """
        Determina si el texto es un encabezado importante que debe preservarse.
        """
        important_patterns = [
            r'^(?:LEY|DECRETO|RESOLUCIÓN|ORDENANZA)',
            r'^(?:ARTÍCULO|CAPÍTULO|SECCIÓN|TÍTULO)',
            r'^(?:POR CUANTO|CONSIDERANDO|RESUELVO|DISPONGO)',
            r'^\w+(?:\s+\w+){0,3}:$',  # Patrones como "HAGO SABER:"
        ]
        
        for pattern in important_patterns:
            if re.match(pattern, text.strip(), re.IGNORECASE):
                return True
        return False

    def _clean_batch(self, batch: List[Dict[str, Any]], apply_lemmatization: bool) -> List[Dict[str, Any]]:
        """
        Limpia un lote de elementos. Usa nlp.pipe() para lematización en batch.
        Método diseñado para ser ejecutado en workers paralelos.
        """
        # Fase 1: limpieza sin lematización
        pre_cleaned: List[Dict[str, Any]] = []
        for element in batch:
            try:
                original_content = element.get("content", "")
                element_type = element.get("metadata", {}).get("type", "CompositeElement")
                cleaned_content = self.clean_element_content(
                    original_content, element_type, apply_lemmatization=False
                )
                if cleaned_content and len(cleaned_content.strip()) >= self.min_cleaned_length:
                    cleaned_element = element.copy()
                    cleaned_element["content"] = original_content
                    cleaned_element["cleaned_content"] = cleaned_content
                    pre_cleaned.append(cleaned_element)
            except Exception:
                continue

        # Fase 2: lematización batch con nlp.pipe si se solicitó
        if apply_lemmatization and self.nlp and pre_cleaned:
            texts = [el["cleaned_content"] for el in pre_cleaned]
            docs = self.nlp.pipe(texts, batch_size=128)
            for el, doc in zip(pre_cleaned, docs):
                el["cleaned_content"] = self._extract_lemmas_from_doc(doc)

        return pre_cleaned

    def clean_documents(self, elements: List[Dict[str, Any]], apply_lemmatization: bool = False) -> List[Dict[str, Any]]:
        """
        Limpia una lista de elementos de documento estructurados (salida de DocumentLoader).
        Procesa en batches paralelos para mejorar rendimiento con datasets grandes.

        :param elements: Lista de diccionarios con elementos extraídos por Unstructured.io
        :param apply_lemmatization: Si es True, aplica lematización y elimina stopwords
        :return: Lista de elementos con contenido limpio
        """
        total = len(elements)
        print(f"🧹 Iniciando limpieza de {total} elementos "
              f"(batch_size={self.batch_size}, workers={self.max_workers or 'auto'})...")
        start_time = time.time()

        # Dividir en batches
        batches = [
            elements[i:i + self.batch_size]
            for i in range(0, total, self.batch_size)
        ]
        num_batches = len(batches)

        cleaned_elements: List[Dict[str, Any]] = []
        processed = 0

        # Para datasets pequeños o un solo batch, procesar directamente sin overhead de procesos
        if num_batches <= 1 or total < self.batch_size * 2:
            for idx, batch in enumerate(batches):
                result = self._clean_batch(batch, apply_lemmatization)
                cleaned_elements.extend(result)
                processed += len(batch)
                self._print_progress(processed, total, start_time)
        else:
            # Procesamiento paralelo con ThreadPool (spaCy objects no son pickle-safe)
            from concurrent.futures import ThreadPoolExecutor
            effective_workers = self.max_workers or min(os.cpu_count() or 1, num_batches)
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {
                    executor.submit(self._clean_batch, batch, apply_lemmatization): idx
                    for idx, batch in enumerate(batches)
                }
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        cleaned_elements.extend(result)
                    except Exception as e:
                        print(f"⚠️ Error en batch: {e}")
                    processed += len(batches[futures[future]])
                    self._print_progress(processed, total, start_time)

        elapsed = time.time() - start_time
        rate = total / elapsed if elapsed > 0 else 0
        print(f"\n✅ Limpieza completada: {len(cleaned_elements)} elementos válidos "
              f"de {total} originales en {elapsed:.2f}s ({rate:.0f} elem/s)")
        return cleaned_elements

    @staticmethod
    def _print_progress(processed: int, total: int, start_time: float) -> None:
        """Imprime progreso de procesamiento."""
        pct = processed / total * 100 if total else 100
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"\r📊 Progreso: {processed}/{total} ({pct:.0f}%) - {rate:.0f} elem/s", end="", flush=True)


def _load_elements(path: str) -> List[Dict[str, Any]]:
    """Carga elementos desde JSON con streaming (ijson) si está disponible."""
    if ijson is not None:
        print(f"📂 Cargando elementos en streaming desde: {path}")
        elements = []
        with open(path, 'rb') as f:
            for item in ijson.items(f, 'item'):
                if isinstance(item, dict):
                    elements.append(item)
        print(f"✅ Cargados {len(elements)} elementos (streaming)")
        return elements

    print(f"📂 Cargando elementos desde: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        elements = json.load(f)
    print(f"✅ Cargados {len(elements)} elementos")
    return elements


def _save_elements(elements: List[Dict[str, Any]], path: str) -> None:
    """Guarda elementos escribiendo de forma incremental para reducir pico de RAM."""
    print(f"💾 Guardando {len(elements)} elementos en: {path}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('[\n')
        for idx, item in enumerate(elements):
            json.dump(item, f, ensure_ascii=False)
            if idx < len(elements) - 1:
                f.write(',\n')
            else:
                f.write('\n')
        f.write(']\n')
    print(f"✅ Guardados {len(elements)} elementos limpios")


if __name__ == "__main__":
    import argparse

    current_script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))

    default_input = os.path.join(
        project_root,
        '.data',
        'processed_loader_output',
        'all_raw_extracted_elements.json',
    )
    legacy_input = os.path.join(
        project_root,
        '.data',
        'loader_output',
        'all_raw_extracted_elements.json',
    )
    default_output = os.path.join(project_root, '.data', 'cleaned_content', 'cleaned_elements.json')

    parser = argparse.ArgumentParser(description="Limpieza de elementos legales para HybridRank")
    parser.add_argument('--input-file', default=default_input, help="JSON con elementos extraidos")
    parser.add_argument('--output-file', default=default_output, help="JSON de salida con cleaned_content")
    parser.add_argument('--lemmatize', action='store_true', help="Aplica lematizacion y filtro de stopwords")
    parser.add_argument(
        '--min-cleaned-length',
        type=int,
        default=10,
        help="Longitud minima de cleaned_content para conservar un elemento",
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=500,
        help="Tamaño de lote para procesamiento paralelo",
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help="Numero de workers paralelos (default: cpu_count)",
    )
    args = parser.parse_args()

    input_file = args.input_file
    if not os.path.exists(input_file) and input_file == default_input and os.path.exists(legacy_input):
        input_file = legacy_input

    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)

    try:
        elements = _load_elements(input_file)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {input_file}")
        exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"❌ Error al leer JSON: {e}")
        exit(1)

    cleaner = TextCleaner(
        min_cleaned_length=args.min_cleaned_length,
        batch_size=args.batch_size,
        max_workers=args.workers,
    )

    print("\n🧹 Procesando elementos...")
    total_start = time.time()
    cleaned_elements = cleaner.clean_documents(elements, apply_lemmatization=args.lemmatize)
    total_elapsed = time.time() - total_start

    try:
        _save_elements(cleaned_elements, args.output_file)
    except Exception as e:
        print(f"❌ Error al guardar archivo: {e}")
        exit(1)

    rate = len(elements) / total_elapsed if total_elapsed > 0 else 0
    print(f"\n📊 Resumen del procesamiento:")
    print(f"   • Elementos originales: {len(elements)}")
    print(f"   • Elementos limpios:    {len(cleaned_elements)}")
    print(f"   • Tiempo total:         {total_elapsed:.2f}s")
    print(f"   • Velocidad:            {rate:.0f} elem/s")
    print(f"   • Archivo de salida:    {args.output_file}")