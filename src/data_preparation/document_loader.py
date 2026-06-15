import os
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Iterator, Optional, Set, Tuple
import json
import uuid 
import spacy
import time

try:
    import ijson
except ImportError:
    ijson = None

# LangChain imports
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_core.documents import Document


class AsyncDocumentLoader:
    """
    Carga documentos de forma asíncrona desde un directorio especificado y sus subdirectorios,
    procesándolos con UnstructuredFileLoader para extraer elementos estructurados.
    """
    def __init__(self, base_data_dir: str, max_workers: int = 4):
        """
        Inicializa el cargador de documentos asíncrono.
        :param base_data_dir: Directorio base que contiene los documentos.
        :param max_workers: Número máximo de trabajadores concurrentes para procesar documentos.
        """
        try:
            self.nlp = spacy.load("es_core_news_md")  # Cargar modelo de lenguaje en español
        except OSError:
            print("Advertencia: Modelo spaCy 'es_core_news_md' no encontrado. Instalalo con: python -m spacy download es_core_news_md")
            self.nlp = None
            
        self.base_data_dir = base_data_dir
        self.max_workers = max_workers
        self.supported_extensions = ['.pdf']
        self.vocabulary = []
        
    def _save_vocabulary(self, elements: List[Dict[str, Any]], lemmatize: bool):
        """
        Guarda el vocabulario en un archivo JSON.
        :param elements: Lista de elementos con contenido.
        :param lemmatize: Si True, lemmatiza las palabras.
        """
        try:
            if not self.nlp:
                print("Advertencia: spaCy no está disponible. No se puede generar vocabulario.")
                return
                
            all_text = " ".join([item['content'] for item in elements if item.get('content')])
            
            if not all_text.strip():
                print("Advertencia: No hay contenido de texto para generar vocabulario.")
                return
                
            doc = self.nlp(all_text)
            vocabulary = []
            
            if lemmatize:
                vocabulary = list(set(token.lemma_.lower() for token in doc 
                                   if token.is_alpha and not token.is_stop and not token.is_punct and len(token.lemma_) > 2))
            else: 
                vocabulary = list(set(token.text.lower() for token in doc 
                                   if token.is_alpha and not token.is_stop and not token.is_punct and len(token.text) > 2))
            
            self.vocabulary = vocabulary
            output_path = os.path.join(self.base_data_dir, 'vocabulary.json')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(vocabulary, f, ensure_ascii=False, indent=4)
            
            print(f"Vocabulario guardado en: {output_path} ({len(vocabulary)} términos únicos)")
            
        except Exception as e:
            print(f"Error al generar vocabulario: {e}")
                
    def _infer_document_type_from_path(self, filepath: str) -> str:
        """
        Infiere el tipo de documento basado en la ruta del archivo.
        """
        return 'gaceta'

    def _load_document_sync(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Método síncrono para cargar un documento individual.
        Se ejecutará en un hilo separado.
        """
        if not os.path.exists(file_path):
            print(f"Advertencia: Archivo no encontrado en: {file_path}")
            return []

        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_extensions:
            return [] 

        print(f"🔄 Procesando: {os.path.basename(file_path)}")
        try:
            loader = UnstructuredFileLoader(
                file_path,
                mode="elements",
                strategy="fast",
                chunking_strategy="basic",
                languages=["es"]
            )
            elements: List[Document] = loader.load()

            processed_elements: List[Dict[str, Any]] = []
            doc_type = self._infer_document_type_from_path(file_path)
            
            for element in elements:
                metadata = element.metadata
                element_type = metadata.get('category') or metadata.get('type', 'NarrativeText')
                
                processed_metadata = {
                    "source": metadata.get('source', file_path),
                    "page_number": metadata.get('page_number'),
                    "coordinates": metadata.get('coordinates'),
                    "coordinate_system": metadata.get('coordinate_system'),
                    "type": element_type,
                    "filetype": metadata.get('filetype'),
                    "filename": os.path.basename(metadata.get('filename', file_path)),
                    "last_modified": metadata.get('last_modified'),
                    "chunk_id": str(uuid.uuid4()), 
                    "document_type": doc_type 
                }
                
                content = element.page_content
                processed_elements.append({
                    "content": content, 
                    "metadata": processed_metadata
                })
            
            print(f"✅ '{os.path.basename(file_path)}' procesado - {len(processed_elements)} elementos extraídos")
            return processed_elements

        except Exception as e:
            print(f"❌ Error procesando '{os.path.basename(file_path)}': {e}")
            return []

    async def load_document_async(self, file_path: str, executor: concurrent.futures.ThreadPoolExecutor) -> List[Dict[str, Any]]:
        """
        Método asíncrono que delega la carga del documento a un hilo separado.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, self._load_document_sync, file_path)

    async def _process_pdf_files_async(self, pdf_files: List[str]) -> List[Dict[str, Any]]:
        """
        Procesa una lista de PDFs concurrentemente y retorna los elementos extraidos.
        """
        if not pdf_files:
            return []

        all_elements: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            semaphore = asyncio.Semaphore(self.max_workers)

            async def process_with_semaphore(file_path: str):
                async with semaphore:
                    return await self.load_document_async(file_path, executor)

            tasks = [process_with_semaphore(file_path) for file_path in pdf_files]

            print(f"🚀 Iniciando procesamiento concurrente con {self.max_workers} trabajadores...")

            batch_size = self.max_workers * 2
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, Exception):
                        print(f"❌ Error en lote: {result}")
                    elif result:
                        all_elements.extend(result)

                print(f"📊 Progreso: {min(i + batch_size, len(tasks))}/{len(tasks)} archivos procesados")

        return all_elements

    async def load_all_documents_from_base_dir_async(self) -> List[Dict[str, Any]]:
        """
        Carga recursivamente todos los documentos soportados de forma asíncrona
        dentro del directorio base y sus subdirectorios.
        """
        pdf_files = []
        print(f"🔍 Escaneando directorio: {self.base_data_dir}")
        
        for root, dirs, files in os.walk(self.base_data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in self.supported_extensions:
                    pdf_files.append(file_path)
        
        print(f"📁 Se encontraron {len(pdf_files)} archivos PDF para procesar")
        
        if not pdf_files:
            print("⚠️ No se encontraron archivos PDF para procesar")
            return []

        start_time = time.time()
        all_elements = await self._process_pdf_files_async(pdf_files)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✨ Procesamiento completado en {processing_time:.2f} segundos")
        print(f"📈 Estadísticas:")
        print(f"   - Archivos procesados: {len(pdf_files)}")
        print(f"   - Elementos extraídos: {len(all_elements)}")
        print(f"   - Velocidad promedio: {len(pdf_files)/processing_time:.2f} archivos/segundo")
        
        if all_elements:
            self._save_vocabulary(all_elements, lemmatize=False)
        
        return all_elements


class GacetaJsonDocumentLoader(AsyncDocumentLoader):
    """
    Cargador para procesar PDFs ya descargados usando un indice JSON de gacetas.
    No carga el JSON completo en memoria cuando `ijson` esta disponible.
    """

    def __init__(self, base_data_dir: str, gaceta_json_path: str, max_workers: int = 4):
        super().__init__(base_data_dir=base_data_dir, max_workers=max_workers)
        self.gaceta_json_path = gaceta_json_path

    def _iter_gaceta_entries(self) -> Iterator[Dict[str, Any]]:
        """
        Itera entradas del indice JSON con estrategia streaming si ijson esta disponible.
        """
        if not os.path.exists(self.gaceta_json_path):
            raise FileNotFoundError(f"No se encontro el indice JSON: {self.gaceta_json_path}")

        if ijson is None:
            print("⚠️ 'ijson' no instalado; usando json.load (mayor uso de memoria)")
            with open(self.gaceta_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data:
                    if isinstance(entry, dict):
                        yield entry
            return

        with open(self.gaceta_json_path, 'rb') as f:
            for entry in ijson.items(f, 'item'):
                if isinstance(entry, dict):
                    yield entry

    def _resolve_downloaded_pdf(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retorna el descriptor del primer PDF descargado valido en `files`.
        """
        files = entry.get('files') or []
        if not isinstance(files, list):
            return None

        for file_info in files:
            if not isinstance(file_info, dict):
                continue
            if (file_info.get('status') or '').lower() != 'downloaded':
                continue
            rel_path = file_info.get('path', '')
            if not rel_path:
                continue
            return file_info
        return None

    def _enrich_with_gaceta_metadata(
        self,
        elements: List[Dict[str, Any]],
        entry: Dict[str, Any],
        file_info: Dict[str, Any],
        absolute_pdf_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Agrega metadatos de gaceta sin alterar el contrato base de salida.
        """
        for element in elements:
            metadata = element.get('metadata', {})
            metadata.update(
                {
                    'gaceta_tipo_edicion': entry.get('tipo_edicion', ''),
                    'gaceta_fecha': entry.get('fecha', ''),
                    'gaceta_numero': entry.get('numero', ''),
                    'gaceta_normas': entry.get('normas', []) or [],
                    'gaceta_pdf_url': entry.get('pdf_url', ''),
                    'gaceta_checksum': file_info.get('checksum', ''),
                    'gaceta_relative_path': file_info.get('path', ''),
                    'source': absolute_pdf_path,
                }
            )
            element['metadata'] = metadata
        return elements

    async def load_from_gaceta_index_async(
        self,
        dedupe_by_checksum: bool = True,
        max_files: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Procesa PDFs referenciados en `gaceta_oficial.json`.

        :param dedupe_by_checksum: Evita reprocesar checksums repetidos.
        :param max_files: Limita la cantidad de PDFs para pruebas rapidas.
        """
        print(f"🔍 Leyendo indice de gacetas: {self.gaceta_json_path}")

        selected_items: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        seen_checksums: Set[str] = set()
        seen_paths: Set[str] = set()

        total_entries = 0
        skipped_invalid = 0
        skipped_duplicates = 0
        missing_files = 0

        for entry in self._iter_gaceta_entries():
            total_entries += 1
            file_info = self._resolve_downloaded_pdf(entry)
            if file_info is None:
                skipped_invalid += 1
                continue

            rel_path = file_info.get('path', '')
            abs_path = os.path.abspath(os.path.join(self.base_data_dir, rel_path))
            checksum = file_info.get('checksum', '')

            duplicate = False
            if dedupe_by_checksum and checksum:
                duplicate = checksum in seen_checksums
            else:
                duplicate = abs_path in seen_paths

            if duplicate:
                skipped_duplicates += 1
                continue

            if not os.path.exists(abs_path):
                missing_files += 1
                continue

            if checksum:
                seen_checksums.add(checksum)
            seen_paths.add(abs_path)

            selected_items.append((abs_path, entry, file_info))
            if max_files is not None and len(selected_items) >= max_files:
                break

        if not selected_items:
            print("⚠️ No hay PDFs validos para procesar desde el indice")
            print(
                f"Resumen indice: total={total_entries}, invalidas={skipped_invalid}, "
                f"duplicadas={skipped_duplicates}, inexistentes={missing_files}"
            )
            return []

        print(f"📁 Se seleccionaron {len(selected_items)} PDFs del indice para procesar")
        start_time = time.time()

        all_elements: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            semaphore = asyncio.Semaphore(self.max_workers)

            async def process_item(item: Tuple[str, Dict[str, Any], Dict[str, Any]]):
                file_path, entry, file_info = item
                async with semaphore:
                    loaded_elements = await self.load_document_async(file_path, executor)
                    return self._enrich_with_gaceta_metadata(loaded_elements, entry, file_info, file_path)

            tasks = [process_item(item) for item in selected_items]
            batch_size = self.max_workers * 2
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                for result in batch_results:
                    if isinstance(result, Exception):
                        print(f"❌ Error procesando entrada de indice: {result}")
                    elif result:
                        all_elements.extend(result)
                print(f"📊 Progreso indice: {min(i + batch_size, len(tasks))}/{len(tasks)} PDFs")

        elapsed = time.time() - start_time
        print(f"✨ Procesamiento desde indice completado en {elapsed:.2f}s")
        print(
            f"Resumen indice: total={total_entries}, validas={len(selected_items)}, "
            f"invalidas={skipped_invalid}, duplicadas={skipped_duplicates}, inexistentes={missing_files}"
        )
        print(f"📈 Elementos extraidos: {len(all_elements)}")

        if all_elements:
            self._save_vocabulary(all_elements, lemmatize=False)

        return all_elements


if __name__ == "__main__":
    import argparse

    current_script_dir = os.path.dirname(__file__)
    src_root = os.path.abspath(os.path.join(current_script_dir, '..'))
    repo_root = os.path.abspath(os.path.join(src_root, '..'))

    default_raw_dir = os.path.join(src_root, 'hr_scraper', 'downloads', 'full')
    default_json_base = os.path.join(src_root, 'hr_scraper', 'downloads')
    default_json_index = os.path.join(src_root, 'hr_scraper', 'gaceta_oficial.json')
    default_output_file = os.path.join(
        repo_root,
        '.data',
        'processed_loader_output',
        'all_raw_extracted_elements.json',
    )

    parser = argparse.ArgumentParser(description="Carga elementos de PDFs para HybridRank")
    parser.add_argument(
        '--source',
        choices=['json-index', 'filesystem'],
        default='json-index',
        help="Fuente de documentos: indice JSON del scraper o escaneo por carpeta",
    )
    parser.add_argument(
        '--base-data-dir',
        default=default_json_base,
        help="Directorio base para resolver paths de PDFs (modo json-index)",
    )
    parser.add_argument(
        '--index-file',
        default=default_json_index,
        help="Ruta a gaceta_oficial.json (modo json-index)",
    )
    parser.add_argument(
        '--raw-data-dir',
        default=default_raw_dir,
        help="Directorio de PDFs para escaneo recursivo (modo filesystem)",
    )
    parser.add_argument('--max-workers', type=int, default=4, help="Numero de hilos concurrentes")
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help="Limite de PDFs a procesar en modo json-index",
    )
    parser.add_argument(
        '--no-dedupe',
        action='store_true',
        help="Desactiva deduplicacion por checksum en modo json-index",
    )
    parser.add_argument(
        '--output-file',
        default=default_output_file,
        help="Archivo JSON de salida con todos los elementos extraidos",
    )
    args = parser.parse_args()

    if args.source == 'filesystem':
        print(f"🔍 Modo filesystem sobre: {args.raw_data_dir}")
        loader = AsyncDocumentLoader(base_data_dir=args.raw_data_dir, max_workers=args.max_workers)
        all_extracted_elements = asyncio.run(loader.load_all_documents_from_base_dir_async())
    else:
        print(f"🔍 Modo json-index sobre: {args.index_file}")
        loader = GacetaJsonDocumentLoader(
            base_data_dir=args.base_data_dir,
            gaceta_json_path=args.index_file,
            max_workers=args.max_workers,
        )
        all_extracted_elements = asyncio.run(
            loader.load_from_gaceta_index_async(
                dedupe_by_checksum=not args.no_dedupe,
                max_files=args.max_files,
            )
        )

    if not all_extracted_elements:
        print("No se extrajeron elementos. Revisa rutas, indice y archivos PDF.")
        raise SystemExit(1)

    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_extracted_elements, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Se extrajeron {len(all_extracted_elements)} elementos")
    print(f"💾 Elementos guardados en: {args.output_file}")