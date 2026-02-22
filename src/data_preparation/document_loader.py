import os
import asyncio
import concurrent.futures
from typing import List, Dict, Any, Union
import json
import uuid 
import spacy
from functools import partial
import time

# LangChain imports
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

        # Procesar documentos de forma asíncrona
        start_time = time.time()
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
                    elif result:  # Si hay elementos extraídos
                        all_elements.extend(result)
                
                print(f"📊 Progreso: {min(i + batch_size, len(tasks))}/{len(tasks)} archivos procesados")
        
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


# --- Código de prueba ---
if __name__ == "__main__":
    current_script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_script_dir, '..')) 
    RAW_DATA_ROOT = os.path.join(project_root, 'hr_scraper', 'downloads', 'full')
    
    print(f"DEBUG: RAW_DATA_ROOT is set to: {RAW_DATA_ROOT}")
    
    loader = AsyncDocumentLoader(base_data_dir=RAW_DATA_ROOT, max_workers=4)
    
    print(f"🚀 Cargando todos los documentos de forma ASÍNCRONA desde: {RAW_DATA_ROOT}")
    all_extracted_elements = asyncio.run(loader.load_all_documents_from_base_dir_async())

    if not all_extracted_elements:
        print("No se extrajeron elementos. Revisa las rutas y los archivos en tu directorio RAW.")
    else:
        print(f"\n--- Se extrajeron {len(all_extracted_elements)} elementos de todos los documentos ---")
        output_dir = os.path.join(project_root, '..', '.data', 'processed_loader_output')
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, 'all_raw_extracted_elements.json')
        
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_extracted_elements, f, ensure_ascii=False, indent=4)
        print(f"\n💾 Todos los elementos extraídos guardados en: {output_filepath}")