from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import uuid 

class EmbeddingGenerator:
    """Genera embeddings con paraphrase-multilingual-MiniLM-L12-v2."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        try:
            self.model = SentenceTransformer(model_name)
            print(f"Modelo de embeddings '{model_name}' cargado exitosamente.")
        except Exception as e:
            print(f"Error al cargar el modelo de embeddings '{model_name}': {e}")
            self.model = None

    def generate_embedding(self, text: str) -> np.ndarray:
        if not self.model:
            raise RuntimeError("El modelo de embeddings no se cargó correctamente.")

        embedding = self.model.encode(text, convert_to_tensor=False)
        print(f"Embedding generado para el texto: '{text[:30]}...' (longitud: {len(embedding)})")
        return embedding

    def generate_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.model:
            print("Advertencia: No se generarán embeddings porque el modelo no está cargado.")
            return chunks

        chunks_with_embeddings = []
        total_chunks = len(chunks)
        
        for i, chunk in enumerate(chunks):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Procesando elemento {i + 1}/{total_chunks}...")
                
            text_to_embed = chunk.get('cleaned_content', '')
            chunk_id = chunk.get('metadata', {}).get('chunk_id', None)
            if not chunk_id:
                chunk_id = str(uuid.uuid4())
                if 'metadata' not in chunk:
                    chunk['metadata'] = {}
                chunk['metadata']['chunk_id'] = chunk_id
                print(f"Advertencia: Elemento {i} no tiene 'chunk_id'. Se generó uno: {chunk_id}")

            if not text_to_embed or not text_to_embed.strip():
                print(f"Advertencia: Elemento {chunk_id} tiene contenido vacío. Se asignará embedding vacío.")
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding['embedding'] = []
            else:
                try:
                    embedding = self.generate_embedding(text_to_embed)
                    chunk_with_embedding = chunk.copy()
                    chunk_with_embedding['embedding'] = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                except Exception as e:
                    print(f"Error al generar embedding para elemento {chunk_id}: {e}")
                    chunk_with_embedding = chunk.copy()
                    chunk_with_embedding['embedding'] = []
                    
            chunks_with_embeddings.append(chunk_with_embedding)
            
        print(f"\nProcesamiento completo: {len(chunks_with_embeddings)} elementos con embeddings generados.")
        return chunks_with_embeddings

if __name__ == "__main__":    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    cleaned_elements_path = os.path.join(project_root, '.data', 'cleaned_content', 'cleaned_elements.json')

    cleaned_elements = []
    if os.path.exists(cleaned_elements_path):
        print(f"Cargando elementos desde {cleaned_elements_path}...")
        with open(cleaned_elements_path, 'r', encoding='utf-8') as f:
            cleaned_elements = json.load(f)
        print(f"Cargados {len(cleaned_elements)} elementos limpios desde {cleaned_elements_path}")
    else:
        print(f"Error: No se encontró el archivo de elementos limpios en {cleaned_elements_path}")
        exit(1)

    print("\nIniciando generación de embeddings...")
    generator = EmbeddingGenerator()
    elements_with_embeddings = generator.generate_embeddings_for_chunks(cleaned_elements)

    total_elements = len(elements_with_embeddings)
    valid_embeddings = sum(1 for elem in elements_with_embeddings if elem.get('embedding') and len(elem['embedding']) > 0)
    empty_embeddings = total_elements - valid_embeddings
    
    print(f"\n--- Estadísticas de Embeddings ---")
    print(f"Total de elementos procesados: {total_elements}")
    print(f"Embeddings generados exitosamente: {valid_embeddings}")
    print(f"Embeddings vacíos (por errores o contenido vacío): {empty_embeddings}")
    
    print("\n--- Primeros 3 Elementos con Embeddings ---")
    for i, element in enumerate(elements_with_embeddings[:3]):
        print(f"\nELEMENTO {i+1}:")
        print(f"ID: {element.get('metadata', {}).get('chunk_id', 'N/A')}")
        print(f"Fuente: {element.get('metadata', {}).get('source', 'N/A')}")
        print(f"Contenido original (fragmento): {element.get('content', '')[:100]}...")
        print(f"Contenido limpio (fragmento): {element.get('cleaned_content', '')[:100]}...")
        embedding_length = len(element.get('embedding', []))
        print(f"Tamaño del embedding: {embedding_length}")
        if embedding_length > 0:
            print(f"Embedding (primeros 5 valores): {element['embedding'][:5]}...")
        else:
            print("Embedding: [] (Vacío)")
        print("-" * 50)

    output_dir = os.path.join(project_root, '.data', 'embeddings')
    os.makedirs(output_dir, exist_ok=True)

    output_embeddings_path = os.path.join(output_dir, 'elements_with_embeddings.json')
    print(f"\nGuardando elementos con embeddings en: {output_embeddings_path}")
    
    with open(output_embeddings_path, 'w', encoding='utf-8') as f:
        json.dump(elements_with_embeddings, f, indent=2, ensure_ascii=False)
    
    print(f"\n¡Proceso completado! {valid_embeddings} elementos con embeddings guardados exitosamente.")
    
    stats_path = os.path.join(output_dir, 'embedding_stats.json')
    stats = {
        'total_elements': total_elements,
        'valid_embeddings': valid_embeddings,
        'empty_embeddings': empty_embeddings,
        'embedding_model': generator.model.get_sentence_embedding_dimension() if generator.model else None,
        'embedding_dimension': len(elements_with_embeddings[0]['embedding']) if valid_embeddings > 0 else 0,
        'processed_at': str(uuid.uuid4())
    }
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Estadísticas guardadas en: {stats_path}")