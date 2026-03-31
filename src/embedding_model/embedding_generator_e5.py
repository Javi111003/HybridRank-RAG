from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import json
import os
import time
import uuid


class EmbeddingGeneratorE5:
    """
    Genera embeddings vectoriales para chunks de texto utilizando el modelo E5 multilingual.
    Usa el parametro prompt='passage: ' nativo de sentence-transformers para el prefijo E5.
    Soporta multi-process encoding, FP16, y normalize_embeddings para busqueda coseno.
    """

    PASSAGE_PROMPT = "passage: "

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        num_workers: Optional[int] = None,
        use_fp16: bool = False,
    ):
        """
        Inicializa el generador de embeddings con el modelo E5.
        :param model_name: Nombre del modelo a cargar desde Hugging Face.
        :param batch_size: Tamano de lote para encode.
        :param normalize_embeddings: Si True, normaliza embeddings a norma unitaria (optimo para coseno).
        :param num_workers: Numero de procesos paralelos para encode (None=1 proceso, >1 multi-process).
                            En multi-process cada worker carga el modelo independientemente.
        :param use_fp16: Si True, carga el modelo en float16 (acelera GPU; en CPU puede ser mas lento).
        """
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.normalize_embeddings = normalize_embeddings
        self.num_workers = max(1, int(num_workers)) if num_workers is not None else 1
        self.use_fp16 = use_fp16
        self.model: Optional[SentenceTransformer] = None

        model_kwargs = {}
        if use_fp16:
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16
                print("FP16 habilitado para GPU.")
            else:
                print("Advertencia: FP16 solicitado pero no hay GPU disponible. Usando FP32.")

        try:
            self.model = SentenceTransformer(model_name, model_kwargs=model_kwargs)
            dim = self.model.get_sentence_embedding_dimension()
            device = self.model.device
            print(f"Modelo E5 '{model_name}' cargado (dim={dim}, device={device}).")
            print(
                f"Config: batch_size={self.batch_size}, normalize={self.normalize_embeddings}, "
                f"workers={self.num_workers}, fp16={self.use_fp16}"
            )
        except Exception as e:
            print(f"Error al cargar el modelo '{model_name}': {e}")

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Genera el embedding para un solo texto, aplicando el prefijo 'passage: ' via prompt nativo.
        :param text: El texto limpio del chunk.
        :return: Un array de numpy representando el embedding.
        """
        if not self.model:
            raise RuntimeError("El modelo de embeddings no se cargo correctamente.")

        embedding = self.model.encode(
            text,
            prompt=self.PASSAGE_PROMPT,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_tensor=False,
        )
        return embedding

    def _prepare_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Prepara los chunks: copia, asigna chunk_id faltantes, separa textos validos de vacios.
        Retorna (chunks_copy, texts_validos, posiciones_validas, count_vacios).
        """
        chunks_copy = [chunk.copy() for chunk in chunks]
        texts: List[str] = []
        valid_positions: List[int] = []
        empty_count = 0

        for i, chunk in enumerate(chunks_copy):
            metadata = chunk.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}
                chunk["metadata"] = metadata

            if not metadata.get("chunk_id"):
                metadata["chunk_id"] = str(uuid.uuid4())

            text = chunk.get("cleaned_content", "")
            if not text or not text.strip():
                chunk["embedding"] = []
                empty_count += 1
                continue

            valid_positions.append(i)
            texts.append(text)

        return chunks_copy, texts, valid_positions, empty_count

    def _resolve_devices(self) -> List[str]:
        """Resuelve la lista de dispositivos para multi-process pool."""
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(min(self.num_workers, n_gpus))]
            if self.num_workers > n_gpus:
                print(f"Advertencia: {self.num_workers} workers solicitados pero solo {n_gpus} GPU(s). Usando {n_gpus}.")
        else:
            devices = ["cpu"] * self.num_workers
        return devices

    def generate_embeddings_for_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Genera embeddings para una lista de elementos limpios.
        Si num_workers > 1, usa multi-process pool para paralelizar el encoding.
        Cada elemento debe tener 'cleaned_content' y 'metadata' con 'chunk_id'.
        """
        if not self.model:
            print("Advertencia: No se generaran embeddings porque el modelo no esta cargado.")
            return chunks

        chunks_out, texts, valid_positions, empty_count = self._prepare_chunks(chunks)
        total = len(chunks)
        valid_total = len(texts)

        print(
            f"Procesando {valid_total}/{total} elementos con contenido "
            f"(vacios: {empty_count}, batch_size={self.batch_size}, workers={self.num_workers})"
        )

        if valid_total == 0:
            print("No hay contenido valido para generar embeddings.")
            return chunks_out

        t0 = time.perf_counter()

        if self.num_workers > 1:
            devices = self._resolve_devices()
            print(f"Iniciando multi-process pool con dispositivos: {devices}")
            pool = self.model.start_multi_process_pool(target_devices=devices)
            try:
                embeddings = self.model.encode(
                    texts,
                    prompt=self.PASSAGE_PROMPT,
                    batch_size=self.batch_size,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=True,
                    convert_to_tensor=False,
                    pool=pool,
                )
            finally:
                self.model.stop_multi_process_pool(pool)
        else:
            embeddings = self.model.encode(
                texts,
                prompt=self.PASSAGE_PROMPT,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                show_progress_bar=True,
                convert_to_tensor=False,
            )

        elapsed = time.perf_counter() - t0
        rate = valid_total / elapsed if elapsed > 0 else 0

        for idx, pos in enumerate(valid_positions):
            emb = embeddings[idx]
            chunks_out[pos]["embedding"] = emb.tolist() if hasattr(emb, "tolist") else emb

        print(
            f"\nProcesamiento completo: {valid_total} embeddings en {elapsed:.1f}s "
            f"({rate:.1f} elem/s)"
        )
        return chunks_out


if __name__ == "__main__":
    import argparse
    import glob
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Generador E5 de embeddings para chunks legales")
    # --- I/O ---
    parser.add_argument("--input-file", default=None,
                        help="Ruta del JSON con cleaned_elements. Default: .data/cleaned_content/cleaned_elements.json")
    parser.add_argument("--output-file", default=None,
                        help="Ruta del JSON de salida con embeddings E5")
    parser.add_argument("--stats-file", default=None,
                        help="Ruta del JSON de estadisticas E5")
    # --- Rendimiento ---
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Tamano de lote para encode (default: 64)")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Procesos paralelos (default: 1). >1 activa multi-process pool.")
    parser.add_argument("--fp16", action="store_true",
                        help="Usar float16 para inferencia (solo acelera en GPU)")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="Normalizar embeddings a norma unitaria (default: True, --no-normalize para desactivar)")
    # --- Checkpointing ---
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Guardar checkpoint cada N elementos procesados (0=deshabilitado, default: 0)")
    parser.add_argument("--resume", action="store_true",
                        help="Reanudar desde el ultimo checkpoint disponible")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cleaned_elements_path = args.input_file or os.path.join(
        project_root, ".data", "cleaned_content", "cleaned_elements.json"
    )
    output_dir = os.path.join(project_root, ".data", "embeddings")
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    # --- Cargar elementos ---
    if not os.path.exists(cleaned_elements_path):
        print(f"Error: No se encontro el archivo de elementos limpios en {cleaned_elements_path}")
        raise SystemExit(1)

    print(f"Cargando elementos desde {cleaned_elements_path}...")
    with open(cleaned_elements_path, "r", encoding="utf-8") as f:
        cleaned_elements = json.load(f)
    print(f"Cargados {len(cleaned_elements)} elementos limpios.")

    # --- Checkpointing: reanudar si se solicita ---
    start_offset = 0
    already_processed: List[Dict[str, Any]] = []

    if args.resume and os.path.isdir(checkpoint_dir):
        ckpt_files = sorted(glob.glob(os.path.join(checkpoint_dir, "e5_checkpoint_*.json")))
        if ckpt_files:
            latest_ckpt = ckpt_files[-1]
            print(f"Reanudando desde checkpoint: {latest_ckpt}")
            with open(latest_ckpt, "r", encoding="utf-8") as f:
                already_processed = json.load(f)
            start_offset = len(already_processed)
            print(f"Checkpoint contiene {start_offset} elementos. Continuando desde elemento {start_offset}.")

    remaining_elements = cleaned_elements[start_offset:]

    if not remaining_elements:
        print("Todos los elementos ya fueron procesados (checkpoint completo).")
        elements_with_embeddings = already_processed
    else:
        # --- Crear generador ---
        print("\nIniciando generacion de embeddings con modelo E5...")
        generator = EmbeddingGeneratorE5(
            batch_size=args.batch_size,
            normalize_embeddings=args.normalize,
            num_workers=args.num_workers,
            use_fp16=args.fp16,
        )

        # --- Checkpointing incremental durante procesamiento ---
        checkpoint_every = args.checkpoint_every
        if checkpoint_every > 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"Checkpointing habilitado: cada {checkpoint_every} elementos en {checkpoint_dir}")

            # Procesar en segmentos de checkpoint_every elementos
            all_new_results: List[Dict[str, Any]] = []
            for seg_start in range(0, len(remaining_elements), checkpoint_every):
                seg_end = min(seg_start + checkpoint_every, len(remaining_elements))
                segment = remaining_elements[seg_start:seg_end]
                print(f"\n--- Segmento {seg_start}–{seg_end} de {len(remaining_elements)} ---")

                seg_results = generator.generate_embeddings_for_chunks(segment)
                all_new_results.extend(seg_results)

                # Guardar checkpoint acumulado (already_processed + lo nuevo hasta ahora)
                cumulative = already_processed + all_new_results
                ckpt_path = os.path.join(checkpoint_dir, f"e5_checkpoint_{start_offset + seg_end:06d}.json")
                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(cumulative, f, ensure_ascii=False)
                print(f"Checkpoint guardado: {ckpt_path} ({len(cumulative)} elementos)")

            elements_with_embeddings = already_processed + all_new_results
        else:
            # Sin checkpointing: procesar todo de una vez
            new_results = generator.generate_embeddings_for_chunks(remaining_elements)
            elements_with_embeddings = already_processed + new_results

    # --- Estadisticas ---
    total_elements = len(elements_with_embeddings)
    valid_embeddings = sum(
        1 for elem in elements_with_embeddings
        if elem.get("embedding") and len(elem["embedding"]) > 0
    )
    empty_embeddings = total_elements - valid_embeddings

    print("\n--- Estadisticas de Embeddings E5 ---")
    print(f"Total de elementos procesados: {total_elements}")
    print(f"Embeddings generados exitosamente: {valid_embeddings}")
    print(f"Embeddings vacios (por errores o contenido vacio): {empty_embeddings}")

    # --- Guardar resultado final ---
    output_embeddings_path = args.output_file or os.path.join(output_dir, "e5_elements_with_embeddings.json")
    print(f"\nGuardando elementos con embeddings E5 en: {output_embeddings_path}")
    with open(output_embeddings_path, "w", encoding="utf-8") as f:
        json.dump(elements_with_embeddings, f, indent=2, ensure_ascii=False)

    # --- Guardar stats ---
    stats_path = args.stats_file or os.path.join(output_dir, "e5_embedding_stats.json")
    stats = {
        "total_elements": total_elements,
        "valid_embeddings": valid_embeddings,
        "empty_embeddings": empty_embeddings,
        "embedding_dimension": (
            len(elements_with_embeddings[0]["embedding"]) if valid_embeddings > 0 else 0
        ),
        "model_name": "intfloat/multilingual-e5-base",
        "batch_size": args.batch_size,
        "num_workers": args.num_workers or 1,
        "fp16": args.fp16,
        "normalize_embeddings": args.normalize,
        "processed_at": datetime.now().isoformat(),
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # --- Limpiar checkpoints tras exito ---
    if os.path.isdir(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
        print("Checkpoints limpiados tras finalizacion exitosa.")

    print(f"\nProceso completado. Estadisticas guardadas en: {stats_path}")
