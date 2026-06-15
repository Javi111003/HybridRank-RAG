import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chainlit as cl

from src.config import config
from src.retriever import BM25Retriever, DenseRetriever, HybridRetriever
from src.retriever.fusion import get_fusion_strategy
from src.rag.pipeline import RAGPipeline
from src.rag.store.norma_store import NormaStore
from src.rag.context.context_builder import ContextBuilder
from src.rag.generator.registry import get_generator

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_pipeline() -> RAGPipeline:
    bm25 = BM25Retriever(index_dir=config.BM25_NORMA_INDEX_DIR)
    dense = DenseRetriever(
        chroma_dir=config.CHROMA_NORMA_DIR,
        collection_name=config.CHROMA_NORMA_COLLECTION,
    )

    fusion_strategy = get_fusion_strategy(config.FUSION_STRATEGY)
    retriever = HybridRetriever(
        retrievers={"bm25": bm25, "dense": dense},
        fusion_strategy=fusion_strategy,
        candidate_k=config.CANDIDATE_K,
    )

    generator = get_generator()
    store = NormaStore()
    context_builder = ContextBuilder(
        max_fragments=config.CONTEXT_MAX_FRAGMENTS,
        max_chars=config.CONTEXT_MAX_CHARS,
    )

    return RAGPipeline(
        retriever=retriever,
        store=store,
        context_builder=context_builder,
        generator=generator,
        top_k=config.TOP_K,
    )


def _build_sources_elements(fragments) -> list:
    """Construye los elementos de fuentes a partir de los fragmentos."""
    elements = []
    for i, frag in enumerate(fragments[:5], start=1):
        snippet = frag.content[:600]
        if len(frag.content) > 600:
            snippet += "..."
        el_content = (
            f"**{frag.citation_key()}**\n\n"
            f"| Campo | Valor |\n"
            f"|-------|-------|\n"
            f"| Organismo | {frag.organismo_emisor} |\n"
            f"| Score | {frag.score:.4f} |\n"
            f"| Fragment ID | {frag.fragment_id} |\n\n"
            f"---\n\n{snippet}"
        )
        elements.append(
            cl.Text(name=f"Fuente {i}", content=el_content, display="side")
        )
    return elements


async def _send_sources_with_button(fragments=None):
    """Envia un mensaje de fuentes con un boton para reabrir el panel."""
    if fragments:
        elements = _build_sources_elements(fragments)
        content = (
            f"**Fuentes** | {len(fragments)} fragmentos recuperados. "
            f"Haz clic en una fuente o usa el boton para abrir el panel."
        )
    else:
        elements = [
            cl.Text(
                name="Estado",
                content="**Aun no hay fuentes recuperadas.**\n\n"
                "Realiza una consulta para ver los fragmentos relevantes aqui.",
                display="side",
            )
        ]
        content = "**Fuentes** | Aun no hay fuentes. Realiza una consulta."

    actions = [
        cl.Action(
            name="ver_fuentes",
            label="Ver Fuentes",
            description="Abre el panel lateral con las fuentes",
            payload={"action": "open"},
        )
    ]

    msg = cl.Message(content=content, elements=elements, actions=actions)
    await msg.send()
    cl.user_session.set("last_fragments", fragments)


@cl.action_callback("ver_fuentes")
async def on_ver_fuentes(action: cl.Action):
    """Reabre el panel de fuentes cuando el usuario presiona el boton."""
    fragments = cl.user_session.get("last_fragments")

    if not fragments:
        elements = [
            cl.Text(
                name="Estado",
                content="**Aun no hay fuentes recuperadas.**\n\n"
                "Realiza una consulta para ver los fragmentos relevantes aqui.",
                display="side",
            )
        ]
        await cl.Message(
            content="Aun no hay fuentes. Realiza una consulta primero.",
            elements=elements,
        ).send()
    else:
        elements = _build_sources_elements(fragments)
        await cl.Message(
            content=f"**Fuentes de la ultima respuesta** | {len(fragments)} fragmentos.",
            elements=elements,
        ).send()


@cl.on_chat_start
async def start():
    try:
        msg = cl.Message(content="Inicializando sistema HybridRank RAG...")
        await msg.send()

        pipeline = _build_pipeline()
        cl.user_session.set("pipeline", pipeline)
        cl.user_session.set("last_fragments", None)

        welcome = (
            "## HybridRank RAG\n\n"
            "Sistema listo para consultas.\n\n"
            "| Componente | Configuracion |\n"
            "|------------|---------------|\n"
            f"| Retriever | `{pipeline._retriever.name}` |\n"
            f"| Generador | `{pipeline._generator.name}` |\n"
            f"| Top-K | `{pipeline._top_k}` |\n"
            f"| Fusion | `{config.FUSION_STRATEGY}` |\n\n"
            "Escribe tu consulta sobre legislacion cubana."
        )
        msg.content = welcome
        await msg.update()

    except Exception as e:
        logger.error("Error inicializando pipeline: %s", e, exc_info=True)
        await cl.Message(content=f"Error al inicializar: {e}").send()


@cl.on_message
async def handle_message(message: cl.Message):
    pipeline: RAGPipeline | None = cl.user_session.get("pipeline")
    if pipeline is None:
        await cl.Message(content="Pipeline no inicializado. Recarga la pagina.").send()
        return

    query = message.content.strip()
    if not query:
        return

    thinking_msg = cl.Message(content="Buscando fuentes relevantes...")
    await thinking_msg.send()

    try:
        result = pipeline.run(query)

        thinking_msg.content = result.answer
        await thinking_msg.update()

        timing = (
            f"**Tiempos de ejecucion:**\n"
            f"- Retrieval: {result.retrieval_time_ms:.0f}ms\n"
            f"- Generacion: {result.generation_time_ms:.0f}ms\n"
            f"- Total: {result.total_time_ms:.0f}ms\n"
        )

        usage = result.generation_result.usage
        if usage:
            timing += (
                f"\n**Tokens:** {usage.get('prompt_tokens', '?')} prompt "
                f"+ {usage.get('completion_tokens', '?')} completion "
                f"= {usage.get('total_tokens', '?')} total"
            )

        await cl.Message(content=timing).send()

        if result.fragments:
            await _send_sources_with_button(result.fragments)
        else:
            await _send_sources_with_button(None)

    except Exception as e:
        logger.error("Error en pipeline: %s", e, exc_info=True)
        thinking_msg.content = f"Error procesando consulta: {e}"
        await thinking_msg.update()
