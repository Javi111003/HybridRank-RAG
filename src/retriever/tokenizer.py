"""
Tokenizador compartido para BM25.
Garantiza consistencia entre la indexacion y la busqueda.
"""

import spacy
from typing import List

try:
    from src.shared.legal_terms import LEGAL_TERMS_TO_KEEP
except ModuleNotFoundError:
    from ..shared.legal_terms import LEGAL_TERMS_TO_KEEP

try:
    _nlp = spacy.load("es_core_news_md", disable=["parser", "ner"])
except OSError:
    _nlp = None

class SpanishTokenizer:
    """
    Tokenizador basado en spaCy para BM25.
    Aplica: lowercase, lematizacion, eliminacion de stopwords y puntuacion.
    Preserva terminos legales importantes y nombres propios/numeros.

    Diseñado para usarse identicamente en indexacion y busqueda,
    garantizando que los terminos coincidan entre documento y query.
    """

    def __init__(self, nlp=None):
        self._nlp = nlp or _nlp
        if self._nlp is None:
            raise RuntimeError(
                "Modelo spaCy 'es_core_news_md' no disponible. "
                "Instalar con: python -m spacy download es_core_news_md"
            )

    def tokenize(self, text: str) -> List[str]:
        """
        Tokeniza un texto para BM25.

        :param text: Texto a tokenizar.
        :return: Lista de tokens (lemmas en minuscula, sin stopwords ni puntuacion).
        """
        doc = self._nlp(text)
        return self._extract_tokens(doc)

    def tokenize_batch(self, texts: List[str], batch_size: int = 1000, n_process: int = 1) -> List[List[str]]:
        """
        Tokeniza multiples textos en batch usando spaCy nlp.pipe().

        Mucho mas rapido que llamar tokenize() en un loop porque spaCy
        procesa internamente en batches y puede paralelizar con n_process.

        :param texts: Lista de textos a tokenizar.
        :param batch_size: Tamaño del batch interno de spaCy.
        :param n_process: Numero de procesos para paralelismo (1 = sin multiprocessing).
        :return: Lista de listas de tokens, una por cada texto de entrada.
        """
        results = []
        for doc in self._nlp.pipe(texts, batch_size=batch_size, n_process=n_process):
            results.append(self._extract_tokens(doc))
        return results

    def _extract_tokens(self, doc) -> List[str]:
        """Extrae tokens limpios de un documento spaCy."""
        tokens = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            token_lower = token.text.lower()
            if (not token.is_stop
                    or token_lower in LEGAL_TERMS_TO_KEEP
                    or token.pos_ in ('PROPN', 'NUM')):
                lemma = token.lemma_.lower().strip()
                if len(lemma) > 1:
                    tokens.append(lemma)
        return tokens
