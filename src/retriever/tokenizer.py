"""
Tokenizador compartido para BM25.
Garantiza consistencia entre la indexacion y la busqueda.
"""

import spacy
from typing import List

try:
    _nlp = spacy.load("es_core_news_md", disable=["parser", "ner"])
except OSError:
    _nlp = None

# Terminos legales importantes que NO deben eliminarse como stopwords
# (mismo conjunto que text_cleaner.py)
_LEGAL_TERMS = {
    'ley', 'decreto', 'artículo', 'inciso', 'apartado', 'resolución',
    'ordenanza', 'disposición', 'reglamento', 'norma', 'código',
    'constitución', 'derecho', 'deber', 'obligación', 'responsabilidad'
}


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
        tokens = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            token_lower = token.text.lower()
            if (not token.is_stop
                    or token_lower in _LEGAL_TERMS
                    or token.pos_ in ('PROPN', 'NUM')):
                lemma = token.lemma_.lower().strip()
                if len(lemma) > 1:
                    tokens.append(lemma)
        return tokens
