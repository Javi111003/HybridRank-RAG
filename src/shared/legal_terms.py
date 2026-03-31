"""Shared legal vocabulary for Spanish legal-text processing."""

LEGAL_TERMS_TO_KEEP = {
    "ley", "decreto", "artículo", "inciso", "apartado", "resolución",
    "ordenanza", "disposición", "reglamento", "norma", "código",
    "constitución", "derecho", "deber", "obligación", "responsabilidad"
}

CUSTOM_STOPWORDS = {
    "ley", "decreto", "resolución", "artículo", "inciso", "apartado", "literal",
    "número", "año", "gaceta", "oficial", "cuba", "cubano", "cubana", "república",
    "ministerio", "consejo", "estado", "gobierno", "poder", "popular", "nacional"
}
