from typing import TypeAlias

RetrievalResult: TypeAlias = tuple[str, float]
RetrievalResults: TypeAlias = list[RetrievalResult]
ResultsByRetriever: TypeAlias = dict[str, RetrievalResults]
