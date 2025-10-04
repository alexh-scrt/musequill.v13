"""Exception classes for the Musequill system."""

from .similarity import TopicExhaustionError, SimilarityCorpusError

__all__ = [
    "TopicExhaustionError",
    "SimilarityCorpusError"
]