"""Graph embedder class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqc3.graph import GraphRepr
    from mqc3.graph.embed.dep_dag import DependencyDAG


@dataclass
class GraphEmbedSettings(ABC):
    """Settings for embedding a dependency graph into a concrete GraphRepr."""

    n_local_macronodes: int
    "the number of local macronodes in graph representation"
    feedforward_distance: tuple[int, int] = (0, 10**9)


@dataclass(kw_only=True)
class GraphEmbedResult:
    """Result of embedding."""

    graph: GraphRepr


class GraphEmbedder(ABC):
    """Abstract base class for embedding a dependency DAG into a concrete GraphRepr.

    This class defines the template method pattern for graph embedding, where
    the specific embedding algorithm is implemented by subclasses.
    """

    _settings: GraphEmbedSettings
    result: GraphEmbedResult | None

    def __init__(self, settings: GraphEmbedSettings) -> None:
        """Initialize embedder with the settings.

        Args:
            settings (GraphEmbedSettings): Settings for the embedding process.
        """
        self._settings = settings
        self.result = None

    def embed(self, dep_dag: DependencyDAG) -> GraphRepr:
        """Embeds the given dependency DAG into a concrete GraphRepr.

        Args:
            dep_dag (DependencyDAG): The dependency DAG to be embedded.

        Returns:
            GraphRepr: The resulting embedded graph representation.
        """
        self.result = self._embed_impl(dep_dag)
        return self.result.graph

    @abstractmethod
    def _embed_impl(self, dep_dag: DependencyDAG) -> GraphEmbedResult:
        """Abstract method to implement the specific graph embedding algorithm.

        Args:
            dep_dag (DependencyDAG): The dependency DAG to be embedded.

        Returns:
            GraphEmbedResult: The result of embedding.
        """
