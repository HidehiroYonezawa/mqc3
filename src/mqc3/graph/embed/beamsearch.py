"""Convert circuit representation to graph representation using beam search."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heappushpop
from typing import TYPE_CHECKING

from mqc3.graph.embed._search_state import SearchState
from mqc3.graph.embed.embed import GraphEmbedder, GraphEmbedResult, GraphEmbedSettings

if TYPE_CHECKING:
    from .dep_dag import DependencyDAG


@dataclass
class BeamSearchEmbedSettings(GraphEmbedSettings):
    """Settings for embedding a dependency DAG into a graph representation with beam search strategy."""

    beam_width: int = 10
    """The number of candidate solutions to keep at each step of the beam search.
    A larger value leads to a more precise solution but increases computation time."""


class BeamSearchEmbedder(GraphEmbedder):
    """Embed a dependency DAG into a graph representation with beam search strategy."""

    _settings: BeamSearchEmbedSettings

    def __init__(self, settings: BeamSearchEmbedSettings) -> None:
        """Initialize embedder with settings.

        Args:
            settings (BeamSearchEmbedSettings): Settings for embedding a dependency DAG into a GraphRepr.
        """
        super().__init__(settings)

    def _embed_impl(self, dep_dag: DependencyDAG) -> GraphEmbedResult:
        """Embed a Dependency DAG into a graph representation with beam search strategy.

        Args:
            dep_dag: Dependency DAG to embed.

        Raises:
            RuntimeError: If the embedding fails.

        Returns:
            GraphEmbedResult: Result of the embedding.
        """
        search_nodes: list[list[SearchState]] = [[]]
        search_nodes[0].append(SearchState(dep_dag, self._settings))
        cutoffs: list[int] = [0]
        length_of_search_nodes = 1
        next_macronode_index = 0

        while True:
            for bs_state in search_nodes[next_macronode_index]:
                if bs_state.is_all_done():
                    return GraphEmbedResult(graph=bs_state.output_graph())
                for next_state in bs_state.generate_next_states():
                    if next_state.index >= length_of_search_nodes:
                        search_nodes.extend([] for _ in range(length_of_search_nodes, next_state.index + 1))
                        cutoffs.extend(0 for _ in range(length_of_search_nodes, next_state.index + 1))
                        length_of_search_nodes = next_state.index + 1
                    next_heap = search_nodes[next_state.index]
                    if len(next_heap) < self._settings.beam_width:
                        heappush(next_heap, next_state)
                    else:
                        heappushpop(next_heap, next_state)
            next_macronode_index += 1
            if next_macronode_index == length_of_search_nodes:
                msg = "Failed to convert the circuit to graph representation using beam search."
                raise RuntimeError(msg)
