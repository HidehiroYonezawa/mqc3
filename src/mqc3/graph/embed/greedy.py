"""Greedy embedder from circuit representation to graph representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mqc3.graph.embed._search_state import SearchState
from mqc3.graph.embed.embed import GraphEmbedder, GraphEmbedResult, GraphEmbedSettings

if TYPE_CHECKING:
    from mqc3.graph.embed.dep_dag import DependencyDAG


@dataclass
class GreedyEmbedSettings(GraphEmbedSettings):
    """Settings for embedding a dependency DAG to a graph representation with greedy strategy."""


class GreedyEmbedder(GraphEmbedder):
    """Embed dependency DAG into graph representation with greedy strategy."""

    def __init__(self, settings: GreedyEmbedSettings) -> None:
        """Initialize greedy embedder with settings."""
        super().__init__(settings)

    def _advance_state_greedily(self, state: SearchState) -> SearchState:
        """Advance search state by one operation with greedy strategy.

        Returns:
            SearchState: Greedily selected state

        Raises:
            RuntimeError: Failed to generate next states
        """
        best_state = None
        for next_state in state.generate_next_states():
            if (
                best_state is None
                or next_state.evaluate() > best_state.evaluate()
                or (next_state.evaluate() == best_state.evaluate() and next_state.index < best_state.index)
            ):
                best_state = next_state
        if best_state is None:
            msg = "Failed to generate next states"
            raise RuntimeError(msg)
        return best_state

    def _embed_impl(self, dep_dag: DependencyDAG) -> GraphEmbedResult:
        state = SearchState(dep_dag, self._settings)
        while not state.is_all_done():
            state = self._advance_state_greedily(state)
        return GraphEmbedResult(graph=state.output_graph())
