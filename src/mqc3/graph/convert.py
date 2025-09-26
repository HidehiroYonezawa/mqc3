"""Converter: convert circuit representation to graph representation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from mqc3.circuit.program import CircuitRepr
from mqc3.graph.embed.beamsearch import BeamSearchEmbedder, BeamSearchEmbedSettings
from mqc3.graph.embed.dep_dag import DependencyDAG
from mqc3.graph.embed.embed import GraphEmbedder, GraphEmbedSettings
from mqc3.graph.embed.greedy import GreedyEmbedder, GreedyEmbedSettings
from mqc3.graph.program import GraphRepr

INF = 10**9


@dataclass
class CGConvertSettings:
    """Settings for converting from circuit representation to graph representation."""

    n_local_macronodes: int
    "the number of local macronodes in graph representation"
    feedforward_distance: tuple[int, int] = (0, INF)

    @abstractmethod
    def to_embed_setting(self) -> GraphEmbedSettings:
        """Converts this CGConvertSetting object into its corresponding GraphEmbedSetting object.

        Returns:
            GraphEmbedSettings: An instance of GraphEmbedSetting which corresponds to
                            this CGConvertSetting object.
        """


class CGConverter(ABC):
    """Abstract class of converter from circuit representation to graph representation."""

    _settings: CGConvertSettings

    def __init__(self, settings: CGConvertSettings) -> None:
        """Constructor.

        Args:
            settings (CGConvertSetting): Conversion settings.
        """
        self._settings = settings

    def convert(self, circuit: CircuitRepr) -> GraphRepr:
        """Run conversion.

        Args:
            circuit (CircuitRepr): Circuit representation.

        Returns:
            GraphRepr: Converted graph representation.

        Examples:
            >>> from mqc3.graph.convert import BeamSearchConverter, BeamSearchConvertSettings
            >>> from mqc3.circuit import CircuitRepr
            >>> from mqc3.circuit.ops import intrinsic
            >>> circuit = CircuitRepr("example")
            >>> circuit.Q(0) | intrinsic.PhaseRotation(1.0)
            [QuMode(id=0)]
            >>> circuit.Q(1) | intrinsic.PhaseRotation(2.0)
            [QuMode(id=1)]
            >>> circuit.Q(0, 1) | intrinsic.ControlledZ(3.0)
            [QuMode(id=0), QuMode(id=1)]
            >>> circuit.Q(0) | intrinsic.Measurement(0.0)
            Var([intrinsic.measurement] [0.0] [QuMode(id=0)])
            >>> circuit.Q(1) | intrinsic.Measurement(0.0)
            Var([intrinsic.measurement] [0.0] [QuMode(id=1)])
            >>> settings = BeamSearchConvertSettings(n_local_macronodes=2)
            >>> converter = BeamSearchConverter(settings)
            >>> graph = converter.convert(circuit)
        """
        dag = DependencyDAG(circuit)
        embedder = self._create_embedder()
        return embedder.embed(dag)

    @abstractmethod
    def _create_embedder(self) -> GraphEmbedder:
        """The corresponding embedder of this converter."""


@dataclass
class GreedyConvertSettings(CGConvertSettings):
    """Settings for converting a circuit representation into a graph representation with greedy strategy."""

    def to_embed_setting(self) -> GreedyEmbedSettings:
        """Converts this convert settings into its corresponding embed settings.

        Returns:
            GreedyEmbedSettings: An instance of GraphEmbedSetting which corresponds to
                            this CGConvertSetting object.
        """
        return GreedyEmbedSettings(
            n_local_macronodes=self.n_local_macronodes, feedforward_distance=self.feedforward_distance
        )


class GreedyConverter(CGConverter):
    """Convert circuit representation into graph representation with greedy strategy."""

    _settings: GreedyConvertSettings

    def __init__(self, settings: GreedyConvertSettings) -> None:
        """Initialize greedy converter with settings."""
        super().__init__(settings)

    def _create_embedder(self) -> GreedyEmbedder:
        return GreedyEmbedder(self._settings.to_embed_setting())


@dataclass
class BeamSearchConvertSettings(CGConvertSettings):
    """Settings for converting a circuit representation into a graph representation with beam search strategy."""

    beam_width: int = 10

    def to_embed_setting(self) -> BeamSearchEmbedSettings:
        """Converts this `BeamSearchConvertSetting` into corresponding `BeamSearchEmbedSetting`.

        Returns:
            BeamSearchEmbedSettings: An instance of GraphEmbedSetting which corresponds to
                            this CGConvertSetting object.
        """
        return BeamSearchEmbedSettings(
            n_local_macronodes=self.n_local_macronodes,
            feedforward_distance=self.feedforward_distance,
            beam_width=self.beam_width,
        )


class BeamSearchConverter(CGConverter):
    """Convert with beamsearch from an optical quantum circuit to a graph representation."""

    _settings: BeamSearchConvertSettings

    def __init__(self, settings: BeamSearchConvertSettings) -> None:
        """BeamSearchConverter constructor.

        Args:
            settings (BeamSearchConvertSettings): Convert settings.

        Example:
            >>> from mqc3.graph.convert import BeamSearchConverter, BeamSearchConvertSettings
            >>> settings = BeamSearchConvertSettings(n_local_macronodes=2)
            >>> converter = BeamSearchConverter(settings)
        """
        super().__init__(settings)

    def _create_embedder(self) -> BeamSearchEmbedder:
        return BeamSearchEmbedder(self._settings.to_embed_setting())
