"""Utility."""

import logging
from collections.abc import Iterable
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)


K = TypeVar("K")
V = TypeVar("V")


class OneToOneDict(Generic[K, V]):  # noqa:UP046
    """A bidirectional dictionary that maintains a one-to-one mapping between keys and values.

    Each key maps to a unique value, and each value maps back to a unique key.
    """

    k_to_v: dict[K, V]
    """Dictionary mapping keys to values."""
    v_to_k: dict[V, K]
    """Dictionary mapping values back to keys."""

    _type_k: type[K]
    """The type of keys used in the dictionary."""
    _type_v: type[V]
    """The type of values used in the dictionary."""

    def _add_elements(self, k: K, v: V) -> None:
        """Adds a key-value pair to the dictionary, ensuring uniqueness in both directions.

        Args:
            k (K): The key to add.
            v (V): The value to add.

        Raises:
            ValueError: If the key or value already exists in the mapping.
        """
        if k in self.k_to_v:
            msg = f"Input {k} already exists in the mapping."
            raise ValueError(msg)
        if v in self.v_to_k:
            msg = f"Output {v} already exists in the mapping."
            raise ValueError(msg)

        self.k_to_v[k] = v
        self.v_to_k[v] = k

    def get_type_k(self) -> type[K]:
        """Returns the type of keys used in the dictionary."""
        return self._type_k

    def get_type_v(self) -> type[V]:
        """Returns the type of values used in the dictionary."""
        return self._type_v

    def __init__(self, processor: Iterable[tuple[K, V]]) -> None:
        """Initializes the OneToOneDict with an iterable of key-value pairs.

        Args:
            processor (Iterable[tuple[K, V]]): An iterable containing key-value pairs.

        Raises:
            ValueError: If duplicate keys or values are found in the input.
        """  # noqa: DOC502
        self.k_to_v = {}
        self.v_to_k = {}
        for k, v in processor:
            self._add_elements(k, v)

        self._type_k = type(next(iter(self.k_to_v)))
        self._type_v = type(next(iter(self.v_to_k)))

    def get_k(self, v: V) -> K:
        """Retrieves the key corresponding to a given value.

        Args:
            v (V): The value to look up.

        Returns:
            K: The corresponding key.

        Raises:
            KeyError: If the value is not found in the dictionary.
        """  # noqa: DOC502
        return self.v_to_k[v]

    def get_v(self, t: K) -> V:
        """Retrieves the value corresponding to a given key.

        Args:
            t (K): The key to look up.

        Returns:
            V: The corresponding value.

        Raises:
            KeyError: If the key is not found in the dictionary.
        """  # noqa: DOC502
        return self.k_to_v[t]

    def __getitem__(self, key: K | V) -> V | K:
        """Retrieves the corresponding value for a given key or vice versa.

        If the types of keys and values are the same, a warning is logged.

        Args:
            key (K | V): The key or value to look up.

        Returns:
            V | K: The corresponding value if a key is provided, or the key if a value is provided.

        Raises:
            KeyError: If the key or value is not found in the dictionary.
        """
        if self._type_k == self._type_v:
            logger.warning(
                "The types of the key and value are the same. `__getitem__` will return the value of the key."
            )

        if isinstance(key, self._type_k):
            return self.get_v(key)
        if isinstance(key, self._type_v):
            return self.get_k(key)

        msg = f"Key {key} not found in the mapping."
        raise KeyError(msg)
