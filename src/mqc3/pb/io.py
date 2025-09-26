"""Protobuf I/O utilities."""

from pathlib import Path
from typing import Literal, TypeVar

from google.protobuf import json_format, text_format
from google.protobuf.message import Message

ProtoFormat = Literal["text", "json", "binary"]
T = TypeVar("T", bound=Message)


def save(proto: Message, path: str | Path, proto_format: ProtoFormat = "text") -> None:
    """Save a Protocol Buffers message to a file.

    Args:
        proto: Protobuf message instance to save.
        path: Destination file path.
        proto_format: Output format. One of ``"text"``, ``"json"``, or ``"binary"``.

    Raises:
        ValueError: If ``proto_format`` is not one of the supported values.
    """
    p = Path(path)

    if proto_format == "text":
        p.write_text(text_format.MessageToString(proto), encoding="utf-8")
        return
    if proto_format == "json":
        p.write_text(
            json_format.MessageToJson(proto, preserving_proto_field_name=True),
            encoding="utf-8",
        )
        return
    if proto_format == "binary":
        p.write_bytes(proto.SerializeToString())
        return

    msg = f"Unsupported proto format {proto_format}."
    raise ValueError(msg)


def load(cls: type[T], path: str | Path, proto_format: ProtoFormat = "text") -> T:  # noqa: UP047
    """Load a Protocol Buffers message from a file.

    This function constructs a new instance of ``cls`` and populates it from
    the file at ``path`` using the given ``proto_format``.

    Args:
        cls: The generated protobuf class (e.g., ``MyMessage``).
        path: Source file path.
        proto_format: Input format. One of ``"text"``, ``"json"``, or ``"binary"``.

    Returns:
        T: A new instance of type ``T`` populated from the file.

    Raises:
        ValueError: If ``proto_format`` is not supported, or if parsing fails.
    """
    p = Path(path)
    if proto_format == "text":
        return text_format.Parse(p.read_text(), cls())
    if proto_format == "json":
        return json_format.Parse(p.read_text(), cls())
    if proto_format == "binary":
        msg = cls()
        msg.ParseFromString(p.read_bytes())
        return msg

    msg = f"Unsupported proto format {proto_format}."
    raise ValueError(msg)
