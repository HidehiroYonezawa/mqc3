"""Save proto safely."""

from pathlib import Path

from google.protobuf.message import Message

from mqc3.pb.io import ProtoFormat, save


def safe_save(proto: Message, path: Path, proto_format: ProtoFormat = "text") -> str | None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        save(proto, path, proto_format)
    except (FileNotFoundError, OSError) as e:
        return f"{e.__class__.__name__}: {e}"

    return None
