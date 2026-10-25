"""Descriptor compatibility for supported ONNX serialization runtimes."""


def is_repeated(field):
    """Prefer the current API, retaining the protobuf 3.20 export environment."""
    try:
        return field.is_repeated
    except AttributeError:
        return field.label == field.LABEL_REPEATED
