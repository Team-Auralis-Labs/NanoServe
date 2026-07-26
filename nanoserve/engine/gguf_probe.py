"""Probe optional llama-cpp-python GGUF runtime."""
from __future__ import annotations


def gguf_available() -> bool:
    try:
        import llama_cpp  # noqa: F401
        return True
    except ImportError:
        return False
