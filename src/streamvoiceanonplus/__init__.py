"""Inference package. Model dependencies load only when an engine is constructed."""
__version__ = "0.2.0"


def __getattr__(name):
    if name == 'InferenceEngine':
        from .runtime import InferenceEngine
        return InferenceEngine
    if name == 'InferenceWrapper':
        from .legacy import InferenceWrapper
        return InferenceWrapper
    raise AttributeError(name)
