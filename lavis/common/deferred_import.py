"""Defer a native dependency until its first operation is actually requested."""
from importlib import import_module
from types import ModuleType


class DeferredModule(ModuleType):
    """Load the original module on first attribute access, without a fallback."""

    def __getattr__(self, name):
        module = self.__dict__.get("_loaded_module")
        if module is None:
            module = import_module(self.__name__)
            self.__dict__["_loaded_module"] = module
        return getattr(module, name)
