"""Generic plugin registry.

A new planner / sensor / simulator backend can be added by writing a class
and decorating it with `@PLANNER_REGISTRY.register("my_name")`. The CLI then
picks it up via `type: my_name` in the YAML config — no central wiring needed.
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: Dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(f"{self._kind} '{name}' already registered")
            self._items[name] = cls
            return cls

        return deco

    def get(self, name: str) -> type[T]:
        if name not in self._items:
            raise KeyError(
                f"unknown {self._kind} '{name}'. available: {sorted(self._items)}"
            )
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)
