from typing import Callable, ParamSpec, TypeVar

from typing_extensions import Concatenate


P = ParamSpec("P")
C = TypeVar("C")
T = TypeVar("T")


def copy_signature(f: Callable[P, T]):
    def identity(func) -> Callable[P, T]:
        return func

    return identity


def copy_method_signature(f: Callable[Concatenate[C, P], T]):
    def identity(func) -> Callable[P, T]:
        return func

    return identity
