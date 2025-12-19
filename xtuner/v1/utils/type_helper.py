from typing import TYPE_CHECKING, Any, Awaitable, Callable, Generic, ParamSpec, TypeVar, overload

from typing_extensions import Concatenate


if TYPE_CHECKING:
    from ray import ObjectRef


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


class RemoteMethod(Generic[P, T]):
    def remote(self, *args: P.args, **kwargs: P.kwargs) -> "ObjectRef[T]": ...

    def bind(self, *args: P.args, **kwargs: P.kwargs) -> Any: ...


@overload
def ray_method(f: Callable[Concatenate[C, P], Awaitable[T]]) -> RemoteMethod[P, T]: ...


@overload
def ray_method(f: Callable[Concatenate[C, P], T]) -> RemoteMethod[P, T]: ...


def ray_method(f):
    import ray

    return ray.method(f)  # type: ignore[ret-type]
