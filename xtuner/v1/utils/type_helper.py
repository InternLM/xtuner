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


def ray_method(f=None, *, num_returns=1, concurrency_group=None):
    """Decorator for Ray actor methods.

    Compatible with Ray versions that require at least one of num_returns or concurrency_group. Ray.method() must be
    called with keyword args only, then applied to the function: ray.method(num_returns=1)(f).
    """
    import ray

    kwargs = {"num_returns": num_returns}
    if concurrency_group is not None:
        kwargs["concurrency_group"] = concurrency_group

    if f is None:
        # Called as @ray_method(num_returns=...) or @ray_method(concurrency_group=...)
        return lambda fn: ray.method(**kwargs)(fn)

    # Called as @ray_method
    return ray.method(**kwargs)(f)  # type: ignore[ret-type]
