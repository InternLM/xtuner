from typing import Any, Iterator
import types
from importlib import import_module
mock = import_module("sphinx.ext.autodoc.mock")


class _MockObject:
    """Used by autodoc_mock_imports."""
    __display_name__ = '_MockObject'
    __name__ = ''
    __sphinx_mock__ = True
    __sphinx_decorator_args__: tuple[Any, ...] = ()
    # Attributes listed here should not be mocked and rather raise an Attribute error:
    __sphinx_empty_attrs__: set[str] = {'__typing_subst__'}

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:  # NoQA: ARG004
        if len(args) == 3 and isinstance(args[1], tuple):
            superclass = args[1][-1].__class__
            if superclass is cls:
                # subclassing MockObject
                return _make_subclass(
                    args[0],
                    superclass.__display_name__,
                    superclass=superclass,
                    attributes=args[2],
                )
        return super().__new__(cls)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.__qualname__ = self.__name__
    def __len__(self) -> int:
        return 0
    def __contains__(self, key: str) -> bool:
        return False
    def __iter__(self) -> Iterator[Any]:
        return iter(())
    def __mro_entries__(self, bases: tuple[Any, ...]) -> tuple[type, ...]:
        return (self.__class__,)
    def __getitem__(self, key: Any):
        return _make_subclass(str(key), self.__display_name__, self.__class__)()

    def __getattr__(self, key: str):
        if key in self.__sphinx_empty_attrs__:
            raise AttributeError
        return _make_subclass(key, self.__display_name__, self.__class__)()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        call = self.__class__()
        call.__sphinx_decorator_args__ = args
        return call
    def __repr__(self) -> str:
        return self.__display_name__

    def __or__(self, value: Any, /) -> types.UnionType:
        return value | self.__class__

def _make_subclass(name: str, module: str, superclass: Any = _MockObject,
                   attributes: Any = None, decorator_args: tuple = ()) -> Any:
    attrs = {'__module__': module,
             '__display_name__': module + '.' + name,
             '__name__': name,
             '__sphinx_decorator_args__': decorator_args}
    attrs.update(attributes or {})

    return type(name, (superclass,), attrs)


mock._make_subclass = _make_subclass  # type: ignore[attr-defined])
