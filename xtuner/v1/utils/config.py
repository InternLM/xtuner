import importlib.util
from pathlib import Path
from typing import Union

from addict import Dict
from yaml import safe_dump, safe_load  # type: ignore[import-untyped]


class Config(Dict):
    """A facility for config and config files."""

    @classmethod
    def fromfile(cls, file: Union[str, Path]) -> "Config":
        """Build a Config instance from config file."""
        path = Path(file).expanduser().resolve()

        if path.suffix == ".py":
            spec = importlib.util.spec_from_file_location(str(path), path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load config from {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cfg_dict = {k: v for k, v in module.__dict__.items() if not k.startswith("__")}
        elif path.suffix in [".yaml", ".yml"]:
            with open(path) as f:
                cfg_dict = safe_load(f) or {}
        else:
            raise NotImplementedError("Only python/yaml type config are supported now!")
        return cls(**cfg_dict)

    def dumpfile(self, file: Union[str, Path]):
        """Dump a Config instance to config file."""
        path = Path(file).expanduser().resolve()
        if path.suffix == ".py":
            with open(path, "w") as f:
                f.write(self.pretty_text)
        elif path.suffix in [".yaml", ".yml"]:
            with open(path, "w") as f:
                safe_dump(self.to_dict(), f, sort_keys=False, indent=2)
        else:
            raise NotImplementedError("Only python/yaml type config are supported now!")

    @property
    def pretty_text(self) -> str:
        """Get formatted python config text."""

        def _format_dict(input_dict):
            use_mapping = not all(str(k).isidentifier() for k in input_dict)

            if use_mapping:
                item_tmpl = "{k}: {v}"
            else:
                item_tmpl = "{k}={v}"

            items = []
            for k, v in input_dict.items():
                v_str = _format_basic_types(v)
                k_str = _format_basic_types(k) if use_mapping else k
                items.append(item_tmpl.format(k=k_str, v=v_str))
            items = ",".join(items)

            if use_mapping:
                return "{" + items + "}"
            else:
                return f"dict({items})"

        def _format_list_tuple_set(input_container):
            items = []

            for item in input_container:
                items.append(_format_basic_types(item))

            if isinstance(input_container, tuple):
                items = items + [""] if len(items) == 1 else items
                return "(" + ",".join(items) + ")"
            elif isinstance(input_container, list):
                return "[" + ",".join(items) + "]"
            elif isinstance(input_container, set):
                return "{" + ",".join(items) + "}"

        def _format_basic_types(input_):
            if isinstance(input_, str):
                return repr(input_)
            elif isinstance(input_, dict):
                return _format_dict(input_)
            elif isinstance(input_, (list, set, tuple)):
                return _format_list_tuple_set(input_)
            else:
                return str(input_)

        items = []
        for k, v in self.items():
            items.append(f"{k} = {_format_basic_types(v)}")

        text = "\n".join(items)
        return text
