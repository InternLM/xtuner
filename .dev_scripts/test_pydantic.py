#! /usr/bin/env python
import os
import sys


XTUNER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, XTUNER_ROOT)


# Write by GPT-5, hope it is correct.
# XTuner.v1 requires all pydantic.BaseModel forbit extra fields in model_config. This scritps is used to
# scan all pydantic.BaseModel subclasses in xtuner.v1 and check their model_config.extra field.


from pathlib import Path
path_root = Path(__file__).parent.parent / "xtuner" / "v1"

# BaseDataloader is an abstract base class. We only need to check its subclasses for model_config.
skip = ["BaseDataloaderConfig"]

basemodel_obj = []

for module in path_root.rglob("*.py"):
    if module.name == "__init__.py":
        continue
    relative_module = module.relative_to(path_root)
    module_parts = relative_module.with_suffix('').parts
    module_name = "xtuner.v1." + ".".join(module_parts)
    try:
        mod = __import__(module_name, fromlist=[''])
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")
        continue
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        from pydantic import BaseModel
        if isinstance(attr, type) and issubclass(attr, BaseModel):
            if "xtuner" in attr.__module__:
                if attr_name in skip:
                    print(f"{attr} skipped")
                    continue
                if not hasattr(attr, "model_config"):
                    raise AssertionError(f"{attr} missing model_config")
                if not "extra" in attr.model_config:
                    raise AssertionError(f"{attr} model_config missing extra")
                if attr.model_config["extra"] != "forbid":
                    raise AssertionError(f"{attr} model_config extra is not forbid")
            basemodel_obj.append((module_name, attr_name))
