import inspect
import json
import pydoc

import torch
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import CoreSchema

from xtuner.v1.datasets import *


TOKENIZE_FN_CONFIGS: set[str] = set()

for imported in list(locals().values()):
    if (
        inspect.isclass(imported)
        and (imported.__name__.endswith("TokenizeFnConfig") or imported.__name__.endswith("TokenizeFunctionConfig"))
        and imported is not BaseTokenizeFnConfig
    ):
        TOKENIZE_FN_CONFIGS.add(f"{imported.__module__}.{imported.__name__}")


class SafeGenerateJsonSchema(GenerateJsonSchema):
    """安全的 schema 生成器，遇到无法处理的类型不抛异常."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._processing_base_tokenize_fn = False

    def is_instance_schema(self, schema: CoreSchema) -> JsonSchemaValue:
        cls = schema.get("cls")

        if cls is BaseTokenizeFnConfig and not self._processing_base_tokenize_fn:
            self._processing_base_tokenize_fn = True
            try:
                refs = []
                for full_name in TOKENIZE_FN_CONFIGS:
                    config_cls = pydoc.locate(full_name)
                    core_schema = config_cls.__pydantic_core_schema__
                    json_schema = self.generate_inner(core_schema)
                    refs.append(json_schema)
                return {"anyOf": refs}
            finally:
                self._processing_base_tokenize_fn = False
        elif cls is torch.dtype:
            return {"type": "string", "enum": ["float32", "bfloat16"], "title": "torch.dtype"}
        else:
            raise RuntimeError(f"Unrecorgnized schema for {schema}")

    def model_schema(self, schema: CoreSchema) -> JsonSchemaValue:
        json_schema = super().model_schema(schema)
        cls = schema.get("cls")
        if cls:
            json_schema["title"] = f"{cls.__module__}.{cls.__name__}"
        return json_schema


def remove_field(schema: dict, field_name: str) -> dict:
    """从 schema 中移除指定字段。

    Args:
        schema (dict): JSON Schema
        field_name (str): 要移除的字段名

    Returns:
        dict: 修改后的 schema
    """
    if "properties" in schema:
        schema["properties"].pop(field_name, None)
    if "required" in schema and field_name in schema["required"]:
        schema["required"].remove(field_name)
    return schema


if __name__ == "__main__":
    from xtuner.v1.train.arguments import TrainingArguments

    schema = TrainingArguments.model_json_schema(schema_generator=SafeGenerateJsonSchema)
    remove_field(schema, "model_cfg")

    with open("/tmp/trainconfig.json", "w") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
