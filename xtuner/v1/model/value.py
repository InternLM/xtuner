"""Generic scalar value-model support for RL critics.

A PPO critic shares the actor's backbone but replaces the vocabulary ``lm_head``
with a scalar head that emits one value per token. Rather than hand-writing a
value variant for every architecture, this module derives one from any existing
:class:`~xtuner.v1.model.base.TransformerConfig` (or
:class:`~xtuner.v1.model.compose.base.BaseComposeConfig`) via
:func:`as_value_config`.

Example:
    Derive a critic configuration from an actor configuration::

        from xtuner.v1.model import get_model_config_from_hf
        from xtuner.v1.model.value import as_value_config

        actor_cfg = get_model_config_from_hf(model_path)
        critic_cfg = as_value_config(actor_cfg)
        critic = critic_cfg.build()
"""

from functools import partial
from pathlib import Path
from typing import Annotated, Any, TypeVar

import torch
from pydantic import create_model

from xtuner.v1.module import LMHead
from xtuner.v1.utils import init_params, log_rank0


# The scalar head reuses the ``lm_head`` attribute so the whole forward path
# (model forward -> LMHead.forward -> loss_ctx.forward) works unchanged. Only
# the checkpoint key differs, which keeps critic checkpoints distinguishable
# from actor ones and lets a critic initialize from a plain actor checkpoint.
LOCAL_VALUE_HEAD_KEY = "lm_head.weight"
HF_VALUE_HEAD_KEY = "value_head.weight"

#: Config attribute that switches :meth:`build_head` to a scalar head. Declared
#: here (rather than on every backbone config) so backbones only need a single
#: ``getattr`` and stay unaware of RL.
VALUE_HEAD_FLAG = "scalar_value_head"


def wants_scalar_value_head(config: Any) -> bool:
    """Whether ``config`` asks for a scalar value head instead of an LM head.

    Args:
        config (Any): A model configuration.

    Returns:
        bool: ``True`` for critic configurations built by :func:`as_value_config`.
    """
    return bool(getattr(config, VALUE_HEAD_FLAG, False))


def build_lm_or_value_head(config: Any) -> LMHead:
    """Build either the vocabulary head or a scalar value head.

    Shared by the Dense and MoE ``build_head`` hooks.

    Args:
        config (Any): Model configuration.

    Returns:
        LMHead: ``hidden_size -> 1`` for critics, ``hidden_size -> vocab_size``
        otherwise. Both are bias-free.
    """
    out_features = 1 if wants_scalar_value_head(config) else config.vocab_size
    return LMHead(config.hidden_size, out_features, bias=False)


class ValueModelMixin:
    """Checkpoint-key handling for a scalar value model.

    Mix this in *before* the backbone class so its overrides win method
    resolution. The head itself is built by :func:`build_lm_or_value_head`,
    driven by the config flag, because the head must exist before the instance
    is fully constructed.
    """

    def to_hf_key_list(self, key: str) -> list[str]:
        """Map the scalar head to its own checkpoint key.

        Args:
            key (str): XTuner parameter key.

        Returns:
            list[str]: Corresponding HuggingFace checkpoint keys.
        """
        if key == LOCAL_VALUE_HEAD_KEY:
            return [HF_VALUE_HEAD_KEY]
        return super().to_hf_key_list(key)  # type: ignore[misc]

    def from_hf(
        self, hf_path: str | Path, strict: bool = True
    ) -> tuple[
        Annotated[set[str], "loaded keys"],
        Annotated[set[str], "unloaded keys"],
        Annotated[set[str], "missing keys"],
    ]:
        """Load a critic checkpoint, or initialize from an actor checkpoint.

        An actor checkpoint holds a vocabulary ``lm_head`` and no scalar value
        head. Because the value head maps to its own checkpoint key, the
        incompatible vocabulary tensor is never read. When that key is absent
        the scalar head is freshly initialized with the small-variance normal
        from Open-Reasoner-Zero; a real critic checkpoint loads it normally.

        Args:
            hf_path (str | Path): HuggingFace checkpoint path.
            strict (bool): Whether missing parameters should raise.

        Returns:
            tuple[set[str], set[str], set[str]]: Loaded, unloaded and missing keys.
        """
        # Load non-strict so an absent value head does not abort the backbone
        # load; strictness is re-applied below once that key is accounted for.
        loaded_keys, unloaded_keys, missing_keys = super().from_hf(hf_path, strict=False)  # type: ignore[misc]

        if HF_VALUE_HEAD_KEY in missing_keys:
            hidden_size = self.config.hidden_size  # type: ignore[attr-defined]
            # A default 0.02-std head emits large arbitrary values on step 0,
            # which GAE then propagates into every advantage.
            value_head_std = 1.0 / (hidden_size + 1)
            # `init_params` writes in place, which autograd forbids on a leaf
            # that requires grad. Every other initializer in the codebase runs
            # under no_grad for the same reason.
            with torch.no_grad():
                init_params(
                    self.lm_head.weight,  # type: ignore[attr-defined]
                    partial(torch.nn.init.normal_, mean=0.0, std=value_head_std),
                )
            unloaded_keys.discard(LOCAL_VALUE_HEAD_KEY)
            missing_keys.discard(HF_VALUE_HEAD_KEY)
            log_rank0.info(f"Initialized missing critic value head with Normal(mean=0, std={value_head_std:.6g})")

        if strict and missing_keys:
            raise RuntimeError(f"Missing parameters from {hf_path}: {sorted(missing_keys)}")

        return loaded_keys, unloaded_keys, missing_keys


_VALUE_MODEL_CLS_CACHE: dict[type, type] = {}
_VALUE_CONFIG_CLS_CACHE: dict[type, type] = {}


def bind_value_model_cls(model: Any) -> Any:
    """Attach :class:`ValueModelMixin` to a freshly built backbone instance.

    The head shape is already correct (``build_head`` reads the config flag);
    this only installs the checkpoint-key overrides. Rebinding ``__class__`` is
    safe because neither ``nn.Module`` nor XTuner's ``BaseModel`` define
    ``__slots__``, so instance layout is unchanged.

    Args:
        model (Any): A backbone instance built from a value config.

    Returns:
        Any: The same instance, re-typed to a ``ValueModelMixin`` subclass.
    """
    model_cls = type(model)
    if issubclass(model_cls, ValueModelMixin):
        return model
    value_cls = _VALUE_MODEL_CLS_CACHE.get(model_cls)
    if value_cls is None:
        value_cls = type(f"{model_cls.__name__}ValueModel", (ValueModelMixin, model_cls), {})
        _VALUE_MODEL_CLS_CACHE[model_cls] = value_cls
    model.__class__ = value_cls
    return model


def _value_config_cls(config_cls: type) -> type:
    """Build (and memoize) the value-config subclass of ``config_cls``.

    The subclass declares the scalar-head flag and wraps ``build`` so the
    resulting backbone gains the value-model checkpoint behavior.
    """
    cached = _VALUE_CONFIG_CLS_CACHE.get(config_cls)
    if cached is not None:
        return cached

    base_build = config_cls.build

    def build(self):  # type: ignore[no-untyped-def]
        return bind_value_model_cls(base_build(self))

    value_cls = create_model(
        f"{config_cls.__name__}Value",
        __base__=config_cls,
        **{VALUE_HEAD_FLAG: (bool, True)},  # type: ignore[arg-type]
    )
    value_cls.build = build  # type: ignore[method-assign]
    _VALUE_CONFIG_CLS_CACHE[config_cls] = value_cls
    return value_cls


ConfigT = TypeVar("ConfigT")


def as_value_config(config: ConfigT) -> ConfigT:
    """Derive a scalar critic configuration from an actor configuration.

    Works for both plain text configs and compose (VLM) configs; for the latter
    only ``text_config`` is converted, since the value head lives on the
    language model while the vision tower and projector are shared verbatim.

    The derived config forces several fields that are incompatible with, or
    meaningless for, a scalar head:

    * ``tie_word_embeddings=False`` -- tying binds ``lm_head.weight`` to the
      embedding matrix, which a ``[1, hidden_size]`` head cannot alias.
    * ``mtp_config=None`` -- multi-token prediction predicts vocabulary logits.
    * ``z_loss_cfg=None`` -- z-loss regularizes a vocabulary softmax.
    * ``mesh_prefix="critic"`` -- keeps the critic's device meshes from
      colliding with the actor's when both live in one process.

    ``balancing_loss_cfg`` is deliberately preserved: MoE routers still need
    load balancing in a critic.

    Args:
        config (ConfigT): The actor model configuration.

    Returns:
        ConfigT: A new configuration of a derived type that builds a value model.
            The input is not mutated.
    """
    fields = type(config).model_fields
    overrides: dict[str, Any] = {}

    if "text_config" in fields:
        # Compose/VLM: only the language model grows a value head.
        overrides["text_config"] = as_value_config(config.text_config)  # type: ignore[attr-defined]
        if "mesh_prefix" in fields:
            overrides["mesh_prefix"] = "critic"
        value_cls = _compose_value_config_cls(type(config))
    else:
        overrides["tie_word_embeddings"] = False
        overrides["mesh_prefix"] = "critic"
        if "mtp_config" in fields:
            overrides["mtp_config"] = None
        if "z_loss_cfg" in fields:
            overrides["z_loss_cfg"] = None
        value_cls = _value_config_cls(type(config))

    kwargs = {name: getattr(config, name) for name in fields}
    kwargs.update(overrides)
    return value_cls(**kwargs)  # type: ignore[return-value]


def _compose_value_config_cls(config_cls: type) -> type:
    """Value-config subclass for a compose (VLM) config.

    A compose config needs no scalar-head flag of its own -- the flag rides on
    ``text_config`` -- but it still needs a distinct type so ``build`` can
    widen the ``text_config`` field to accept the derived subclass.
    """
    cached = _VALUE_CONFIG_CLS_CACHE.get(config_cls)
    if cached is not None:
        return cached
    value_cls = create_model(f"{config_cls.__name__}Value", __base__=config_cls)
    _VALUE_CONFIG_CLS_CACHE[config_cls] = value_cls
    return value_cls
