"""Tests for the generic scalar value-model wrapper used by RL critics."""

from datetime import timedelta
from functools import partial
from itertools import chain
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import pytest
import torch

from xtuner.v1.model.compose.qwen3_5.qwen3_5_config import Qwen3_5_VLMoE35BA3Config
from xtuner.v1.model.dense.qwen3 import Qwen3DenseConfig
from xtuner.v1.model.moe.qwen3_5_text import Qwen3_5_VLTextMoE35BA3BConfig
from xtuner.v1.model.value import (
    HF_VALUE_HEAD_KEY,
    LOCAL_VALUE_HEAD_KEY,
    ValueModelMixin,
    as_value_config,
    wants_scalar_value_head,
)
from xtuner.v1.module.attention import MHAConfig
from xtuner.v1.utils import init_params


class _RankLocalMissingCheckpointModel:
    """Minimal base model that reproduces rank-local missing-key reports."""

    def from_hf(self, hf_path: str | Path, strict: bool = True) -> tuple[set[str], set[str], set[str]]:
        del hf_path, strict
        if self.lm_head.weight.to_local().shape[0]:
            return set(), {LOCAL_VALUE_HEAD_KEY}, {HF_VALUE_HEAD_KEY}
        return set(), set(), set()


class _DistributedValueModel(ValueModelMixin, _RankLocalMissingCheckpointModel):
    pass


def _run_distributed_missing_value_head_init(
    rank: int, world_size: int, rendezvous_path: str, checkpoint_path: str, hidden_size: int
) -> None:
    """Initialize a dim-0-sharded scalar head in a spawned worker."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        torch.manual_seed(0)
        head = torch.nn.Parameter(distribute_tensor(torch.zeros(1, hidden_size), mesh, placements=[Shard(0)]))
        # With one output row and two ranks, only rank 0 owns head storage.
        assert head.to_local().shape[0] == (1 if rank == 0 else 0)

        model = _DistributedValueModel()
        model.config = SimpleNamespace(hidden_size=hidden_size)
        model.lm_head = SimpleNamespace(weight=head)
        _, unloaded_keys, missing_keys = model.from_hf(checkpoint_path)

        assert unloaded_keys == set()
        assert missing_keys == set()
        # This is a collective too, so it also verifies that both ranks left
        # `from_hf` and still agree on the complete initialized parameter.
        full_head = head.full_tensor()
        assert bool(torch.isfinite(full_head).all())
        assert bool(torch.count_nonzero(full_head))
    finally:
        dist.destroy_process_group()


def _tiny_dense_config(**overrides) -> Qwen3DenseConfig:
    """A CPU-constructible dense config small enough for meta-device builds."""
    kwargs = dict(
        vocab_size=32,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        num_hidden_layers=1,
        hidden_size=16,
        intermediate_size=32,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention=MHAConfig(num_attention_heads=2, num_key_value_heads=1, head_dim=8, qk_norm=True),
    )
    kwargs.update(overrides)
    return Qwen3DenseConfig(**kwargs)


class TestValueConfigDerivation:
    def test_scalar_head_flag_is_set_and_source_is_not_mutated(self) -> None:
        actor_cfg = _tiny_dense_config()
        value_cfg = as_value_config(actor_cfg)

        assert wants_scalar_value_head(value_cfg)
        assert not wants_scalar_value_head(actor_cfg)

    def test_tied_embeddings_are_untied(self) -> None:
        # A [1, hidden_size] value head cannot alias the embedding matrix, so
        # tying must be disabled even when the actor ties.
        actor_cfg = _tiny_dense_config(tie_word_embeddings=True)
        value_cfg = as_value_config(actor_cfg)

        assert actor_cfg.tie_word_embeddings is True
        assert value_cfg.tie_word_embeddings is False

    def test_mesh_prefix_is_distinct_from_actor(self) -> None:
        value_cfg = as_value_config(_tiny_dense_config())
        assert value_cfg.mesh_prefix == "critic"

    def test_mtp_and_z_loss_are_disabled_for_moe(self) -> None:
        value_cfg = as_value_config(Qwen3_5_VLTextMoE35BA3BConfig())

        assert value_cfg.mtp_config is None
        assert value_cfg.z_loss_cfg is None
        # MoE routers still need load balancing in a critic.
        assert value_cfg.balancing_loss_cfg is not None

    def test_compose_config_converts_only_the_language_model(self) -> None:
        actor_cfg = Qwen3_5_VLMoE35BA3Config()
        value_cfg = as_value_config(actor_cfg)

        assert wants_scalar_value_head(value_cfg.text_config)
        assert value_cfg.text_config.mesh_prefix == "critic"
        # Vision tower and projector are shared verbatim.
        assert not wants_scalar_value_head(value_cfg.vision_config)
        assert not wants_scalar_value_head(value_cfg.projector_config)
        assert not wants_scalar_value_head(actor_cfg.text_config)

    def test_derived_config_class_is_reused(self) -> None:
        first = as_value_config(_tiny_dense_config())
        second = as_value_config(_tiny_dense_config())
        assert type(first) is type(second)


class TestValueModelHead:
    @pytest.mark.parametrize("tie_word_embeddings", [False, True])
    def test_head_is_scalar(self, tie_word_embeddings: bool) -> None:
        cfg = _tiny_dense_config(tie_word_embeddings=tie_word_embeddings)
        with torch.device("meta"):
            actor = cfg.build()
            critic = as_value_config(cfg).build()

        assert actor.lm_head.weight.shape == (cfg.vocab_size, cfg.hidden_size)
        assert critic.lm_head.weight.shape == (1, cfg.hidden_size)
        assert critic.lm_head.bias is None
        # The critic head must never be tied to the embedding matrix.
        assert critic.lm_head.weight is not critic.embed_tokens.weight

    def test_model_gains_value_mixin_without_touching_actor_class(self) -> None:
        cfg = _tiny_dense_config()
        with torch.device("meta"):
            actor = cfg.build()
            critic = as_value_config(cfg).build()

        assert isinstance(critic, ValueModelMixin)
        assert not isinstance(actor, ValueModelMixin)

    def test_head_emits_one_value_per_token(self) -> None:
        cfg = _tiny_dense_config()
        critic = as_value_config(cfg).build()
        critic.init_weights()

        hidden_states = torch.randn(1, 8, cfg.hidden_size)
        _, (values, _) = critic.lm_head(hidden_states, None)

        assert values.shape == (1, 8, 1)


class TestValueHeadInitialization:
    """`ValueModelMixin.from_hf` initializes a missing value head in place.

    `init_params` reaches `param.copy_(...)` only for a sharded (DTensor)
    parameter, and autograd rejects an in-place write to a leaf that requires
    grad -- which is exactly what a freshly built, trainable critic head is.
    A plain CPU tensor takes the other branch and hides the problem, so this
    exercises the DTensor path directly.
    """

    @staticmethod
    def _init_head(head: torch.Tensor, hidden_size: int) -> None:
        """Replicate what the mixin does for an absent value head."""
        with torch.no_grad():
            init_params(head, partial(torch.nn.init.normal_, mean=0.0, std=1.0 / (hidden_size + 1)))

    def test_head_is_a_trainable_leaf(self) -> None:
        # The precondition that makes the in-place write illegal.
        critic = as_value_config(_tiny_dense_config()).build()
        head = critic.lm_head.weight
        assert head.requires_grad
        assert head.is_leaf

    def test_missing_sharded_head_initializes_on_every_rank(self, tmp_path: Path) -> None:
        """All ranks enter initialization even when only rank 0 owns a row."""
        import torch.distributed as dist
        from safetensors.torch import save_file
        from torch.multiprocessing import spawn

        if not dist.is_available() or not dist.is_gloo_available():
            pytest.skip("Gloo distributed backend unavailable")

        checkpoint_path = tmp_path / "actor-checkpoint"
        checkpoint_path.mkdir()
        save_file({"backbone.weight": torch.ones(1)}, checkpoint_path / "model.safetensors")

        world_size = 2
        context = spawn(
            _run_distributed_missing_value_head_init,
            args=(
                world_size,
                str(tmp_path / "distributed-init"),
                str(checkpoint_path),
                _tiny_dense_config().hidden_size,
            ),
            nprocs=world_size,
            join=False,
        )
        deadline = monotonic() + 30
        completed = False
        try:
            while not completed and (remaining := deadline - monotonic()) > 0:
                completed = context.join(timeout=remaining)
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()

        assert completed, "distributed value-head initialization deadlocked"

    def test_in_place_init_needs_no_grad_for_sharded_params(self) -> None:
        import torch.distributed as dist

        if not dist.is_available():
            pytest.skip("torch.distributed unavailable")
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
        try:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.tensor import distribute_tensor

            mesh = init_device_mesh("cpu", (1,))
            cfg = _tiny_dense_config()
            head = torch.nn.Parameter(distribute_tensor(torch.zeros(1, cfg.hidden_size), mesh))
            assert head.requires_grad

            with pytest.raises(RuntimeError, match="leaf Variable that requires grad"):
                init_params(head, partial(torch.nn.init.normal_, mean=0.0, std=0.01))

            # Under no_grad the same call succeeds, which is what the mixin does.
            self._init_head(head, cfg.hidden_size)
            assert bool(torch.isfinite(head.to_local()).all())
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def test_init_uses_small_variance(self) -> None:
        # A default 0.02-std head emits large arbitrary values on step 0, which
        # GAE would then propagate into every advantage.
        cfg = _tiny_dense_config()
        critic = as_value_config(cfg).build()
        expected_std = 1.0 / (cfg.hidden_size + 1)
        self._init_head(critic.lm_head.weight, cfg.hidden_size)
        # Loose bound: a 16-wide head is a small sample.
        assert critic.lm_head.weight.std().item() < 5 * expected_std


class TestValueModelCheckpointKeys:
    def test_value_head_maps_to_its_own_key(self) -> None:
        with torch.device("meta"):
            critic = as_value_config(_tiny_dense_config()).build()

        assert critic.to_hf_key_list(LOCAL_VALUE_HEAD_KEY) == [HF_VALUE_HEAD_KEY]

    def test_backbone_keys_match_the_actor_exactly(self) -> None:
        """A critic must differ from its actor by the head key alone.

        This is what lets a critic initialize from a plain actor checkpoint:
        every backbone tensor is found, and only the value head is missing.
        """
        cfg = _tiny_dense_config()
        with torch.device("meta"):
            actor = cfg.build()
            critic = as_value_config(cfg).build()

        actor_keys = set(chain(*map(actor.to_hf_key_list, actor.state_dict())))
        critic_keys = set(chain(*map(critic.to_hf_key_list, critic.state_dict())))

        assert critic_keys - actor_keys == {HF_VALUE_HEAD_KEY}
        assert actor_keys - critic_keys == {"lm_head.weight"}
