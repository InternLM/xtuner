from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from xtuner.v1.rl.weight_update import transport as module
from xtuner.v1.rl.weight_update.data import WeightUpdateBatch


class TestSharedStoreEpoch:
    @staticmethod
    def make_transport(monkeypatch: pytest.MonkeyPatch, shared: bool = True) -> module.IPCWeightTransport:
        t = module.IPCWeightTransport.__new__(module.IPCWeightTransport)
        t._shared_store_group = object() if shared else None
        t._shared_store_epoch = None
        t.cpu_group = object()
        t.head_rank = 0
        t.rollout_info = SimpleNamespace(_ipc_update_target=SimpleNamespace(server_url="http://owned"), api_key=None)
        t._adapter = SimpleNamespace(
            before_update=Mock(),
            after_update_all_groups=Mock(),
            after_update_per_batch=Mock(),
            serialize=Mock(return_value="payload"),
            build_request=lambda batch, data: module.WeightUpdateRequest(
                endpoint="update_weights", body={"finished": batch.finished}
            ),
        )
        t.post_json = Mock()
        monkeypatch.setattr(module, "DEVICE_MODULE", SimpleNamespace(empty_cache=lambda: None))
        monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(module.dist, "barrier", lambda **kwargs: None)
        return t

    def test_only_final_bucket_carries_common_epoch_and_next_update_changes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        t = self.make_transport(monkeypatch)
        versions = iter(["a" * 32, "b" * 32])

        def broadcast(value: list[str | None], src: int, group: object) -> None:
            assert group is t._shared_store_group
            value[0] = next(versions)

        monkeypatch.setattr(module.dist, "broadcast_object_list", broadcast)
        batches = [WeightUpdateBatch({}), WeightUpdateBatch({}, finished=True)]
        iterator = SimpleNamespace(iter_batch_groups=lambda: [batches])
        t.update(iterator)
        t.update(iterator)
        bodies = [call.args[2] for call in t.post_json.call_args_list]
        assert "weight_version" not in bodies[0] and "weight_version" not in bodies[2]
        assert bodies[1]["weight_version"] == "a" * 32
        assert bodies[3]["weight_version"] == "b" * 32
        assert t._shared_store_epoch is None

    def test_default_does_not_add_epoch_or_collective(self, monkeypatch: pytest.MonkeyPatch) -> None:
        t = self.make_transport(monkeypatch, shared=False)
        broadcast = Mock(side_effect=AssertionError("default must not add a collective"))
        monkeypatch.setattr(module.dist, "broadcast_object_list", broadcast)
        t.update(SimpleNamespace(iter_batch_groups=lambda: [[WeightUpdateBatch({}, finished=True)]]))
        assert "weight_version" not in t.post_json.call_args.args[2]
        broadcast.assert_not_called()

    def test_failed_send_clears_pending_epoch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        t = self.make_transport(monkeypatch)
        monkeypatch.setattr(module.dist, "broadcast_object_list", lambda *a, **k: None)
        t.post_json.side_effect = RuntimeError("injected update failure")
        with pytest.raises(RuntimeError, match="update failure"):
            t.update(SimpleNamespace(iter_batch_groups=lambda: [[WeightUpdateBatch({}, finished=True)]]))
        assert t._shared_store_epoch is None

    @pytest.mark.parametrize("complete", [False, True])
    def test_group_is_owned_and_partial_recovery_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch, complete: bool
    ) -> None:
        adapter = Mock()
        monkeypatch.setattr(module.IPCWeightTransport, "_build_adapter", lambda self: adapter)
        group = object()
        cpu_mesh = SimpleNamespace(mesh=torch.tensor([0]), get_group=lambda: group)
        mesh = Mock()
        mesh.__getitem__ = Mock(return_value=cpu_mesh)
        monkeypatch.setattr(module, "DeviceMesh", lambda *a, **k: mesh)
        monkeypatch.setattr(module.dist, "get_world_size", lambda: 4)
        create, destroy = Mock(return_value=group), Mock()
        monkeypatch.setattr(module.dist, "new_group", create)
        monkeypatch.setattr(module.dist, "destroy_process_group", destroy)
        info = SimpleNamespace(
            backend="pytorch",
            ep=2,
            tp=1,
            ipc_rank_mesh=[[0, 1], [2, 3]],
            rollout_config=SimpleNamespace(
                extra_rollout_config={
                    "lmdeploy_kv_transfer_config": {"kv_connector_extra_config": {"rl_share_across_engines": True}}
                }
            ),
            weight_update_targets=[SimpleNamespace(update_ranks=(i,)) for i in range(4 if complete else 3)],
        )
        if not complete:
            with pytest.raises(ValueError, match="partial recovery"):
                module.IPCWeightTransport(rank=0, logger=Mock(), config=object(), rollout_info=info)
            create.assert_not_called()
        else:
            t = module.IPCWeightTransport(rank=0, logger=Mock(), config=object(), rollout_info=info)
            create.assert_called_once_with(backend="gloo")
            t.teardown()
            t.teardown()
            destroy.assert_called_once_with(group)

    @pytest.mark.parametrize("shared", [False, True])
    def test_final_bucket_reaches_lmdeploy_with_optional_weight_version(
        self, monkeypatch: pytest.MonkeyPatch, shared: bool
    ) -> None:
        from contextlib import nullcontext

        pytest.importorskip("lmdeploy")
        from lmdeploy.messages import KVTransferConfig
        from lmdeploy.pytorch.engine.model_agent import agent as agent_module
        from lmdeploy.serve.openai.protocol import UpdateParamsRequest

        transport = self.make_transport(monkeypatch, shared=shared)
        adapter = module.LMDeployIPCBackendAdapter(rollout_tp=1, backend="pytorch", default_ipc_tensor_bytes=8)
        monkeypatch.setattr(adapter, "before_update", lambda: None)
        transport._adapter = adapter
        monkeypatch.setattr(
            module.dist, "broadcast_object_list", lambda value, **kwargs: value.__setitem__(0, "step-17")
        )
        agent = agent_module.BaseModelAgent.__new__(agent_module.BaseModelAgent)
        extra = {"rl_weight_versioning": True, "rl_share_across_engines": shared, "weight_version": "bootstrap"}
        agent.cache_config = SimpleNamespace(
            kv_transfer_config=KVTransferConfig(
                kv_connector="MooncakeStoreConnector", kv_role="kv_both", kv_connector_extra_config=extra
            )
        )
        agent.kv_connector = None
        agent.dist_config = SimpleNamespace(world_size=1)
        agent.dist_ctx = SimpleNamespace(rank=0, cpu_group=None)
        agent._rl_store_needs_shared_commit = True
        agent.patched_model = SimpleNamespace(get_model=lambda: object())
        agent.spec_agent = SimpleNamespace(get_model=lambda: None)
        agent._split_updated_weights = lambda weights: (weights, [])
        agent.all_context = nullcontext
        monkeypatch.setattr(agent_module, "process_weights_after_loading", lambda target: None)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        bodies = []

        def deliver(url: str, endpoint: str, body: dict, **kwargs: object) -> dict:
            assert endpoint == "update_weights" and "rl_store_epoch" not in body
            bodies.append(body.copy())
            agent.update_params(UpdateParamsRequest(**body))
            return {}

        transport.post_json = deliver
        batches = [WeightUpdateBatch({}), WeightUpdateBatch({}, finished=True)]
        transport.update(SimpleNamespace(iter_batch_groups=lambda: [batches]))
        assert "weight_version" not in bodies[0]
        assert bodies[1].get("weight_version") == ("step-17" if shared else None)
        if shared:
            assert extra["weight_version"] == "step-17"
        else:
            assert extra["weight_version"] != "bootstrap"
        assert agent._rl_store_needs_shared_commit is False
