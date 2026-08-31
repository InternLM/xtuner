import json
import runpy

import pytest

from xtuner._testing.moonep_acceptance import AcceptanceRun, compare_runs


def _write_tracker(path, *, tgs_scale: float, mtp: bool) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w", encoding="utf-8") as output:
        for step in range(1, 21):
            record = {
                "step": step,
                "runtime_info/text_tokens": 65536,
                "runtime_info/tgs": (1000 + step) * tgs_scale,
                "loss/reduced_llm_loss": 2.0 - step / 100,
                "loss/reduced_balancing_loss": 0.01 + step / 10000,
                "loss/local_loss": 2.01 - step / 100 + step / 10000,
                "grad_norm": 0.5 + step / 1000,
            }
            if mtp:
                record["loss/reduced_mtp_loss"] = 0.2 - step / 1000
                record["loss/local_loss"] += record["loss/reduced_mtp_loss"]
            output.write(json.dumps(record) + "\n")


@pytest.mark.parametrize("backend", ["deepep", "moonep"])
@pytest.mark.parametrize("mtp", [False, True])
def test_qwen35_acceptance_config_locks_the_formal_workload(monkeypatch, tmp_path, backend, mtp) -> None:
    monkeypatch.setenv("MOONEP_ACCEPTANCE_BACKEND", backend)
    monkeypatch.setenv("MOONEP_ACCEPTANCE_MTP", str(int(mtp)))
    monkeypatch.setenv("MOONEP_ACCEPTANCE_PACK_LENGTH", "65536")
    monkeypatch.setenv("MOONEP_ACCEPTANCE_WORK_DIR", str(tmp_path / "run"))
    monkeypatch.setenv("MOONEP_ACCEPTANCE_MODEL_PATH", "/model")
    monkeypatch.setenv("MOONEP_ACCEPTANCE_DATA_PATH", "/data")

    trainer = runpy.run_path("tests/acceptance/sft_qwen35_moonep_acceptance.py")["trainer"]
    model = trainer.model_cfg

    assert trainer.total_step == 20
    assert trainer.global_batch_size == 8
    assert trainer.intra_layer_micro_batch == 1
    assert trainer.sp_size == 1
    assert trainer.debug_skip_save is True
    assert trainer.dataloader_cfg.pack_to_max_length is True
    assert trainer.dataloader_cfg.pack_max_length == 65536
    assert trainer.fsdp_cfg.ep_size == 4
    assert trainer.fsdp_cfg.param_dtype.__str__() == "torch.bfloat16"
    assert trainer.fsdp_cfg.reduce_dtype.__str__() == "torch.bfloat16"
    assert trainer.fsdp_cfg.torch_compile is True
    assert trainer.fsdp_cfg.cpu_offload is False
    assert model.only_llm_forward is True
    assert model.text_config.ep_size == 4
    assert model.text_config.dispatcher == backend
    assert model.text_config.moonep_staging_reference is False
    assert model.text_config.moonep_num_sms == 64
    assert model.text_config.router_async_offload is False
    assert model.text_config.router_compute_dtype == "float32"
    assert (
        model.text_config.compile_cfg["xtuner.v1.module.attention.mha.MultiHeadAttention.forward"]["fullgraph"]
        is False
    )
    assert (model.text_config.mtp_config is not None) is mtp
    if mtp:
        assert model.text_config.mtp_config.num_layers == 1
        assert model.text_config.mtp_config.share_weights is False


def test_acceptance_report_compares_all_steps_and_warm_throughput(tmp_path) -> None:
    deepep_tracker = tmp_path / "deepep" / "tracker.jsonl"
    moonep_tracker = tmp_path / "moonep" / "tracker.jsonl"
    _write_tracker(deepep_tracker, tgs_scale=1.0, mtp=True)
    _write_tracker(moonep_tracker, tgs_scale=0.96, mtp=True)

    deepep = AcceptanceRun.from_tracker(deepep_tracker, backend="deepep", mtp=True, pack_length=65536)
    moonep = AcceptanceRun.from_tracker(moonep_tracker, backend="moonep", mtp=True, pack_length=65536)
    result = compare_runs(deepep, moonep)

    assert result.passed
    assert result.throughput_ratio == pytest.approx(0.96)
    assert result.throughput_steps == list(range(6, 21))
    assert set(result.curves) == {
        "reduced_llm_loss",
        "reduced_mtp_loss",
        "reduced_balancing_loss",
        "total_loss",
        "grad_norm",
    }
    assert all(curve.cosine_similarity >= 0.99 for curve in result.curves.values())
    assert all(curve.mean_relative_difference < 0.01 for curve in result.curves.values())


def test_acceptance_report_rejects_incomplete_or_mismatched_runs(tmp_path) -> None:
    deepep_tracker = tmp_path / "deepep" / "tracker.jsonl"
    moonep_tracker = tmp_path / "moonep" / "tracker.jsonl"
    _write_tracker(deepep_tracker, tgs_scale=1.0, mtp=False)
    _write_tracker(moonep_tracker, tgs_scale=0.94, mtp=False)

    deepep = AcceptanceRun.from_tracker(deepep_tracker, backend="deepep", mtp=False, pack_length=65536)
    slow = AcceptanceRun.from_tracker(moonep_tracker, backend="moonep", mtp=False, pack_length=65536)
    assert not compare_runs(deepep, slow).passed

    mismatched = AcceptanceRun.from_tracker(moonep_tracker, backend="moonep", mtp=False, pack_length=32768)
    with pytest.raises(ValueError, match="pack_length"):
        compare_runs(deepep, mismatched)

    lines = moonep_tracker.read_text().splitlines()
    moonep_tracker.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="exactly steps 1..20"):
        AcceptanceRun.from_tracker(moonep_tracker, backend="moonep", mtp=False, pack_length=65536)
