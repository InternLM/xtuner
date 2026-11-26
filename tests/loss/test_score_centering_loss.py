"""Score-centering loss and its sampler top-k data contract (CPU only).

Covered behavior:
- TestScoreCenteringGradient: the logits gradient equals a dense, full-vocabulary oracle of the
  centered estimator (head + rescaled trainer tail) for weight none/tis/mis, including a sampled
  token outside the head, padding slots and rows, a head covering the whole vocabulary and a tail
  mass below ``tail_eps``; zero expected gradient for a constant advantage; no correction on-policy.
- TestScoreCenteringModes: chunk and eager modes agree (loss, gradients, metrics, KL penalty);
  sequence parallel (two gloo processes) matches SP=1, including SP padding of the top-k fields.
- TestSamplerTopKContract: SamplerTopK layout checks, RolloutState publication and reset, and
  TrainBatch loading, shifting, padding and value validation.
- TestScoreCenteringConfig: configuration validation and required pack fields.
"""

from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from xtuner.v1.data_proto.rl_data import RolloutState, SamplerTopK, Status
from xtuner.v1.loss.utils import sp_split
from xtuner.v1.rl.distillation import DistillationTrainerAdapter
from xtuner.v1.rl.loss import GRPOLossConfig, ScoreCenteringLossConfig, kl_penalty
from xtuner.v1.rl.rollout_is import RolloutImportanceSampling
from xtuner.v1.rl.trainer.pack import RankPlan
from xtuner.v1.rl.trainer.train_batch import TrainBatch
from xtuner.v1.utils.test_utils import init_data_mesh


IGNORE_IDX = -100
VOCAB_SIZE = 11
HIDDEN_SIZE = 5
MIS_BOUNDS = (2.0, 0.5)
# Fixed per-case seeds; the mis/mixed seed is chosen so that some head ratios fall outside MIS_BOUNDS.
SEEDS = {"mixed": 0, "full_head": 1, "tiny_tail": 2}


def _weight_fn(weight: str, bounds: tuple[float, float]):
    upper, lower = bounds
    if weight == "none":
        return lambda ratio: torch.ones_like(ratio)
    if weight == "tis":
        return lambda ratio: ratio.clamp(max=upper)
    return lambda ratio: torch.where((ratio >= lower) & (ratio <= upper), ratio, torch.zeros_like(ratio))


def _dense_oracle_grad(data: dict, logits: torch.Tensor, weight: str, bounds: tuple[float, float], eps: float):
    """Gradient of the loss w.r.t. logits, written over the full vocabulary.

    The sampler distribution is the recorded head plus the trainer's tail rescaled by rho; the
    gradient of one token is -A / N * (w_y s_y - sum_v qhat_v w_v s_v) with s_v = e_v - p.
    """
    f = _weight_fn(weight, bounds)
    logits = logits.detach().double()[0]
    probs = logits.softmax(-1)
    labels = data["shifted_labels"][0]
    valid = labels != IGNORE_IDX
    num_tokens = max(int(valid.sum()), 1)
    grad = torch.zeros_like(logits)
    for t in range(logits.shape[0]):
        if not valid[t]:
            continue
        p = probs[t]
        ids = data["sampler_topk_ids"][0, t]
        head = ids[ids >= 0]
        q_head = data["sampler_topk_logprobs"][0, t][ids >= 0].double().exp()
        rho = max(1.0 - float(q_head.sum()), eps) / max(1.0 - float(p[head].sum()), eps)
        q_hat = rho * p.clone()
        q_hat[head] = q_head
        weighted = q_hat * f(p / q_hat)
        y = int(labels[t])
        w_y = f((p[y].log() - data["rollout_logprobs"][0, t].double()).exp())
        one_hot = torch.zeros_like(p)
        one_hot[y] = 1.0
        centered = w_y * (one_hot - p) - (weighted - weighted.sum() * p)
        grad[t] = -data["advantages"][0, t].double() * centered / num_tokens
    return grad.unsqueeze(0)


def _sampler_head(q_logp: torch.Tensor, ids: list[int], k: int) -> tuple[torch.Tensor, torch.Tensor]:
    row_ids = torch.full((k,), -1, dtype=torch.long)
    row_logp = torch.full((k,), float("-inf"))
    row_ids[: len(ids)] = torch.tensor(ids, dtype=torch.long)
    row_logp[: len(ids)] = q_logp[ids]
    return row_ids, row_logp


def _make_case(kind: str, seed: int = 0) -> tuple[dict, torch.Tensor]:
    """Return loss data and logits ``[1, T, V]`` of one test layout."""
    gen = torch.Generator().manual_seed(seed)
    if kind == "mixed":
        num_rows, k = 7, 4
    elif kind == "full_head":
        num_rows, k = 3, VOCAB_SIZE
    else:  # tiny_tail
        num_rows, k = 3, VOCAB_SIZE - 1
    logits = torch.randn(1, num_rows, VOCAB_SIZE, generator=gen)
    q_logits = logits + torch.randn(1, num_rows, VOCAB_SIZE, generator=gen)
    if kind == "tiny_tail":
        # The token outside the head has negligible mass under both distributions.
        logits[..., -1] = -40.0
        q_logits[..., -1] = -40.0
    q_logp = q_logits.log_softmax(-1)

    labels = torch.empty(1, num_rows, dtype=torch.long)
    ids = torch.full((1, num_rows, k), -1, dtype=torch.long)
    sampler_logp = torch.full((1, num_rows, k), float("-inf"))
    for t in range(num_rows):
        order = q_logp[0, t].argsort(descending=True).tolist()
        y = int(torch.multinomial(q_logp[0, t].exp(), 1, generator=gen))
        head = order[:k]
        if kind == "mixed":
            if t == 2:  # sampled token outside the head
                y = order[-1]
            elif t == 3:  # two padding slots
                head = order[:2]
            elif t == 4:  # unsupervised row without a recorded distribution (e.g. tool output)
                head = []
            elif t == 0:  # prompt row
                head = []
        labels[0, t] = y
        ids[0, t], sampler_logp[0, t] = _sampler_head(q_logp[0, t], head, k)
    if kind == "mixed":
        labels[0, 0] = IGNORE_IDX
        labels[0, 4] = IGNORE_IDX
        labels[0, 5] = IGNORE_IDX
    rollout_logprobs = q_logp[0].gather(-1, labels[0].clamp_min(0).unsqueeze(-1)).squeeze(-1).unsqueeze(0)
    rollout_logprobs = rollout_logprobs.masked_fill(labels == IGNORE_IDX, 0.0)
    data = {
        "shifted_labels": labels,
        "advantages": torch.randn(1, num_rows, generator=gen).masked_fill(labels == IGNORE_IDX, 0.0),
        "rollout_logprobs": rollout_logprobs,
        "sampler_topk_ids": ids,
        "sampler_topk_logprobs": sampler_logp,
    }
    return data, logits


def _config(weight: str, **kwargs) -> ScoreCenteringLossConfig:
    bounds = MIS_BOUNDS if weight == "mis" else (2.0, 0.0)
    return ScoreCenteringLossConfig(weight=weight, weight_bounds=bounds, **kwargs)


def _logits_grad(cfg: ScoreCenteringLossConfig, data: dict, logits: torch.Tensor):
    """Run the loss with an identity head so the hidden-state gradient is the logits gradient."""
    ctx = cfg.build(data, device="cpu")
    assert ctx is not None
    ctx = ctx.build_batches([ctx])[0]
    hidden = logits.clone().requires_grad_(True)
    loss, (_, extra_info) = ctx.forward(hidden, torch.eye(VOCAB_SIZE))
    loss.backward()
    assert hidden.grad is not None
    return loss.detach(), hidden.grad, extra_info


class TestScoreCenteringGradient:
    @pytest.mark.parametrize("weight", ["none", "tis", "mis"])
    @pytest.mark.parametrize("kind", ["mixed", "full_head", "tiny_tail"])
    def test_gradient_matches_dense_oracle(self, weight, kind):
        cfg = _config(weight)
        data, logits = _make_case(kind, seed=SEEDS[kind])
        loss, grad, _ = _logits_grad(cfg, data, logits)

        expected = _dense_oracle_grad(data, logits, weight, cfg.weight_bounds, cfg.tail_eps)
        assert torch.isfinite(loss)
        torch.testing.assert_close(grad.double(), expected, rtol=1e-4, atol=1e-6)
        # Ignored rows get no gradient even though their head slots hold real ids.
        ignored = data["shifted_labels"][0] == IGNORE_IDX
        assert torch.equal(grad[0, ignored], torch.zeros_like(grad[0, ignored]))

    def test_mis_case_zeroes_some_weights(self):
        # Guard that the mis oracle case actually exercises the zeroed band.
        data, logits = _make_case("mixed", seed=SEEDS["mixed"])
        ratio = (
            logits.log_softmax(-1).gather(-1, data["sampler_topk_ids"].clamp_min(0)) - data["sampler_topk_logprobs"]
        ).exp()
        head = data["sampler_topk_ids"] >= 0
        upper, lower = MIS_BOUNDS
        assert ((ratio[head] < lower) | (ratio[head] > upper)).any()

    @pytest.mark.parametrize("weight", ["none", "tis", "mis"])
    def test_constant_advantage_has_zero_expected_gradient_with_full_head(self, weight):
        cfg = _config(weight)
        gen = torch.Generator().manual_seed(7)
        logits = torch.randn(1, 1, VOCAB_SIZE, generator=gen)
        q_logp = (logits + torch.randn(1, 1, VOCAB_SIZE, generator=gen)).log_softmax(-1)
        ids = torch.arange(VOCAB_SIZE).view(1, 1, -1)
        expected_grad = torch.zeros(1, 1, VOCAB_SIZE, dtype=torch.float64)
        for y in range(VOCAB_SIZE):
            data = {
                "shifted_labels": torch.tensor([[y]]),
                "advantages": torch.ones(1, 1),
                "rollout_logprobs": q_logp[..., y],
                "sampler_topk_ids": ids,
                "sampler_topk_logprobs": q_logp,
            }
            _, grad, _ = _logits_grad(cfg, data, logits)
            expected_grad += q_logp[0, 0, y].double().exp() * grad.double()
        torch.testing.assert_close(expected_grad, torch.zeros_like(expected_grad), rtol=0, atol=1e-6)

    @pytest.mark.parametrize("weight", ["none", "tis"])
    def test_on_policy_correction_has_no_gradient(self, weight):
        cfg = _config(weight)
        gen = torch.Generator().manual_seed(3)
        logits = torch.randn(1, 4, VOCAB_SIZE, generator=gen)
        logp = logits.log_softmax(-1)
        ids = logp.topk(4, dim=-1).indices
        labels = torch.tensor([[1, 5, 9, 2]])
        advantages = torch.randn(1, 4, generator=gen)
        data = {
            "shifted_labels": labels,
            "advantages": advantages,
            "rollout_logprobs": logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1),
            "sampler_topk_ids": ids,
            "sampler_topk_logprobs": logp.gather(-1, ids),
        }
        _, grad, _ = _logits_grad(cfg, data, logits)
        # Plain REINFORCE with unit weights: -A / N * (e_y - p).
        one_hot = torch.nn.functional.one_hot(labels, VOCAB_SIZE).float()
        expected = -advantages.unsqueeze(-1) * (one_hot - logp.exp()) / labels.numel()
        torch.testing.assert_close(grad, expected, rtol=1e-5, atol=1e-6)


def _parameter_grads(cfg: ScoreCenteringLossConfig, data: dict, hidden: torch.Tensor, head: torch.Tensor):
    ctx = cfg.build(data, device="cpu")
    assert ctx is not None
    ctx = ctx.build_batches([ctx])[0]
    hidden = hidden.clone().requires_grad_(True)
    head = head.clone().requires_grad_(True)
    loss, (_, extra_info) = ctx.forward(hidden, head)
    loss.backward()
    return loss.detach(), hidden.grad, head.grad, extra_info.get()


def _hidden_case(seed: int, num_rows: int = 9) -> tuple[dict, torch.Tensor, torch.Tensor]:
    data, _ = _make_case("mixed", seed=seed)
    data = {key: torch.cat([value, value[:, :2]], dim=1)[:, :num_rows] for key, value in data.items()}
    gen = torch.Generator().manual_seed(seed + 1)
    hidden = torch.randn(1, num_rows, HIDDEN_SIZE, generator=gen)
    head = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, generator=gen)
    data["ref_logprobs"] = -torch.rand(1, num_rows, generator=gen)
    return data, hidden, head


def _sp_worker(rank: int, rendezvous: str, expected_path: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=120)
    )
    try:
        sp_mesh = init_data_mesh("cpu", 2)["sp"]
        data, hidden, head = _hidden_case(seed=11)
        cfg = _config("tis", mode="chunk", chunk_size=2, use_kl_loss=True, kl_loss_type="low_var_kl")
        ctx = cfg.build(data, sp_mesh=sp_mesh, device="cpu")
        assert ctx is not None
        # Nine rows split into two shards of five: rank 1 holds one SP padding row.
        kwargs = ctx.loss_kwargs
        assert kwargs.sampler_topk_ids is not None and kwargs.sampler_topk_logprobs is not None
        assert kwargs.sampler_topk_ids.shape == (1, 5, 4)
        if rank == 1:
            assert (kwargs.sampler_topk_ids[0, -1] == -1).all()
            assert torch.isneginf(kwargs.sampler_topk_logprobs[0, -1]).all()
        # Reference logprobs come from an SP-local forward in training; bind the local shard.
        kwargs.ref_logprobs = sp_split(data["ref_logprobs"], sp_mesh=sp_mesh, split_dim=1, padding_value=0.0)
        ctx = ctx.build_batches([ctx])[0]
        local_hidden = sp_split(hidden, sp_mesh=sp_mesh, split_dim=1, padding_value=0.0).requires_grad_(True)
        local_head = head.clone().requires_grad_(True)
        loss, _ = ctx.forward(local_hidden, local_head)
        loss.backward()
        assert local_head.grad is not None and local_hidden.grad is not None
        head_grad = local_head.grad.clone()
        # The autograd all-reduce scales each rank's gradient by the world size.
        dist.all_reduce(head_grad, op=dist.ReduceOp.AVG)
        hidden_grads = [torch.empty_like(local_hidden.grad) for _ in range(2)]
        dist.all_gather(hidden_grads, local_hidden.grad.contiguous())
        hidden_grad = torch.cat(hidden_grads, dim=1)[:, : hidden.shape[1]] / 2

        expected = torch.load(expected_path)
        torch.testing.assert_close(loss.detach(), expected["loss"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(head_grad, expected["head_grad"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(hidden_grad, expected["hidden_grad"], rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


class TestScoreCenteringModes:
    @pytest.mark.parametrize("weight", ["none", "tis", "mis"])
    def test_chunk_matches_eager(self, weight):
        data, hidden, head = _hidden_case(seed=5)
        kwargs = {"use_kl_loss": True, "kl_loss_type": "low_var_kl", "kl_loss_coef": 0.1}
        eager = _parameter_grads(_config(weight, mode="eager", **kwargs), data, hidden, head)
        chunk = _parameter_grads(_config(weight, mode="chunk", chunk_size=2, **kwargs), data, hidden, head)
        for eager_value, chunk_value in zip(eager[:3], chunk[:3]):
            torch.testing.assert_close(chunk_value, eager_value, rtol=1e-5, atol=1e-6)
        eager_metrics, chunk_metrics = eager[3], chunk[3]
        sc_keys = [key for key in eager_metrics if key.startswith("reduced_sc_")]
        assert "reduced_sc_tail_ratio_max" in sc_keys and "reduced_sc_valid_count" in sc_keys
        for key in sc_keys:
            assert chunk_metrics[key] == pytest.approx(eager_metrics[key], rel=1e-5, abs=1e-6)

    def test_kl_penalty_is_added(self):
        data, hidden, head = _hidden_case(seed=5)
        with_kl = _parameter_grads(
            _config("tis", use_kl_loss=True, kl_loss_type="low_var_kl", kl_loss_coef=0.1), data, hidden, head
        )
        without_kl = _parameter_grads(_config("tis"), data, hidden, head)
        logp = (hidden @ head.T).log_softmax(-1).gather(-1, data["shifted_labels"].clamp_min(0).unsqueeze(-1))
        valid = (data["shifted_labels"] != IGNORE_IDX).float()
        expected_kl = kl_penalty(logp.squeeze(-1), data["ref_logprobs"], valid * 0.1 / valid.sum(), "low_var_kl")
        torch.testing.assert_close(with_kl[0] - without_kl[0], expected_kl, rtol=1e-5, atol=1e-6)

    def test_sequence_parallel_matches_single_rank(self, tmp_path):
        data, hidden, head = _hidden_case(seed=11)
        cfg = _config("tis", mode="eager", use_kl_loss=True, kl_loss_type="low_var_kl")
        loss, hidden_grad, head_grad, _ = _parameter_grads(cfg, data, hidden, head)
        expected_path = tmp_path / "expected.pt"
        torch.save({"loss": loss, "hidden_grad": hidden_grad, "head_grad": head_grad}, expected_path)
        mp.spawn(_sp_worker, args=(str(tmp_path / "rendezvous"), str(expected_path)), nprocs=2, join=True)

    def test_finalize_metrics_reports_token_means(self):
        cfg = _config("mis")
        data, logits = _make_case("mixed", seed=SEEDS["mixed"])
        _, _, extra_info = _logits_grad(cfg, data, logits)
        metrics = cfg.finalize_metrics(extra_info.get(), "cpu")
        valid = data["shifted_labels"] != IGNORE_IDX
        assert not any(key.startswith("reduced_sc_") for key in metrics)
        assert metrics["sc_sampler_head_mass"] == pytest.approx(
            float(data["sampler_topk_logprobs"].exp().sum(-1)[valid].mean()), rel=1e-5
        )
        assert 0.0 <= metrics["sc_weight_zero_frac"] <= 1.0
        assert metrics["sc_tail_ratio_max"] >= metrics["sc_tail_ratio_mean"] > 0.0
        assert metrics["sc_rollout_kl3"] >= 0.0


def _topk(rows: list[list[tuple[int, float]]], k: int) -> SamplerTopK:
    ids = np.full((len(rows), k), -1, dtype=np.int32)
    logprobs = np.full((len(rows), k), -np.inf, dtype=np.float32)
    for index, row in enumerate(rows):
        for slot, (token_id, logprob) in enumerate(row):
            ids[index, slot] = token_id
            logprobs[index, slot] = logprob
    return SamplerTopK(token_ids=ids, logprobs=logprobs)


def _published_state(sampler_topk: SamplerTopK | None, uid: int = 1) -> RolloutState:
    state = RolloutState(
        rollout_id=uid,
        group_id=1,
        message=[],
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5, 6],
        status=Status.COMPLETED,
        reward={"score": 1.0},
    )
    state.response_model_steps = [1, 1, 1]
    state.set_training_fields(
        input_ids=[1, 2, 3, 4, 5, 6],
        labels=[-100, -100, -100, 4, 5, 6],
        logprobs=[0.0, 0.0, 0.0, -0.1, -0.2, -0.3],
        sampler_topk=sampler_topk,
    )
    return state


def _response_topk(k: int = 2) -> SamplerTopK:
    # Prompt rows hold no distribution; response rows 3..5 produced tokens 4, 5, 6.
    return _topk([[], [], [], [(4, -0.1), (7, -2.5)], [(8, -0.5), (5, -0.2)][:k], [(6, -0.3)]], k)


def _train_batch(states: list[RolloutState], loss_cfg, pack_max_length: int | None = None) -> TrainBatch:
    if pack_max_length is None:
        pack_max_length = sum(len(state.input_ids or []) - 1 for state in states)
    return TrainBatch(
        states,
        RankPlan(global_indices=tuple(range(len(states))), steps=((tuple(range(len(states))),),)),
        {  # type: ignore[arg-type]
            "rollout_idx": 0,
            "use_3d_position_ids": False,
            "pack_loss_keys": ["rollout_logprobs"],
            "has_routed_experts": False,
        },
        DistillationTrainerAdapter(None),
        pack_max_length=pack_max_length,
        model_cfg=SimpleNamespace(  # type: ignore[arg-type]
            n_routed_experts=0, num_hidden_layers=1, num_experts_per_tok=1, vocab_size=VOCAB_SIZE
        ),
        teacher_loss_keys=[],
        distillation_config=None,
        required_loss_keys=loss_cfg.required_loss_keys,
    )


class TestSamplerTopKContract:
    def test_sampler_topk_rejects_bad_layout(self):
        good_ids = np.zeros((3, 2), dtype=np.int32)
        good_logprobs = np.zeros((3, 2), dtype=np.float32)
        for ids, logprobs in (
            (good_ids.astype(np.int64), good_logprobs),
            (good_ids, good_logprobs.astype(np.float64)),
            (good_ids, good_logprobs[:2]),
            (good_ids[:, :0], good_logprobs[:, :0]),
            (good_ids.reshape(-1), good_logprobs.reshape(-1)),
        ):
            with pytest.raises(ValueError, match="Sampler top-k arrays"):
                SamplerTopK(token_ids=ids, logprobs=logprobs)

    def test_publish_validate_and_reset(self):
        state = _published_state(_response_topk())
        assert state.train_meta is not None and state.train_meta.has_sampler_topk
        state.validate_training()

        state.reset_response()
        assert state.sampler_topk is None

        plain = _published_state(None)
        assert plain.train_meta is not None and not plain.train_meta.has_sampler_topk

    def test_row_mismatch_leaves_no_partial_update(self):
        state = _published_state(None)
        with pytest.raises(ValueError, match="sampler_topk rows mismatch"):
            state.set_training_fields(
                input_ids=[1, 2, 3, 4, 5, 6],
                labels=[-100, -100, -100, 4, 5, 6],
                logprobs=None,
                sampler_topk=_topk([[]] * 5, 2),
            )
        assert state.sampler_topk is None
        assert state.logprobs == [0.0, 0.0, 0.0, -0.1, -0.2, -0.3]

    def test_validate_training_detects_stale_flag(self):
        state = _published_state(_response_topk())
        state.sampler_topk = None
        with pytest.raises(ValueError, match="inconsistent"):
            state.validate_training()

    def test_train_batch_shifts_and_pads_sampler_topk(self):
        states = [_published_state(_response_topk(), uid=1), _published_state(_response_topk(), uid=2)]
        batch = _train_batch(states, ScoreCenteringLossConfig(), pack_max_length=13)
        batch.set_advantages([1.0, -1.0])
        pack = batch.materialize((0, 1))
        ids = pack.full_data["sampler_topk_ids"]
        logprobs = pack.full_data["sampler_topk_logprobs"]
        assert ids.dtype == torch.long and logprobs.dtype == torch.float32
        assert ids.shape == (1, 13, 2) and logprobs.shape == (1, 13, 2)
        # Shifted like rollout_logprobs: loss position t uses input row t + 1.
        expected_ids = [[-1, -1], [-1, -1], [4, 7], [8, 5], [6, -1]]
        assert ids[0, :5].tolist() == expected_ids
        assert ids[0, 5:10].tolist() == expected_ids
        assert (ids[0, 10:] == -1).all() and torch.isneginf(logprobs[0, 10:]).all()
        torch.testing.assert_close(logprobs[0, 2], torch.tensor([-0.1, -2.5]))
        assert torch.isneginf(logprobs[0, 4, 1])
        assert pack.full_data["rollout_logprobs"].shape == (1, 13)

        padding_pack = batch.materialize(())
        assert padding_pack.full_data["sampler_topk_ids"].shape == (1, 13, 2)
        assert (padding_pack.full_data["sampler_topk_ids"] == -1).all()

        loss_ctx = ScoreCenteringLossConfig().build(pack.full_data, device="cpu")
        assert loss_ctx is not None and loss_ctx.loss_kwargs.sampler_topk_ids is not None

    def test_train_batch_skips_sampler_topk_for_other_losses(self):
        batch = _train_batch([_published_state(_response_topk())], GRPOLossConfig(policy_loss_cfg={}))
        batch.set_advantages([1.0])
        assert "sampler_topk_ids" not in batch.materialize((0,)).full_data

    def test_train_batch_requires_sampler_topk(self):
        with pytest.raises(ValueError, match="requires sampler_topk"):
            _train_batch([_published_state(None)], ScoreCenteringLossConfig())

    @pytest.mark.parametrize(
        "row, message",
        [
            ([(VOCAB_SIZE, -0.1)], "must be -1 or in"),
            ([(4, -0.1), (4, -0.2)], "distinct"),
            ([(4, 0.5)], "finite and <= 0"),
            ([(4, -np.inf)], "finite and <= 0"),
            ([(4, np.nan)], "finite and <= 0"),
        ],
    )
    def test_train_batch_rejects_invalid_values(self, row, message):
        topk = _topk([[], [], [], row, [(5, -0.2)], [(6, -0.3)]], 2)
        with pytest.raises(ValueError, match=message):
            _train_batch([_published_state(topk)], ScoreCenteringLossConfig())

    def test_train_batch_rejects_supervised_token_without_head(self):
        topk = _response_topk()
        topk.token_ids[4] = -1
        topk.logprobs[4] = -np.inf
        with pytest.raises(ValueError, match=r"head for every supervised token.*\[3\]"):
            _train_batch([_published_state(topk)], ScoreCenteringLossConfig())

    def test_train_batch_accepts_unsupervised_row_without_head(self):
        state = _published_state(_response_topk())
        state.set_training_fields(
            input_ids=state.input_ids,
            labels=[-100, -100, -100, 4, -100, 6],
            logprobs=state.logprobs,
            sampler_topk=_topk([[], [], [], [(4, -0.1)], [], [(6, -0.3)]], 2),
        )
        batch = _train_batch([state], ScoreCenteringLossConfig())
        batch.set_advantages([1.0])
        assert batch.materialize((0,)).full_data["sampler_topk_ids"][0, 3].tolist() == [-1, -1]

    def test_train_batch_rejects_logprob_on_empty_slot(self):
        topk = _response_topk()
        topk.logprobs[4, 0] = -1.0
        topk.token_ids[4, 0] = -1
        with pytest.raises(ValueError, match="-inf for id -1"):
            _train_batch([_published_state(topk)], ScoreCenteringLossConfig())

    def test_train_batch_ignores_unused_first_row(self):
        # Row 0 produced no trained position, so its contents are not validated.
        topk = _response_topk()
        topk.token_ids[0, 0] = VOCAB_SIZE + 5
        topk.logprobs[0, 0] = 1.0
        _train_batch([_published_state(topk)], ScoreCenteringLossConfig())

    def test_train_batch_rejects_mixed_widths(self):
        wide = _topk([[], [], [], [(4, -0.1)], [(5, -0.2)], [(6, -0.3)]], 3)
        states = [_published_state(_response_topk(2), uid=1), _published_state(wide, uid=2)]
        with pytest.raises(ValueError, match="width must be equal"):
            _train_batch(states, ScoreCenteringLossConfig())


class TestScoreCenteringConfig:
    def test_required_loss_keys(self):
        assert ScoreCenteringLossConfig().required_loss_keys == [
            "rollout_logprobs",
            "sampler_topk_ids",
            "sampler_topk_logprobs",
        ]
        assert GRPOLossConfig(policy_loss_cfg={"loss_type": "vanilla"}).required_loss_keys == []

    @pytest.mark.parametrize(
        "rollout_is",
        [
            RolloutImportanceSampling(rollout_is_threshold=(2.0, 0.0)),
            RolloutImportanceSampling(rollout_is_mask_threshold=(2.0, 0.5)),
            RolloutImportanceSampling(rollout_is_veto_threshold=(10.0, 0.1)),
        ],
    )
    def test_rejects_enabled_rollout_is(self, rollout_is):
        with pytest.raises(ValueError, match="rollout_is"):
            ScoreCenteringLossConfig(rollout_is=rollout_is)

    @pytest.mark.parametrize(
        "policy_loss_cfg",
        [{"cliprange_low": 0.2, "cliprange_high": 0.2}, {"loss_type": "vanilla"}, {"clip_ratio_c": 3.0}],
    )
    def test_rejects_policy_loss_cfg(self, policy_loss_cfg):
        with pytest.raises(ValueError, match="policy_loss_cfg"):
            ScoreCenteringLossConfig(policy_loss_cfg=policy_loss_cfg)

    @pytest.mark.parametrize("bounds", [(1.0, 1.0), (0.5, 1.0), (2.0, -0.1), (float("inf"), 0.0)])
    def test_rejects_bad_weight_bounds(self, bounds):
        with pytest.raises(ValueError, match="weight_bounds"):
            ScoreCenteringLossConfig(weight="mis", weight_bounds=bounds)

    def test_rejects_liger_mode_and_bad_eps(self):
        with pytest.raises(ValueError, match="mode"):
            ScoreCenteringLossConfig(mode="liger")
        with pytest.raises(ValueError, match="tail_eps"):
            ScoreCenteringLossConfig(tail_eps=0.0)

    def test_rejects_unknown_weight(self):
        with pytest.raises(ValueError, match="weight"):
            ScoreCenteringLossConfig(weight="sequence")  # type: ignore[arg-type]
