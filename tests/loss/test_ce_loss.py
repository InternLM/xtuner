from unittest import TestCase
import torch
import torch.nn as nn
from xtuner.v1.loss.ce_loss import CELossConfig, CELossContextInputItem
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils.test_utils import assert_verbose_allclose
from xtuner.v1.loss.utils import cal_global_grad_tokens, cal_global_sum_loss_weight, len2weight
from torch.testing._internal.common_distributed import DistributedTestBase
import os
import torch.distributed as dist
from xtuner.v1.data_proto.utils import pad_to_multiple_of, split_for_sequence_parallel
from xtuner.v1.utils.test_utils import init_data_mesh
import parametrize
from functools import wraps


class TestCELoss(TestCase):
    def setUp(self) -> None:
        self.device = 'cuda'  # liger loss must be tested on GPU
        self.dtype = torch.bfloat16
        self.input_dim = 2
        self.vocab_size = 4
        self.lm_head1 = nn.Linear(self.input_dim, self.vocab_size, bias=False).to(device=self.device, dtype=self.dtype)
        self.lm_head2 = nn.Linear(self.input_dim, self.vocab_size, bias=False).to(device=self.device, dtype=self.dtype)
        self.lm_head2.weight.data = self.lm_head1.weight.data.clone()

    @parametrize.parametrize(
        "loss_mode, grad_accumulation_steps, chunk_size, atol, rtol",
        [
            ("eager", 1, -1, 1e-4, 5e-2),
            ("chunk", 1, 1024, 1e-4, 5e-2),
            ("chunk", 1, 4096, 1e-4, 5e-2),
            ("chunk", 1, 14096, 1e-4, 5e-2),
            ("eager", 4, -1, 1e-4, 5e-2),
            ("chunk", 4, 1024, 1e-4, 5e-2),
        ],
    )
    def test_global_loss_reduction(self, loss_mode, grad_accumulation_steps, chunk_size, atol, rtol):
        B, S, D = 2, 4097, self.input_dim

        targets = []
        data_batch = []
        for _ in range(grad_accumulation_steps):
            target = torch.randint(0, D, (B * S,), device=self.device, dtype=torch.long)
            # Assign some random number of elements as ignore_index
            num_elements_to_assign = torch.randint(
                1, B * S // 2, (1,)
            ).item()  # Random number of elements to set to ignore_index
            indices_to_assign = torch.randperm(B * S)[:num_elements_to_assign]  # Randomly select indices
            target[indices_to_assign] = -100
            target = target.reshape(B, S)
            targets.append(target)

            # Note: input_ids/position_ids/cu_seq_lens_q/max_length_q 等都是假数据
            cu_seq_lens_q = torch.tensor([0, S, 2 * S], device=self.device, dtype=torch.int32)
            seq_ctx = SequenceContext(input_ids=torch.ones((1, 2), device=self.device, dtype=torch.long),
                                      position_ids=torch.ones((1, 2), device=self.device, dtype=torch.long),
                                      cu_seq_lens_q=cu_seq_lens_q,
                                      cu_seq_lens_k=cu_seq_lens_q,
                                      max_length_q=1,
                                      max_length_k=1,
                                      device=self.device,
                                      )
            seq_ctx.to(self.device)
            data_batch.append({'seq_ctx': seq_ctx, 'shifted_labels': target})

        global_grad_tokens = cal_global_grad_tokens(targets)

        loss_cfg = CELossConfig(mode=loss_mode, chunk_size=chunk_size, loss_reduction="token")
        LossContext = loss_cfg.loss_ctx_cls
        seq_ctx_list: list[SequenceContext] = []
        loss_ctx_input_list: list[CELossContextInputItem] = []
        for data in data_batch:
            seq_ctx = data["seq_ctx"]
            loss_ctx_input = CELossContextInputItem(shifted_labels=data["shifted_labels"])
            seq_ctx_list.append(seq_ctx)
            loss_ctx_input_list.append(loss_ctx_input)
        batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
            loss_ctx_input_list, loss_cfg, cu_seq_lens_list=[seq_ctx.cu_seq_lens_q for seq_ctx in seq_ctx_list])

        for i in range(grad_accumulation_steps):
            _tensor = torch.randn(B, S, D, device=self.device, dtype=self.dtype) * 2
            _input = _tensor.detach().clone().requires_grad_(True)
            _input2 = _tensor.detach().clone().requires_grad_(True)

            target = targets[i]

            # GT CE loss
            torch_ce = torch.nn.CrossEntropyLoss()
            logits = self.lm_head1(_input)
            loss1 = torch_ce(logits.float().view(-1, self.vocab_size), target.view(-1))
            loss1 = loss1 * (target >= 0).sum() / global_grad_tokens

            loss_kwargs = batches_loss_kwargs[i]
            loss_ctx = LossContext(loss_cfg, loss_kwargs)
            out = loss_ctx.forward(_input2, self.lm_head2.weight)
            loss2 = out[0]

            assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

            loss1.backward(gradient=torch.ones_like(loss1))
            loss2.backward(gradient=torch.ones_like(loss2))

            assert_verbose_allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)
            _input.grad.zero_()
            _input2.grad.zero_()

        assert_verbose_allclose(
            self.lm_head1.weight.grad,
            self.lm_head2.weight.grad,
            atol=atol,
            rtol=rtol,
        )
        self.lm_head2.weight.grad.zero_()
        self.lm_head1.weight.grad.zero_()

    @parametrize.parametrize(
        "loss_reduction, loss_mode, grad_accumulation_steps, chunk_size, atol, rtol",
        [
            ('square', "eager", 1, -1, 1e-4, 5e-2),
            ('square', "chunk", 1, 1024, 1e-4, 5e-2),
            ('square', "chunk", 1, 4096, 1e-4, 5e-2),
            ('square', "chunk", 1, 14096, 1e-4, 5e-2),
            ('square', "eager", 4, -1, 1e-4, 5e-2),
            ('square', "chunk", 4, 1024, 1e-4, 5e-2),
            ('sample', "eager", 1, -1, 1e-4, 5e-2),
            ('sample', "chunk", 1, 1024, 1e-4, 5e-2),
            ('sample', "chunk", 1, 4096, 1e-4, 5e-2),
            ('sample', "chunk", 1, 14096, 1e-4, 5e-2),
            ('sample', "eager", 4, -1, 1e-4, 5e-2),
            ('sample', "chunk", 4, 1024, 1e-4, 5e-2),
        ],
    )
    def test_other_loss_reduction(self, loss_reduction, loss_mode, grad_accumulation_steps, chunk_size, atol, rtol):
        B, S, D = 2, 4097, self.input_dim

        targets = []
        data_batch = []
        num_tokens_list = []
        for _ in range(grad_accumulation_steps):
            target = torch.randint(0, D, (B * S,), device=self.device, dtype=torch.long)
            # Assign some random number of elements as ignore_index
            num_elements_to_assign = torch.randint(
                1, B * S // 2, (1,)
            ).item()  # Random number of elements to set to ignore_index
            indices_to_assign = torch.randperm(B * S)[:num_elements_to_assign]  # Randomly select indices
            target[indices_to_assign] = -100
            target = target.reshape(1, -1)
            targets.append(target)

            # Note: input_ids/position_ids/cu_seq_lens_q/max_length_q 等都是假数据
            cu_seq_lens_q = torch.tensor([0, S, 2 * S], device=self.device, dtype=torch.int32)
            num_tokens = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]
            num_tokens_list.append(num_tokens)

            # Note: input_ids/position_ids/max_length_q 等都是假数据
            seq_ctx = SequenceContext(input_ids=torch.ones((1, 2), device=self.device, dtype=torch.long),
                                      position_ids=torch.ones((1, 2), device=self.device, dtype=torch.long),
                                      cu_seq_lens_q=cu_seq_lens_q,
                                      cu_seq_lens_k=cu_seq_lens_q,
                                      max_length_q=1,
                                      max_length_k=1,
                                      device=self.device,
                                      )
            seq_ctx.to(self.device)
            data_batch.append({'seq_ctx': seq_ctx, 'shifted_labels': target})

        global_grad_tokens, batch_loss_weights = cal_global_sum_loss_weight(targets, num_tokens_list, loss_reduction)
        for i in range(len(batch_loss_weights)):
            batch_loss_weights[i] = batch_loss_weights[i] / (global_grad_tokens + 1e-8)

        loss_cfg = CELossConfig(mode=loss_mode, chunk_size=chunk_size, loss_reduction=loss_reduction)
        LossContext = loss_cfg.loss_ctx_cls
        seq_ctx_list: list[SequenceContext] = []
        loss_ctx_input_list: list[CELossContextInputItem] = []
        for data in data_batch:
            seq_ctx = data["seq_ctx"]
            loss_ctx_input = CELossContextInputItem(shifted_labels=data["shifted_labels"])
            seq_ctx_list.append(seq_ctx)
            loss_ctx_input_list.append(loss_ctx_input)
        batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
            loss_ctx_input_list, loss_cfg, cu_seq_lens_list=[seq_ctx.cu_seq_lens_q for seq_ctx in seq_ctx_list])

        for i in range(grad_accumulation_steps):
            _tensor = torch.randn(B, S, D, device=self.device, dtype=self.dtype) * 2
            _input = _tensor.detach().clone().requires_grad_(True)
            _input2 = _tensor.detach().clone().requires_grad_(True)

            target = targets[i]
            _input_ = _input.reshape(1, -1, D)
            _input2_ = _input2.reshape(1, -1, D)

            # GT CE loss
            torch_ce = nn.CrossEntropyLoss(reduction='none')
            logits = self.lm_head1(_input_)
            loss1 = torch_ce(logits.float().view(-1, self.vocab_size), target.view(-1))
            loss1 = loss1 * batch_loss_weights[i]
            loss1 = loss1.sum()

            loss_kwargs = batches_loss_kwargs[i]
            loss_ctx = LossContext(loss_cfg, loss_kwargs)
            out = loss_ctx.forward(_input2_, self.lm_head2.weight)
            loss2 = out[0]

            assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

            loss1.backward(gradient=torch.ones_like(loss1))
            loss2.backward(gradient=torch.ones_like(loss2))

            assert_verbose_allclose(_input.grad, _input2.grad, atol=atol, rtol=rtol)
            _input.grad.zero_()
            _input2.grad.zero_()

        assert_verbose_allclose(
            self.lm_head1.weight.grad,
            self.lm_head2.weight.grad,
            atol=atol,
            rtol=rtol,
        )
        self.lm_head2.weight.grad.zero_()
        self.lm_head1.weight.grad.zero_()


def broadcast_weight(lm_head):
    rank = dist.get_rank()
    if rank == 0:
        master_weight = lm_head.weight.detach()
    else:
        master_weight = torch.empty_like(lm_head.weight)
    dist.broadcast(master_weight, src=0)
    with torch.no_grad():
        lm_head.weight.copy_(master_weight)


def prepare(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.device = 'cuda'  # liger loss must be tested on GPU
        self.dtype = torch.bfloat16
        self.input_dim = 2
        self.vocab_size = 4
        self.create_pg(self.device)
        self.lm_head1 = nn.Linear(self.input_dim, self.vocab_size, bias=False).to(device=self.device, dtype=self.dtype)
        self.lm_head2 = nn.Linear(self.input_dim, self.vocab_size, bias=False).to(device=self.device, dtype=self.dtype)
        # 确保 world_size 张卡权重完全一致
        broadcast_weight(self.lm_head1)
        self.lm_head2.weight.data = self.lm_head1.weight.data.clone()

        fn(self, *args, **kwargs)

    return wrapper


class TestCELossWithSP(DistributedTestBase):

    def create_pg(self, device):
        ret = super().create_pg(device)
        os.environ["LOCAL_RANK"] = str(dist.get_rank())
        return ret

    @parametrize.parametrize(
        "loss_mode, sp_size, grad_accumulation_steps, chunk_size, atol, rtol",
        [
            ("eager", 1, 1, -1, 1e-4, 5e-2),
            ("eager", 2, 1, -1, 1e-4, 5e-2),
            ("chunk", 2, 1, 1024, 1e-4, 5e-2),
            ("chunk", 2, 1, 4096, 1e-4, 5e-2),
            ("chunk", 2, 1, 14096, 1e-4, 5e-2),
        ],
    )
    @prepare
    def test_sp_global_loss_reduction(self, loss_mode, sp_size, grad_accumulation_steps, chunk_size, atol, rtol):
        B, S, D = 2, 4097, self.input_dim
        target = torch.randint(0, D, (B * S,), device='cuda', dtype=torch.long)
        # Assign some random number of elements as ignore_index
        num_elements_to_assign = torch.randint(
            1, B * S // 2, (1,)
        ).item()  # Random number of elements to set to ignore_index
        indices_to_assign = torch.randperm(B * S)[:num_elements_to_assign]  # Randomly select indices
        target[indices_to_assign] = -100

        target = target.reshape(1, -1)
        # target[0, :4098] = -100

        dist.broadcast(target, src=0)

        _tensor = torch.randn(B, S, D, device=self.device, dtype=self.dtype) * 2
        dist.broadcast(_tensor, src=0)

        _input = _tensor.detach().clone().requires_grad_(True)
        _input2 = _tensor.detach().clone().requires_grad_(True)
        _input_ = _input.reshape(1, -1, D)
        _input2_ = _input2.reshape(1, -1, D)

        global_grad_tokens = (target >= 0).sum()

        torch_ce = torch.nn.CrossEntropyLoss()
        logits = self.lm_head1(_input_)
        loss1 = torch_ce(logits.float().view(-1, self.vocab_size), target.view(-1))
        loss1 = loss1 * (target >= 0).sum() / global_grad_tokens

        # Note: input_ids/position_ids/cu_seq_lens_q/max_length_q 等都是假数据
        cu_seq_lens_q = torch.tensor([0, S, 2 * S], device=self.device, dtype=torch.int32)
        seq_ctx = SequenceContext(input_ids=torch.ones((1, 2), device=self.device, dtype=torch.long),
                                  position_ids=torch.ones((1, 2), device=self.device, dtype=torch.long),
                                  cu_seq_lens_q=cu_seq_lens_q,
                                  cu_seq_lens_k=cu_seq_lens_q,
                                  max_length_q=1,
                                  max_length_k=1,
                                  device=self.device,
                                  )
        seq_ctx.to(self.device)

        data_mesh = init_data_mesh(self.device, sp_size=sp_size)
        sp_mesh = data_mesh['sp']
        seq_ctx.sequence_parallel_mesh = sp_mesh
        seq_ctx_list = [seq_ctx]
        loss_ctx_input_list: list[CELossContextInputItem] = [CELossContextInputItem(shifted_labels=target)]
        if sp_size > 1:
            seq_ctx_list[0] = seq_ctx_list[0].split(sequence_parallel_mesh=sp_mesh)
            loss_ctx_input_list[0] = loss_ctx_input_list[0].sp_split(sp_mesh=sp_mesh)

        loss_cfg = CELossConfig(mode=loss_mode, chunk_size=chunk_size, loss_reduction="token")
        LossContext = loss_cfg.loss_ctx_cls
        batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
            loss_ctx_input_list, 
            loss_cfg,
            cu_seq_lens_list=[cu_seq_lens_q],
            sp_mesh=sp_mesh,
        )
        loss_kwargs = batches_loss_kwargs[0]
        loss_ctx = LossContext(loss_cfg, loss_kwargs)

        multiple_of = sp_mesh.size()
        if sp_size > 1:
            pad_input = pad_to_multiple_of(_input2_, 0, multiple_of, 1)
            _input2_ = split_for_sequence_parallel(pad_input, dim=1, sp_mesh=sp_mesh)
        
        out = loss_ctx.forward(_input2_, self.lm_head2.weight)
        loss2 = out[0]
        assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    @parametrize.parametrize(
        "loss_reduction, loss_mode, sp_size, grad_accumulation_steps, chunk_size, atol, rtol",
        [
            ('square', "eager", 1, 1, -1, 1e-4, 5e-2),
            ('square', "eager", 2, 1, -1, 1e-4, 5e-2),
            ('sample', "chunk", 1, 1, 1024, 1e-4, 5e-2),
            ('sample', "chunk", 2, 1, 4096, 1e-4, 5e-2),
            ('sample', "eager", 2, 1, -1, 1e-4, 5e-2),
            ('sample', "chunk", 2, 1, 1024, 1e-4, 5e-2),
            ('sample', "chunk", 2, 4, 4096, 1e-4, 5e-2),
        ],
    )
    @prepare
    def test_sp_others_loss_reduction(self, loss_reduction, loss_mode, sp_size, grad_accumulation_steps, chunk_size, atol, rtol):
        B, S, D = 2, 4097, self.input_dim
        target = torch.randint(0, D, (B * S,), device='cuda', dtype=torch.long)
        # Assign some random number of elements as ignore_index
        num_elements_to_assign = torch.randint(
            1, B * S // 2, (1,)
        ).item()  # Random number of elements to set to ignore_index
        indices_to_assign = torch.randperm(B * S)[:num_elements_to_assign]  # Randomly select indices
        target[indices_to_assign] = -100

        target = target.reshape(1, -1)
        target[0, :4098] = -100

        dist.broadcast(target, src=0)

        _tensor = torch.randn(B, S, D, device=self.device, dtype=self.dtype) * 2
        dist.broadcast(_tensor, src=0)

        _input = _tensor.detach().clone().requires_grad_(True)
        _input2 = _tensor.detach().clone().requires_grad_(True)
        _input_ = _input.reshape(1, -1, D)
        _input2_ = _input2.reshape(1, -1, D)

        num_tokens = torch.tensor([S, S]).to(self.device, dtype=torch.int32)
        labels_list = torch.split(target, num_tokens.tolist(), dim=1)
        loss_weights_list = []
        for _labels in labels_list:
            num_effective_tokens = (_labels >= 0).sum().item()
            loss_weight = len2weight(num_effective_tokens, loss_reduction)
            loss_weight = torch.full(_labels.shape, loss_weight, device=_labels.device)
            loss_weight[_labels == -100] = 0.0
            loss_weights_list.append(loss_weight)
        loss_weights = torch.cat(loss_weights_list, dim=1)
        global_sum_loss_weight = loss_weights.sum()
        loss_weights = loss_weights / global_sum_loss_weight

        torch_ce = nn.CrossEntropyLoss(reduction='none')
        logits = self.lm_head1(_input_)
        loss1 = torch_ce(logits.float().view(-1, self.vocab_size), target.view(-1))
        loss1 = loss1 * loss_weights
        loss1 = loss1.sum()

        # Note: input_ids/position_ids/cu_seq_lens_q/max_length_q 等都是假数据
        cu_seq_lens_q = torch.tensor([0, S, 2 * S], device=self.device, dtype=torch.int32)
        seq_ctx = SequenceContext(input_ids=torch.ones((1, 2), device=self.device, dtype=torch.long),
                                  position_ids=torch.ones((1, 2), device=self.device, dtype=torch.long),
                                  cu_seq_lens_q=cu_seq_lens_q,
                                  cu_seq_lens_k=cu_seq_lens_q,
                                  max_length_q=1,
                                  max_length_k=1,
                                  device=self.device,
                                  )
        seq_ctx.to(self.device)

        data_mesh = init_data_mesh(self.device, sp_size=sp_size)
        sp_mesh = data_mesh['sp']
        seq_ctx.sequence_parallel_mesh = sp_mesh
        seq_ctx_list = [seq_ctx]
        loss_ctx_input_list: list[CELossContextInputItem] = [CELossContextInputItem(shifted_labels=target)]
        if sp_size > 1:
            seq_ctx_list[0] = seq_ctx_list[0].split(sequence_parallel_mesh=sp_mesh)
            loss_ctx_input_list[0] = loss_ctx_input_list[0].sp_split(sp_mesh=sp_mesh)

        loss_cfg = CELossConfig(mode=loss_mode, chunk_size=chunk_size, loss_reduction="token")
        LossContext = loss_cfg.loss_ctx_cls
        batches_loss_kwargs = LossContext.build_batches_loss_kwargs(
            loss_ctx_input_list, 
            loss_cfg,
            cu_seq_lens_list=[cu_seq_lens_q],
            sp_mesh=sp_mesh,
        )
        loss_kwargs = batches_loss_kwargs[0]
        loss_ctx = LossContext(loss_cfg, loss_kwargs)

        multiple_of = sp_mesh.size()
        if sp_size > 1:
            pad_input = pad_to_multiple_of(_input2_, 0, multiple_of, 1)
            _input2_ = split_for_sequence_parallel(pad_input, dim=1, sp_mesh=sp_mesh)
        loss2, _ = loss_ctx.forward(_input2_, self.lm_head2.weight)
        assert_verbose_allclose(loss1, loss2, atol=atol, rtol=rtol)

    @property
    def world_size(self) -> int:
        return int(os.getenv("XTUNER_TEST_WORLD_SIZE", "2"))
