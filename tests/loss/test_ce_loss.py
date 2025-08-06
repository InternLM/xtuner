from unittest import TestCase
import torch
import torch.nn as nn
from xtuner.v1.loss import CELossContext
from xtuner.v1.loss.ce_loss import len2weight
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.utils.test_utils import assert_verbose_allclose
from xtuner.v1.loss.utils import cal_global_grad_tokens
import parametrize


class TestCELossModel(TestCase):
    def setUp(self) -> None:
        self.device = 'cuda'  # liger loss must be tested on GPU
        self.dtype = torch.bfloat16
        self.input_dim = 2
        self.vocab_size = 4
        self.lm_head1 = nn.Linear(self.input_dim , self.vocab_size, bias=False).to(device=self.device, dtype=self.dtype)
        self.lm_head2 = nn.Linear(self.input_dim , self.vocab_size, bias=False).to(device=self.device, dtype=self.dtype)
        self.lm_head2.weight.data = self.lm_head1.weight.data.clone()

    @parametrize.parametrize(
        "loss_class, grad_accumulation_steps, chunk_size, atol, rtol",
        [
            ("cross_entropy", 1, -1, 1e-3, 5e-2),
            ("liger_cross_entropy", 1, -1, 1e-3, 5e-2),
            ("chunk_cross_entropy", 1, 1024, 1e-3, 5e-2),
            ("chunk_cross_entropy", 1, 4096, 1e-3, 5e-2),
            ("chunk_cross_entropy", 1, 14096, 1e-3, 5e-2),
            ("cross_entropy", 4, -1, 1e-3, 5e-2),
            ("liger_cross_entropy", 4, -1, 1e-3, 5e-2),
            ("chunk_cross_entropy", 4, 1024, 1e-3, 5e-2),
            ("chunk_cross_entropy", 4, 4096, 1e-3, 5e-2),
            ("chunk_cross_entropy", 4, 14096, 1e-3, 5e-2),
        ],
    )
    def test_global_loss_reduction(self, loss_class, grad_accumulation_steps, chunk_size, atol, rtol):
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
            seq_ctx = SequenceContext(input_ids=torch.ones((1, 2), device=self.device, dtype=torch.long),
                                      position_ids=torch.ones((1, 2), device=self.device, dtype=torch.long),
                                      cu_seq_lens_q=torch.tensor(1, device=self.device, dtype=torch.int32),
                                      cu_seq_lens_k=torch.tensor(1, device=self.device, dtype=torch.int32),
                                      max_length_q=1,
                                      max_length_k=1,
                                      device=self.device,
                                      )
            seq_ctx.to(self.device)
            data_batch.append({'seq_ctx': seq_ctx, 'labels': target})

        global_grad_tokens = cal_global_grad_tokens(targets)

        loss_ctx = CELossContext(loss_class=loss_class, chunk_size=chunk_size)
        data_batch = loss_ctx.build_list_ctx(data_batch, grad_accumulation_steps=grad_accumulation_steps)

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

            loss_ctx = data_batch[i]['loss_ctx']
            loss2, _ = loss_ctx.forward(_input2, self.lm_head2.weight)

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
        "loss_reduction, loss_class, grad_accumulation_steps, chunk_size, atol, rtol",
        [
            ('square', "cross_entropy", 1, -1, 1e-3, 5e-2),
            ('square', "liger_cross_entropy", 1, -1, 1e-3, 5e-2),
            ('square', "chunk_cross_entropy", 1, 1024, 1e-3, 5e-2),
            ('square', "chunk_cross_entropy", 1, 4096, 1e-3, 5e-2),
            ('square', "chunk_cross_entropy", 1, 14096, 1e-3, 5e-2),
            ('square', "cross_entropy", 4, -1, 1e-3, 5e-2),
            ('square', "liger_cross_entropy", 4, -1, 1e-3, 5e-2),
            ('square', "chunk_cross_entropy", 4, 1024, 1e-3, 5e-2),
            ('square', "chunk_cross_entropy", 4, 4096, 1e-3, 5e-2),
            ('square', "chunk_cross_entropy", 4, 14096, 1e-3, 5e-2),
            ('sample', "liger_cross_entropy", 1, -1, 1e-3, 5e-2),
            ('sample', "chunk_cross_entropy", 1, 1024, 1e-3, 5e-2),
            ('sample', "chunk_cross_entropy", 4, 4096, 1e-3, 5e-2),
        ],
    )
    def test_other_loss_reduction(self, loss_reduction, loss_class, grad_accumulation_steps, chunk_size, atol, rtol):
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
            target = target.reshape(1, -1)
            targets.append(target)

            # Note: input_ids/position_ids/cu_seq_lens_q/max_length_q 等都是假数据
            cu_seq_lens_q = torch.tensor([0, S, 2 * S], device=self.device, dtype=torch.int32)
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
            data_batch.append({'seq_ctx': seq_ctx, 'labels': target})

        loss_ctx = CELossContext(loss_reduction=loss_reduction, loss_class=loss_class, chunk_size=chunk_size)
        data_batch = loss_ctx.build_list_ctx(data_batch, grad_accumulation_steps=grad_accumulation_steps)

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
            num_tokens = [S, S]
            labels_list = torch.split(target, num_tokens, dim=1)
            loss_weights_list = []
            for _labels in labels_list:
                num_effective_tokens = (_labels >= 0).sum().item()
                loss_weight = len2weight(num_effective_tokens, loss_reduction)
                loss_weights_list.append(torch.full(_labels.shape, loss_weight, device=_labels.device))
            loss_weights = torch.cat(loss_weights_list, dim=1)
            loss1 = loss1 * loss_weights
            loss1 = loss1.sum() / loss_weights.sum()
            loss1 = loss1 / grad_accumulation_steps

            loss_ctx = data_batch[i]['loss_ctx']
            loss2, _ = loss_ctx.forward(_input2_, self.lm_head2.weight)

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
