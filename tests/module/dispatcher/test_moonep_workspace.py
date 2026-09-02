import unittest

import torch
import torch.distributed as dist

from xtuner._testing import DeterministicDDPTestCase
from xtuner.v1.module.dispatcher.moonep_workspace import _ExpertVMMWorkspace


@unittest.skipUnless(torch.cuda.device_count() >= 8, "requires 8 CUDA devices")
class TestMoonEPOneSegmentWorkspace(DeterministicDDPTestCase):
    def test_ep2_ep4_ep8_share_the_one_segment_contract(self) -> None:
        """Exercise the real VMM/transport path for every supported EP size."""
        self.create_pg("cuda")
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()

        for ep_size in (2, 4, 8):
            rank_lists = [list(range(start, start + ep_size)) for start in range(0, world_size, ep_size)]
            groups = [dist.new_group(ranks=ranks) for ranks in rank_lists]
            ep_group = groups[global_rank // ep_size]
            ep_rank = dist.get_rank(ep_group)
            experts_per_rank = 2
            num_experts = experts_per_rank * ep_size
            device = torch.device("cuda", global_rank)

            workspace = _ExpertVMMWorkspace.allocate(
                projection_shapes=((512, 1024), (1024, 512)),
                num_experts=num_experts,
                ep_group=ep_group,
                gradient_slots=2,
            )
            from moonep import Buffer

            buffer = Buffer(
                S=64,
                H=128,
                K=1,
                E=num_experts,
                num_ep_ranks=ep_size,
                B=experts_per_rank,
                num_sms=8,
                token_padding=16,
                group=ep_group,
                explicitly_destroy=True,
            )
            try:
                for projection, landing in enumerate(workspace.landing(0)):
                    for local_expert in range(experts_per_rank):
                        expert = ep_rank * experts_per_rank + local_expert
                        landing[local_expert].fill_(100 * projection + expert + 1)

                # A globally hot expert forces duplicate weight placement.
                topk_ids = torch.zeros((64, 1), dtype=torch.int32, device=device)
                tokens_per_expert = torch.bincount(topk_ids.flatten(), minlength=num_experts).to(torch.int32)
                hidden = torch.randn(64, 128, dtype=torch.bfloat16, device=device)
                _, _, cu_seqlens, plan = buffer.dispatch(
                    hidden,
                    topk_experts_sk=topk_ids,
                    tokens_per_expert=tokens_per_expert,
                )

                local_weights, gradients_0 = workspace.prefetch_weights(
                    buffer=buffer,
                    plan=plan,
                    generation=0,
                    grad_slot=0,
                )
                _, gradients_1 = workspace.prefetch_weights(
                    buffer=buffer,
                    plan=plan,
                    generation=0,
                    grad_slot=1,
                )

                # One grouped GEMM receives exactly one contiguous [B+B]
                # segment.  Its home prefix aliases the current FSDP landing.
                assert workspace.local_token_counts(cu_seqlens).shape == (2 * experts_per_rank,)
                for projection, weight in enumerate(local_weights):
                    assert weight.is_contiguous()
                    assert weight.shape[0] == 2 * experts_per_rank
                    assert torch.equal(
                        weight[:experts_per_rank],
                        workspace.landing(0)[projection],
                    )

                # Gradient slots are independent. Duplicate BF16 partials are
                # returned to their home chunk and cleared without repacking.
                for gradient in gradients_0:
                    gradient.zero_()
                    gradient[experts_per_rank:].fill_(1)
                for gradient in gradients_1:
                    gradient.fill_(7)
                home_grads = workspace.complete_gradients(
                    buffer=buffer,
                    plan=plan,
                    local_grads=gradients_0,
                    grad_slot=0,
                )
                assert all(torch.count_nonzero(gradient[experts_per_rank:]) == 0 for gradient in gradients_0)
                assert all(torch.all(gradient == 7) for gradient in gradients_1)

                copied = torch.count_nonzero(plan.experts_to_copy >= 0)
                dist.all_reduce(copied, group=ep_group)
                assert copied > 0
                hot_gradient = home_grads[0][0].float().sum() if ep_rank == 0 else torch.zeros((), device=device)
                dist.all_reduce(hot_gradient, group=ep_group)
                assert hot_gradient > 0
            finally:
                buffer.destroy()
                workspace.destroy()
            dist.barrier()

    @property
    def world_size(self) -> int:
        return 8
