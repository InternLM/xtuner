import os
import re

import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss import CELossContext
from xtuner.v1.utils.activation_offload import async_save_on_cpu

from .moe import MoEModelOutputs
from .qwen3 import Qwen3MoE, Qwen3MoE30BA3Config, Qwen3MoE235BA22Config, Qwen3MoEConfig


class Qwen3VLTextMoE(Qwen3MoE):
    def to_hf_key_list(self, key: str) -> list[str]:
        if "layers" in key or "embed_tokens" in key:
            key = "model.language_model." + key

        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate)", r"layers.\1.mlp.\2", key)

        if "fused_w1w3.weight" in key:
            key = key.replace("fused_w1w3.weight", "gate_up_proj")
        elif "fused_w2.weight" in key:
            key = key.replace("fused_w2.weight", "down_proj")
        if "fused_w1w3.bias" in key:
            key = key.replace("fused_w1w3.bias", "gate_up_proj_bias")
        elif "fused_w2.bias" in key:
            key = key.replace("fused_w2.bias", "down_proj_bias")

        if key.startswith("norm."):
            return [key.replace("norm.", "model.language_model.norm.")]
        elif key.startswith("rotary_emb."):
            # FoPE has model.rotary_emb.sin_coef and model.rotary_emb.cos_coef in the safetensors
            return [key.replace("rotary_emb.", "model.language_model.rotary_emb.")]
        else:
            return [key]

    def safetensors_to_params(
        self,
        safetensors: list[torch.Tensor],
        local_tensor: torch.Tensor,
        param_name: str,
        start: int | None,
        end: int | None,
        dim: int | None,
    ):
        if len(safetensors) > 1:
            assert dim is not None, "Internal Error dim must not be None when len(safetensors) > 1"
            loaded_tensor = torch.cat(safetensors, dim=dim)
        else:
            loaded_tensor = safetensors[0]

        if "fused_w1w3.weight" in param_name:
            # hf: num_experts, hidden_size, 2 * expert_dim
            # xtuner: num_experts * 2 * expert_dim, hidden_size
            num_experts, hidden_size = loaded_tensor.shape[:2]
            loaded_tensor = loaded_tensor.transpose(1, 2)  # num_experts, 2 * expert_dim, hidden_size
            # num_experts * 2 * expert_dim, hidden_size
            loaded_tensor = loaded_tensor.reshape(-1, hidden_size)

        elif "fused_w2.weight" in param_name:
            # hf: num_experts, expert_dim, hidden_size
            # xtuner: num_experts * hidden_size, expert_dim
            loaded_tensor = loaded_tensor.transpose(1, 2).flatten(0, 1)

        if start is not None and end is not None:
            start = min(start, loaded_tensor.shape[self.FSDP_SHARD_DIM])
            end = min(end, loaded_tensor.shape[self.FSDP_SHARD_DIM])
            loaded_tensor_slice = loaded_tensor.index_select(
                dim=self.FSDP_SHARD_DIM, index=torch.arange(start, end, dtype=torch.int64, device=loaded_tensor.device)
            )
            non_pad_len = end - start
            local_tensor[:non_pad_len].copy_(loaded_tensor_slice)

            if non_pad_len < local_tensor.shape[self.FSDP_SHARD_DIM]:
                assert self.config.float8_cfg is not None
                local_tensor[non_pad_len:].copy_(0.0)  # type: ignore  # padded part must be set to 0
        else:
            local_tensor.copy_(loaded_tensor)

    def param_to_safetensor(
        self,
        safetensor: torch.Tensor,
        hf_param_name: str,
    ):
        assert isinstance(hf_param_name, str)
        if "gate_up_proj" in hf_param_name:
            # xtuner: num_experts * 2 * expert_dim, hidden_size
            # hf: num_experts, hidden_size, 2 * expert_dim
            num_experts = self.config.n_routed_experts
            hidden_size = safetensor.size(1)
            safetensor = safetensor.reshape(num_experts, -1, hidden_size)  # num_experts, 2 * expert_dim, hidden_size
            safetensor = safetensor.transpose(1, 2).contiguous()  # num_experts, hidden_size, 2 * expert_dim
        elif "down_proj" in hf_param_name:
            # xtuner: num_experts * hidden_size, expert_dim
            # hf: num_experts, expert_dim, hidden_size
            num_experts = self.config.n_routed_experts
            expert_dim = safetensor.size(1)
            safetensor = safetensor.reshape(num_experts, -1, expert_dim).transpose(1, 2).contiguous()
        return safetensor

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def _forward(
        self,
        seq_ctx: SequenceContext,  # todo(@yehaochen): support intra layer micro-batch
        loss_ctx: CELossContext | None,
        return_router_logits: bool = False,
    ) -> MoEModelOutputs:
        input_ids = seq_ctx.input_ids
        position_ids = seq_ctx.position_ids

        if input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = seq_ctx.inputs_embeds

        # create position embeddings to be shared across the decoder layers
        assert position_ids is not None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        output: dict = {}  # type: ignore
        if self.config.return_hidden_states:
            output["hidden_states"] = []

        output["router_logits"] = {}
        output["router_weights"] = {}

        self._mark_dynamic(seq_ctx)

        # =====================================================
        deepstack_visual_embeds = seq_ctx.deepstack_visual_embeds
        visual_pos_masks = seq_ctx.visual_pos_masks
        # =====================================================

        for idx, decoder_layer in self.layers.items():
            if int(idx) < self.config.first_k_dense_replace:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    seq_ctx=seq_ctx,
                )
            else:
                if int(os.getenv("XTUNER_ACTIVATION_OFFLOAD", "0")) == 1:
                    offload_stream = decoder_layer._get_fsdp_state()._comm_ctx.all_gather_stream
                    with async_save_on_cpu(
                        h2d_stream=offload_stream,
                        d2h_stream=offload_stream,
                        block_idx=int(idx),
                        depth=len(self.layers),
                        custom_check_fn=lambda x: x.data_ptr() == hidden_states.data_ptr(),
                    ):
                        hidden_states, router_results, router_weights = decoder_layer(
                            hidden_states,
                            position_embeddings=position_embeddings,
                            seq_ctx=seq_ctx,
                        )

                else:
                    hidden_states, router_results, router_weights = decoder_layer(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        seq_ctx=seq_ctx,
                    )

                output["router_logits"][f"layer{idx}"] = router_results
                output["router_weights"][f"layer{idx}"] = router_weights

            if deepstack_visual_embeds is not None and ((idx := int(idx)) in range(len(deepstack_visual_embeds))):
                assert visual_pos_masks is not None
                hidden_states = self._deepstack_process(hidden_states, visual_pos_masks, deepstack_visual_embeds[idx])

            if self.config.return_hidden_states:
                output["hidden_states"].append(hidden_states)

        hidden_states = self.norm(hidden_states)

        loss, (logits, extra_info) = self.lm_head(hidden_states, loss_ctx)  # type: ignore
        output["loss"] = loss
        output["logits"] = logits
        output["extra_info"] = extra_info

        router_logits_list = list(output["router_logits"].values())  # type: ignore
        router_weights_list = list(output["router_weights"].values())  # type: ignore
        router_logits = self._select_non_pad_router_logits(router_logits_list, seq_ctx.mask)
        router_weights = self._select_non_pad_router_logits(router_weights_list, seq_ctx.mask)

        if self.balancing_loss:
            balancing_loss = self.balancing_loss(
                router_weights=router_weights,
                n_routed_experts=self.config.n_routed_experts,
                num_experts_per_tok=self.config.num_experts_per_tok,
            )
            output["balancing_loss"] = balancing_loss

        if self.z_loss:
            z_loss = self.z_loss(router_logits=router_logits)
            output["z_loss"] = z_loss

        tokens_per_expert_global = self._cal_tokens_per_expert(router_logits)
        output["tokens_per_expert_global"] = tokens_per_expert_global

        del router_logits

        if self.config.return_router_results or return_router_logits:
            # raise NotImplementedError
            # TODO: Move router logits to CPU is cost
            for layer_name, router_logits in output["router_logits"].items():
                output["router_logits"][layer_name] = router_logits.detach().unsqueeze(0)
        else:
            output["router_logits"] = None

        return MoEModelOutputs(**output)  # type: ignore[typeddict-item]


class Qwen3VLTextMoEBaseConfig(Qwen3MoEConfig):
    def build(self) -> Qwen3MoE:
        return Qwen3VLTextMoE(self)


class Qwen3VLTextMoE30BA3Config(Qwen3MoE30BA3Config):
    def build(self) -> Qwen3MoE:
        return Qwen3VLTextMoE(self)


class Qwen3VLTextMoE235BA22Config(Qwen3MoE235BA22Config):
    def build(self) -> Qwen3MoE:
        return Qwen3VLTextMoE(self)
