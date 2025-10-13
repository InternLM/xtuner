import re
from .qwen3 import Qwen3MoE,Qwen3MoE30BA3Config
from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss import CELossContext
from .moe import MoEModelOutputs
from xtuner.v1.utils.activation_offload import async_save_on_cpu
import os
from xtuner.v1.module import NoAuxRouterConfig


class Qwen3VLTextMoE(Qwen3MoE):
    def to_hf_key_list(self, key: str) -> list[str]:
        if "layers" in key or "embed_tokens" in key:
            key = "model." + key

        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate)", r"layers.\1.mlp.\2", key)

        n_routed_experts = self.config.n_routed_experts

        if "fused_w1w3.weight" in key:
            w1w3_keys: list[str] = []

            for i in range(n_routed_experts):
                w1w3_keys.append(key.replace("fused_w1w3.weight", f"{i}.gate_proj.weight"))
                w1w3_keys.append(key.replace("fused_w1w3.weight", f"{i}.up_proj.weight"))

            return w1w3_keys

        elif "fused_w2.weight" in key:
            w2_keys: list[str] = []
            for i in range(n_routed_experts):
                w2_keys.append(key.replace("fused_w2.weight", f"{i}.down_proj.weight"))
            return w2_keys

        elif key.startswith("norm."):
            return [key.replace("norm.", "model.norm.")]
        else:
            return [key]

    def _forward(
            self,
            seq_ctx: SequenceContext,  # todo(@yehaochen): support intra layer micro-batch
            loss_ctx: CELossContext | None,
    ) -> MoEModelOutputs:
        input_ids = seq_ctx.input_ids
        position_ids = seq_ctx.position_ids

        if input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = seq_ctx.inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        output: dict = {}  # type: ignore
        if self.config.return_hidden_states:
            output["hidden_states"] = []

        output["router_logits"] = {}

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
                        hidden_states, router_results = decoder_layer(
                            hidden_states,
                            position_embeddings=position_embeddings,
                            seq_ctx=seq_ctx,
                        )

                else:
                    hidden_states, router_results = decoder_layer(
                        hidden_states,
                        position_embeddings=position_embeddings,
                        seq_ctx=seq_ctx,
                    )

                output["router_logits"][f"layer{idx}"] = router_results

            if deepstack_visual_embeds is not None and idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[idx],
                )

            if self.config.return_hidden_states:
                output["hidden_states"].append(hidden_states)

        hidden_states = self.norm(hidden_states)

        loss, logits = self.lm_head(hidden_states, loss_ctx)  # type: ignore
        output["loss"] = loss
        output["logits"] = logits

        router_logits_list = list(output["router_logits"].values())  # type: ignore
        router_logits = self._select_non_pad_router_logits(router_logits_list, seq_ctx.mask)

        if self.balancing_loss:
            balancing_loss = self.balancing_loss(
                router_logits=router_logits,
                n_routed_experts=self.config.n_routed_experts,
                num_experts_per_tok=self.config.num_experts_per_tok,
            )
            output["balancing_loss"] = balancing_loss

        if self.z_loss:
            z_loss = self.z_loss(router_logits=router_logits)
            output["z_loss"] = z_loss

        if isinstance(self.config.router, NoAuxRouterConfig) and self.config.router.router_bias_update_speed > 0:
            tokens_per_expert_global = self._cal_tokens_per_expert(router_logits)
            output["tokens_per_expert_global"] = tokens_per_expert_global

        del router_logits

        if self.config.return_router_results:
            raise NotImplementedError
            # TODO: Move router logits to CPU is cost
            # for layer_name, router_logits in output["router_logits"].items():
            #     output["router_logits"][layer_name] = router_logits.detach().cpu().unsqueeze(0)
        else:
            output["router_logits"] = None

        return MoEModelOutputs(**output)  # type: ignore[typeddict-item]


class Qwen3VLTextMoE30BA3Config(Qwen3MoE30BA3Config):
    def build(self) -> Qwen3MoE:
        return Qwen3VLTextMoE(self)
