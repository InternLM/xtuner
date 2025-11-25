import re

import torch

from xtuner.v1.data_proto import SequenceContext
from xtuner.v1.loss import CELossContext
from xtuner.v1.model.base import ModelOutputs

from .qwen3 import Qwen3Dense, Qwen3Dense4BConfig, Qwen3Dense8BConfig


class Qwen3VLTextDense(Qwen3Dense):
    def to_hf_key_list(self, key: str) -> list[str]:
        if self.config.tie_word_embeddings and "lm_head" in key:
            key = key.replace("lm_head", "embed_tokens")

        if "layers" in key or "embed_tokens" in key:
            key = "model.language_model." + key

        if "layers" in key:
            key = re.sub(r"layers\.(\d+)\.(experts|gate)", r"layers.\1.mlp.\2", key)

        if key.startswith("norm."):
            return [key.replace("norm.", "model.language_model.norm.")]
        else:
            return [key]

    def _deepstack_process(
        self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor, visual_embeds: torch.Tensor
    ):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
        hidden_states[visual_pos_masks, :] = local_this
        return hidden_states

    def forward(
        self,
        seq_ctx: SequenceContext,  # todo(@yehaochen): support intra layer micro-batch
        loss_ctx: CELossContext,
    ) -> ModelOutputs:
        input_ids = seq_ctx.input_ids
        position_ids = seq_ctx.position_ids

        if input_ids is not None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = seq_ctx.inputs_embeds

        # create position embeddings to be shared across the decoder layers
        assert position_ids is not None
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        output: dict = {}
        if self.config.return_hidden_states:
            output["hidden_states"] = []

        # =====================================================
        deepstack_visual_embeds = seq_ctx.deepstack_visual_embeds
        visual_pos_masks = seq_ctx.visual_pos_masks
        # =====================================================

        for idx, decoder_layer in self.layers.items():
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings,
                seq_ctx,
            )

            if deepstack_visual_embeds is not None and (idx := int(idx) in range(len(deepstack_visual_embeds))):
                assert visual_pos_masks is not None
                hidden_states = self._deepstack_process(hidden_states, visual_pos_masks, deepstack_visual_embeds[idx])

            if self.config.return_hidden_states:
                output["hidden_states"].append(hidden_states)

        hidden_states = self.norm(hidden_states)

        loss, (logits, extra_info) = self.lm_head(hidden_states, loss_ctx)
        output["loss"] = loss
        output["logits"] = logits
        output["extra_info"] = extra_info
        return ModelOutputs(**output)  # type: ignore[typeddict-item]


class Qwen3VLTextDense4BConfig(Qwen3Dense4BConfig):
    def build(self) -> Qwen3VLTextDense:
        return Qwen3VLTextDense(self)


class Qwen3VLTextDense8BConfig(Qwen3Dense8BConfig):
    def build(self) -> Qwen3VLTextDense:
        return Qwen3VLTextDense(self)
