import os
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss
except ImportError:
    LigerFusedLinearCrossEntropyLoss = None


def minicpmv_forward(self, data, **kwargs):
    vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
    use_liger_kernel = os.environ.get('USE_LIGER_KERNEL')
    labels = data.get('labels')
    if use_liger_kernel and labels is not None and self.training:
        output_attentions = self.config.output_attentions
        output_hidden_states = self.config.output_hidden_states
        return_dict = self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.llm.model(
            inputs_embeds=vllm_embedding,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten tokens
        shift_hidden_states = shift_hidden_states.view(-1, self.llm.config.hidden_size)
        shift_labels = shift_labels.view(-1)

        if LigerFusedLinearCrossEntropyLoss is None:
            raise ImportError('LigerFusedLinearCrossEntropyLoss is not available, '
                              'please install liger-kernel by "pip install liger_kernel".')
        lce = LigerFusedLinearCrossEntropyLoss()
        loss = lce(self.llm.lm_head.weight, shift_hidden_states, shift_labels)
        if not return_dict:
            output = (None,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    else:
        return self.llm(
            input_ids=None,
            inputs_embeds=vllm_embedding,
            labels=labels,
            **kwargs
        )
