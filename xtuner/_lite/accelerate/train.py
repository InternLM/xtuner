def packed_sequence_fwd_and_bwd(model, packed_input_ids, packed_pos_ids,
                                packed_labels, unpack_sizes):
    from mmengine import MessageHub
    varlen_ctx = MessageHub.get_instance('varlen_attention_context')
    varlen_ctx.update_info('chunk_sizes', unpack_sizes)

    outputs = model(
        input_ids=packed_input_ids,
        labels=packed_labels,
        position_ids=packed_pos_ids,
    )
    varlen_ctx.update_info('chunk_sizes', None)
    outputs.loss.backward()
    return outputs.loss.item()
