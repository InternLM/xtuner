# Copyright (c) OpenMMLab. All rights reserved.


def dsa_topk_source_layer(
    *,
    layer_idx: int,
    indexer_types: list[str] | None,
    index_skip_topk_offset: int,
    index_topk_freq: int,
) -> int:
    """Resolve the GLM-5.2 source layer whose DSA top-k IDs a layer
    consumes."""
    if indexer_types is not None:
        if layer_idx < len(indexer_types) and indexer_types[layer_idx] == "full":
            return layer_idx
        for source_layer_idx in range(min(layer_idx, len(indexer_types) - 1), -1, -1):
            if indexer_types[source_layer_idx] == "full":
                return source_layer_idx
        raise ValueError(f"DSA layer {layer_idx} has no preceding full indexer layer.")

    if index_topk_freq <= 1:
        return layer_idx

    source_layer_idx = layer_idx
    while (max(source_layer_idx + 1 - index_skip_topk_offset, 0) % index_topk_freq) != 0:
        source_layer_idx -= 1
    return source_layer_idx


def dsa_topk_source_layers(
    *,
    num_layers: int,
    indexer_types: list[str] | None,
    index_skip_topk_offset: int,
    index_topk_freq: int,
) -> tuple[int, ...]:
    """Return the source-layer index for every layer in one decoder stack."""
    return tuple(
        dsa_topk_source_layer(
            layer_idx=layer_idx,
            indexer_types=indexer_types,
            index_skip_topk_offset=index_skip_topk_offset,
            index_topk_freq=index_topk_freq,
        )
        for layer_idx in range(num_layers)
    )
