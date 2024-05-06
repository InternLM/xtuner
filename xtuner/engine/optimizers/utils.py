
def get_layer_depth_for_CLIPVisionModel(self, param_name: str, prefix: str = 'vision_model.'):
    """Get the layer-wise depth of a parameter.

    Args:
        param_name (str): The name of the parameter.
        prefix (str): The prefix for the parameter.
            Defaults to an empty string.

    Returns:
        Tuple[int, int]: The layer-wise depth and the num of layers.

    Note:
        The first depth is the stem module (``layer_depth=0``), and the
        last depth is the subsequent module (``layer_depth=num_layers-1``)
    """
    num_layers = self.config.num_hidden_layers + 2

    if not param_name.startswith(prefix):
        # For subsequent module like head
        return num_layers - 1, num_layers

    param_name = param_name[len(prefix):]

    if param_name.startswith('embeddings'):
        layer_depth = 0
    elif param_name.startswith('pre_layrnorm'):
        layer_depth = 0
    elif param_name.startswith('encoder.layers'):
        layer_id = int(param_name.replace('encoder.', '').split('.')[1])
        layer_depth = layer_id + 1
    else:
        layer_depth = num_layers - 1

    return layer_depth, num_layers
