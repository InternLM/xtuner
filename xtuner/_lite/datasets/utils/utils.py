# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Mapping

import torch
from PIL import Image

_EXIF_ORIENT = 274  # exif 'Orientation' tag


def apply_exif_orientation(image):
    """Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image


def move_data_to_device(data, device="cuda"):
    """Prepares one `data` before feeding it to the model, be it a tensor or a
    nested list/dictionary of tensors."""
    if isinstance(data, Mapping):
        return type(data)({k: move_data_to_device(v) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(move_data_to_device(v) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        return data.to(non_blocking=True, **kwargs)
    return data
