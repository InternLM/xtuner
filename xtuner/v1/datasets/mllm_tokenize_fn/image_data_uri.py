# Copyright (c) OpenMMLab. All rights reserved.
"""Read base64-embedded images without changing the reference-image loaders.

Per Data Format Types (V0.5 Agent), sections 7.1-7.2, image.url uses
``data:image/<subtype>;base64,<payload>`` for embedded images. The whole string
is a data URI; only the payload after the comma is base64. Bare base64 strings
remain references, while malformed data URIs must not fall back to path I/O.

This is a training I/O adapter, not the full InternValidata image validator.
Geometry metadata, MIME/container agreement and dataset policy limits remain the
responsibility of upstream validation. RGB conversion matches the path loaders;
the caller retains responsibility for EXIF orientation and model preprocessing.
"""

import base64
import binascii
import io
import re

from PIL import Image


_DATA_URI_PREFIX = re.compile(r"^\s*data:", re.IGNORECASE)
_IMAGE_HEADER = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64")


def is_data_uri(value: str) -> bool:
    """Route even malformed data URIs to validation, never to a path loader.

    Whitespace/case are only tolerated for routing; the loader does not repair
    them. Bare base64 strings are references and do not match this predicate.
    """
    return _DATA_URI_PREFIX.match(value) is not None


def load_image_data_uri(value: str) -> Image.Image:
    """Decode a canonical image data URI into an independent RGB image.

    Error messages deliberately omit the URI and suppress decoder exceptions
    that could expose the input. No temporary file or network access is used.
    """
    header, separator, payload = value.partition(",")
    if not separator or _IMAGE_HEADER.fullmatch(header) is None or not payload:
        raise ValueError("Invalid image data URI: expected data:image/<subtype>;base64,<payload>")
    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("Invalid image data URI: invalid base64 payload") from None
    # validate=True alone accepts nonzero pad bits and some unnecessary padding.
    if base64.b64encode(image_bytes).decode("ascii") != payload:
        raise ValueError("Invalid image data URI: noncanonical base64 payload")
    try:
        with io.BytesIO(image_bytes) as buffer, Image.open(buffer) as image:
            return image.convert("RGB")
    except (OSError, ValueError, SyntaxError):
        raise ValueError("Invalid image data URI: cannot decode image pixels") from None
