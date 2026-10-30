"""Embedded image I/O contract, independent of model weights and OSS access."""

import base64
import io
import traceback

import pytest
from PIL import Image

from xtuner.v1.datasets.mllm_tokenize_fn.base_mllm_tokenize_fn import load_image
from xtuner.v1.datasets.mllm_tokenize_fn.image_data_uri import is_data_uri, load_image_data_uri
from xtuner.v1.datasets.utils import apply_exif_orientation


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_png_matches_file_loader(tmp_path, mode):
    path = tmp_path / "image.png"
    Image.new(mode, (35, 21)).save(path)
    uri = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    actual = load_image_data_uri(uri)
    expected = load_image(str(path))
    assert actual.mode == expected.mode == "RGB"
    assert actual.size == expected.size
    # Pixels remain usable after the loader closes the source buffer.
    assert actual.tobytes() == expected.tobytes()


def test_jpeg_exif_matches_file_loader(tmp_path):
    path = tmp_path / "image.jpg"
    image = Image.new("RGB", (35, 21), (200, 50, 10))
    image.putpixel((0, 0), (0, 0, 255))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)
    uri = "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    embedded = load_image_data_uri(uri)
    reference = load_image(str(path))
    assert embedded.getexif()[274] == reference.getexif()[274] == 6
    actual, expected = apply_exif_orientation(embedded), apply_exif_orientation(reference)
    assert actual.size == expected.size == (21, 35)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "value",
    [
        "relative.png",
        "/absolute.png",
        "s3://bucket/a.png",
        "https://host/a.png",
        "aGVsbG8=",
        "data.png",
        " data:image/png;base64,YQ==",
        "DATA:image/png;base64,YQ==",
        "\t\nDaTa:image/png;base64,YQ==",
    ],
)
def test_references_are_not_data_uris(value):
    assert not is_data_uri(value)


@pytest.mark.parametrize(
    "value",
    [
        "data:image/png;base64,",
        "data:image/png;charset=utf-8;base64,YQ==",
        "data:image/png,YQ==",
        "data:text/plain;base64,YQ==",
        "data:image/png;base64,YQ",
        "data:image/png;base64,YQ===",
        "data:image/png;base64,YR==",  # Nonzero padding bits.
        "data:image/png;base64,YWJj=",  # Unnecessary padding.
        "data:image/png;base64,_w==",
        "data:image/png;base64,/w==\n",
        "data:image/png;base64,/ w==",
        "data:image/png;base64,你好",
        "data:image/png;base64,cHJpdmF0ZS1ub3QtYW4taW1hZ2U=",
    ],
)
def test_invalid_data_never_falls_back_or_echoes_payload(value):
    assert is_data_uri(value)
    with pytest.raises(ValueError, match="Invalid image data URI") as error:
        load_image_data_uri(value)
    formatted = "".join(traceback.format_exception(error.type, error.value, error.tb))
    assert value not in str(error.value)
    assert "cHJpdmF0ZS1ub3QtYW4taW1hZ2U=" not in formatted


def test_truncated_image_is_rejected():
    buffer = io.BytesIO()
    Image.new("RGB", (128, 128)).save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()[:50]).decode("ascii")
    with pytest.raises(ValueError, match="cannot decode image pixels"):
        load_image_data_uri("data:image/png;base64," + payload)
