import hashlib
import os
import stat
import sys
from urllib.parse import urlsplit


class UnsupportedSourceIO(ValueError):
    pass


def require_local_source(uri: str) -> None:
    if type(uri) is not str or not uri or "\x00" in uri:
        raise UnsupportedSourceIO("optimized source I/O requires a local regular-file path")
    try:
        scheme = urlsplit(uri).scheme
    except ValueError as error:
        raise UnsupportedSourceIO("optimized source I/O requires a local regular-file path") from error
    if scheme:
        raise UnsupportedSourceIO(
            "optimized URI/OSS source I/O has no qualified bounded reader; "
            "choose unsupported_policy='eager' for the native path"
        )


def read_verified_local_source(uri: str, source_bytes: int, content_sha256: str) -> bytes:
    require_local_source(uri)
    if type(source_bytes) is not int or not 0 < source_bytes < sys.maxsize:
        raise ValueError("certified source byte count must be an exact positive bounded integer")
    if (
        type(content_sha256) is not str
        or len(content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in content_sha256)
    ):
        raise ValueError("certified source revision needs a lowercase SHA256")
    descriptor = os.open(uri, os.O_RDONLY | os.O_NONBLOCK)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise UnsupportedSourceIO("optimized source I/O requires a local regular file")
        if metadata.st_size != source_bytes:
            raise ValueError("image bytes differ from certified source revision")
        with os.fdopen(descriptor, "rb", buffering=0, closefd=False) as stream:
            content = stream.read(source_bytes + 1)
    finally:
        os.close(descriptor)
    if len(content) != source_bytes or hashlib.sha256(content).hexdigest() != content_sha256:
        raise ValueError("image bytes differ from certified source revision")
    return content
