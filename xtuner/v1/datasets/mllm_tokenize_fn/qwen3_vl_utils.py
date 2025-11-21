# Copyright (c) OpenMMLab. All rights reserved.
import io
import os
import re
import time
from typing import Literal

import numpy as np
import torch
from PIL import Image

from transformers.image_utils import ChannelDimension
from transformers.video_utils import to_channel_dimension_format
from xtuner.v1.utils.oss_utils import get_oss_backend


try:
    from decord import VideoReader
except ImportError:
    pass


def pil_loader(img_str):
    buff = io.BytesIO(img_str)
    img = Image.open(buff)
    return img.convert("RGB")


def extract_frame_number(filename):
    # Extract the numeric part from the filename using regular expressions
    match = re.search(r"_(\d+).jpg$", filename)
    return int(match.group(1)) if match else -1


def sort_frames(frame_paths):
    # Extract filenames from each path and sort by their numeric part
    return sorted(frame_paths, key=lambda x: extract_frame_number(os.path.basename(x)))


def numpy_to_tensor(frames):
    result = []
    for frame in frames:
        video = to_channel_dimension_format(frame, ChannelDimension.FIRST)
        # not using F.to_tensor as it doesn't handle (C, H, W) numpy arrays
        video = torch.from_numpy(video).contiguous()
        result.append(video)
    return result


def read_frames_folder(video_path, frames_indices, timestamps=None, client=None):
    oss_read_time = 0
    if "s3://" in video_path:
        assert client is not None, "client should be provided for s3 backend"
        image_list = sort_frames(client.list(video_path))
        image_list = [os.path.join(video_path.split(image.split("/")[0])[0], image) for image in image_list]
        frames = []

        for image in image_list:
            start_time = time.time()
            image_byte = client.get(image)
            oss_read_time += time.time() - start_time
            frame = Image.open(io.BytesIO(image_byte))
            frames.append(np.array(frame))
    else:
        image_list = sort_frames(list(os.listdir(video_path)))
        frames = []
        for image in image_list:
            fp = os.path.join(video_path, image)
            frame = Image.open(fp).convert("RGB")
            frames.append(np.array(frame))
    vlen = len(frames)

    if isinstance(frames_indices, list):
        assert timestamps is not None, "timestamps should be provided when frames_indices is a list"
        assert len(frames_indices) == len(timestamps) * 2, "frames_indices and timestamps should have the same length"
        # 如果外面提供了，则用 index 进行采样，但是如果采样错误，则改为随机均匀采样。这种情况要注意，实际上是不合理的，说明数据存储有问题
        try:
            frames = numpy_to_tensor([frames[i] for i in frames_indices])
        except Exception as e:
            print(
                f"！！！Warning: Error sample frames from {video_path} of index {frames_indices}: {e}. Rand {len(frames_indices)} frames."
            )
            timestamps = None  # 防止错误，强制清空
            if vlen > len(frames_indices):
                # 均匀采样
                frames_indices = np.linspace(0, vlen - 1, len(frames_indices)).round().astype(int)
            else:
                frames_indices = np.arange(vlen)
            frames = numpy_to_tensor([frames[i] for i in frames_indices])
    else:
        # 如果外面没有提供 frame index，则随机均匀采样，并且时间戳清空
        assert timestamps is None, "timestamps should be None when frames_indices is an int"
        if vlen > frames_indices:  # 此时 frames_indices=num_frames
            # 均匀采样
            frames_indices = np.linspace(0, vlen - 1, frames_indices).round().astype(int)
        else:
            frames_indices = np.arange(vlen)
        frames = numpy_to_tensor([frames[i] for i in frames_indices])
    return frames, oss_read_time, vlen, frames_indices, timestamps


def read_frames_decord(
    video_path,
    frames_indices,
    timestamps=None,
    client=None,
):
    decord_video_threads = int(os.getenv("XTUNER_DECORD_VIDEO_THREADS", 0))
    start_time = time.time()
    oss_read_time = 0
    if "s3://" in video_path:
        assert client is not None, "client should be provided for s3 backend"
        video_bytes = client.get(video_path)
        oss_read_time = time.time() - start_time
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=decord_video_threads)
        start_time = time.time()
    else:
        video_reader = VideoReader(video_path, num_threads=decord_video_threads)
        start_time = time.time()
    vlen = len(video_reader)

    if isinstance(frames_indices, list):
        assert timestamps is not None, "timestamps should be provided when frames_indices is a list"
        assert len(frames_indices) == len(timestamps) * 2, "frames_indices and timestamps should have the same length"
        # 如果外面提供了，则用 index 进行采样，但是如果采样错误，则改为随机均匀采样。这种情况要注意，实际上是不合理的，说明数据存储有问题
        try:
            frames = video_reader.get_batch(frames_indices).asnumpy()  # (T, H, W, C), np.uint8
        except Exception as e:
            print(
                f"！！！Warning: Error sample frames from {video_path} of index {frames_indices}: {e}. Rand {len(frames_indices)} frames."
            )
            timestamps = None  # 防止错误，强制清空
            frames_indices = np.linspace(0, vlen - 1, len(frames_indices)).round().astype(int)
            frames = video_reader.get_batch(frames_indices).asnumpy()  # (T, H, W, C), np.uint8
    else:
        # 如果外面没有提供 frame index，则随机均匀采样，并且时间戳清空
        assert timestamps is None, "timestamps should be None when frames_indices is an int"
        frames_indices = np.linspace(0, vlen - 1, frames_indices).round().astype(int)
        frames = video_reader.get_batch(frames_indices).asnumpy()  # (T, H, W, C), np.uint8
    video_get_batch_time = time.time() - start_time
    frames = numpy_to_tensor(frames)
    return frames, oss_read_time, video_get_batch_time, vlen, frames_indices, timestamps


# qwen3 vl 一定是均匀采样
def read_qwen3_vl_video(
    path,
    frames_indices,
    timestamps=None,
    client=None,
    debug=False,
    oss_time_log_thr=10,
):
    start_time = time.time()
    video_get_batch_time = 0
    if path.endswith("/"):
        frames, oss_read_time, vlen, frame_indices, timestamps = read_frames_folder(
            path, frames_indices, timestamps, client=client
        )
    elif path.endswith(".gif"):
        raise NotImplementedError("gif format is not supported")
    elif (
        path.endswith(".mp4")
        or path.endswith(".avi")
        or path.endswith(".mov")
        or path.endswith(".webm")
        or path.endswith(".flv")
        or path.endswith(".wmv")
        or path.endswith(".mkv")
        or path.endswith(".rmvb")
        or path.endswith(".ts")
    ):
        frames, oss_read_time, video_get_batch_time, vlen, frame_indices, timestamps = read_frames_decord(
            path, frames_indices, timestamps, client=client
        )
    else:
        raise ValueError(f"Unsupported video format: {path}")
    end_time = time.time() - start_time
    if debug and end_time > oss_time_log_thr:
        print(
            f"[Warning] OSS read video {path} cost {end_time} seconds, "
            f"oss_read_time {oss_read_time}, video_get_batch_time {video_get_batch_time}, vlen {vlen}"
        )
    return frames, frame_indices, timestamps


class Qwen3VLOSSLoader:
    # Singleton instance
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, backend: Literal["petrel"] = "petrel", debug: bool = False, oss_time_log_thr: int = 10, **kwargs
    ):
        self.client = get_oss_backend(backend, **kwargs)
        self.debug = debug
        self.oss_time_log_thr = oss_time_log_thr

    def __call__(self, path, image_type="image", frames_indices=None, timestamps=None):
        if image_type == "image":
            start_time = time.time()
            img_value_str = self.client.get(path)
            if self.debug:
                end_time = time.time()
                if end_time - start_time > self.oss_time_log_thr:
                    print(f"[Warning] OSS read one image {path} cost {end_time - start_time} seconds")
            img = pil_loader(img_value_str)
            return img

        elif image_type == "video":
            return read_qwen3_vl_video(
                path,
                frames_indices=frames_indices,
                timestamps=timestamps,
                client=self.client,
                debug=self.debug,
                oss_time_log_thr=self.oss_time_log_thr,
            )
        else:
            raise ValueError(f"Unsupported image type: {image_type}")
