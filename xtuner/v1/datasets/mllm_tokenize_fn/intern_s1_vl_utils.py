# Copyright (c) OpenMMLab. All rights reserved.
import io
import os
import random
import re
from typing import Literal

import cv2
import imageio
import numpy as np
from PIL import Image

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


def get_frame_indices(num_frames, vlen, sample="rand", fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:  # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == "rand":
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except Exception:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == "middle":
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[: len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if 0 < max_num_frames < len(frame_indices):
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def read_frames_folder(
    video_path,
    num_frames,
    sample="rand",
    fix_start=None,
    client=None,
    clip=None,
    min_num_frames=4,
    random_frame_num=None,
):
    if "s3://" in video_path:
        assert client is not None, "client should be provided for s3 backend"
        image_list = sort_frames(client.list(video_path))
        image_list = [os.path.join(video_path.split(image.split("/")[0])[0], image) for image in image_list]
        frames = []
        for image in image_list:
            frame = Image.open(io.BytesIO(client.get(image)))
            frames.append(frame)
    else:
        image_list = sort_frames(list(os.listdir(video_path)))
        frames = []
        for image in image_list:
            fp = os.path.join(video_path, image)
            frame = Image.open(fp).convert("RGB")
            frames.append(frame)
    vlen = len(frames)

    if random_frame_num is None:
        t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    else:
        t_num_frames = random_frame_num

    if vlen > t_num_frames:
        frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start)
        frames = [frames[i] for i in frame_indices]
    return frames


def read_frames_gif(
    video_path, num_frames, sample="rand", fix_start=None, client=None, min_num_frames=4, random_frame_num=None
):
    if "s3://" in video_path:
        assert client is not None, "client should be provided for s3 backend"
        video_bytes = client.get(video_path)
        gif = imageio.get_reader(io.BytesIO(video_bytes))
    else:
        gif = imageio.get_reader(video_path)
    vlen = len(gif)

    if random_frame_num is None:
        t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    else:
        t_num_frames = random_frame_num

    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    for index, frame in enumerate(gif):
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB).astype(np.uint8)
            frame = Image.fromarray(frame)
            frames.append(frame)
    return frames


def read_frames_decord(
    video_path,
    num_frames,
    sample="rand",
    fix_start=None,
    client=None,
    clip=None,
    min_num_frames=4,
    random_frame_num=None,
):
    if "s3://" in video_path:
        assert client is not None, "client should be provided for s3 backend"
        video_bytes = client.get(video_path)
        video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    # t_num_frames = min(max(int(duration * sample_fps), min_num_frames), num_frames)
    if random_frame_num is None:
        t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    else:
        t_num_frames = random_frame_num

    frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps)
    if clip:
        frame_indices = [f + start_index for f in frame_indices]
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames


def read_interns1_vl_video(
    path, min_num_frames, max_num_frames, random_frame_num, sample="rand", clip=None, client=None
):
    if path.endswith("/"):
        frames = read_frames_folder(
            path,
            num_frames=max_num_frames,
            min_num_frames=min_num_frames,
            client=client,
            sample=sample,
            random_frame_num=random_frame_num,
        )
    elif path.endswith(".gif"):
        frames = read_frames_gif(
            path,
            num_frames=max_num_frames,
            min_num_frames=min_num_frames,
            client=client,
            sample=sample,
            random_frame_num=random_frame_num,
        )
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
        frames = read_frames_decord(
            path,
            num_frames=max_num_frames,
            min_num_frames=min_num_frames,
            client=client,
            sample=sample,
            clip=clip,
            random_frame_num=random_frame_num,
        )
    else:
        raise ValueError(f"Unsupported video format: {path}")
    return frames


class InternS1VLOSSLoader:
    # Singleton instance
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, backend: Literal["petrel"] = "petrel", **kwargs):
        self.client = get_oss_backend(backend, **kwargs)

    def __call__(
        self,
        path,
        image_type="image",
        max_num_frames=-1,
        min_num_frames=8,
        sample="rand",
        clip=None,
        random_frame_num=None,
    ):
        if image_type == "image":
            img_value_str = self.client.get(path)
            img = pil_loader(img_value_str)
            return img

        elif image_type == "video":
            return read_interns1_vl_video(
                path,
                min_num_frames,
                max_num_frames,
                random_frame_num=random_frame_num,
                sample=sample,
                clip=clip,
                client=self.client,
            )
