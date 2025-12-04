# Copyright (c) OpenMMLab. All rights reserved.
import io
import os
import random
import re
import time
from pathlib import Path
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
            except KeyboardInterrupt as e:
                raise e
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
    video_extra_dict=None,
):
    oss_read_time = 0
    if video_extra_dict is not None and "processed_video_length" in video_extra_dict:
        processed_video_length = video_extra_dict["processed_video_length"]
        image_list = [f"{i:08d}.jpg" for i in range(1, processed_video_length + 1, 1)]
        image_list = [os.path.join(video_path, img) for img in image_list]

        if "s3://" not in video_path:
            for image in image_list:
                if not os.path.exists(image):
                    image_list = sort_frames(list(os.listdir(video_path)))
                    break
    else:
        if "s3://" in video_path:
            assert client is not None, "client should be provided for s3 backend"
            image_list = sort_frames(client.list(video_path))
            image_list = [os.path.join(video_path.split(image.split("/")[0])[0], image) for image in image_list]
        else:
            image_list = sort_frames(list(os.listdir(video_path)))
    vlen = len(image_list)

    if random_frame_num is None:
        t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    else:
        t_num_frames = random_frame_num

    if vlen > t_num_frames:
        frame_indices = get_frame_indices(t_num_frames, vlen, sample=sample, fix_start=fix_start)
    else:
        frame_indices = list(range(vlen))

    frame_list = []
    for frame_index in frame_indices:
        if "s3://" in video_path:
            start_time = time.time()
            image_byte = client.get(image_list[frame_index])
            oss_read_time += time.time() - start_time
            frame = Image.open(io.BytesIO(image_byte))
            frame_list.append(frame)
        else:
            fp = os.path.join(video_path, image_list[frame_index])
            frame = Image.open(fp).convert("RGB")
            frame_list.append(frame)

    return frame_list, oss_read_time, len(frame_list)


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
    video_get_batch_time = time.time() - start_time
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames, oss_read_time, video_get_batch_time, vlen


def read_interns1_vl_video(
    path,
    min_num_frames,
    max_num_frames,
    random_frame_num,
    sample="rand",
    clip=None,
    client=None,
    debug=False,
    oss_time_log_thr=10,
    video_extra_dict=None,
):
    start_time = time.time()
    oss_read_time = 0
    vlen = 0
    video_get_batch_time = 0
    if path.endswith("/") or Path(path).is_dir():
        frames, oss_read_time, vlen = read_frames_folder(
            path,
            num_frames=max_num_frames,
            min_num_frames=min_num_frames,
            client=client,
            sample=sample,
            random_frame_num=random_frame_num,
            video_extra_dict=video_extra_dict,
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
        frames, oss_read_time, video_get_batch_time, vlen = read_frames_decord(
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
    end_time = time.time() - start_time
    if debug and end_time > oss_time_log_thr:
        print(
            f"[Warning] OSS read video {path} cost {end_time} seconds, "
            f"oss_read_time {oss_read_time}, video_get_batch_time {video_get_batch_time}, vlen {vlen}"
        )
    return frames


class InternS1VLOSSLoader:
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

    def __call__(
        self,
        path,
        image_type="image",
        max_num_frames=-1,
        min_num_frames=8,
        sample="rand",
        clip=None,
        random_frame_num=None,
        video_extra_dict=None,
    ):
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
            return read_interns1_vl_video(
                path,
                min_num_frames,
                max_num_frames,
                random_frame_num=random_frame_num,
                sample=sample,
                clip=clip,
                client=self.client,
                debug=self.debug,
                oss_time_log_thr=self.oss_time_log_thr,
                video_extra_dict=video_extra_dict,
            )
