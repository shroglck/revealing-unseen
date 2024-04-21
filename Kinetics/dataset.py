from __future__ import annotations
import gc
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import torch.utils.data
from pytorchvideo.data.clip_sampling import ClipSampler
from .labeled_video_paths import LabeledVideoPaths
from .utils import MultiProcessSampler
logger = logging.getLogger(__name__)
from abc import ABC, abstractmethod
from typing import BinaryIO, Dict, Optional
import torch
from iopath.common.file_io import g_pathmgr
import numpy as np
import cv2 as cv
import random
import os
import xml.etree.ElementTree
import PIL.Image
import math
import json

import pytorchvideo
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)
import cv2 as cv





class VideoPathHandler(object):
    """
    Utility class that handles all deciphering and caching of video paths for
    encoded and frame videos.
    """

    def __init__(self) -> None:
        # Pathmanager isn't guaranteed to be in correct order,
        # sorting is expensive, so we cache paths in case of frame video and reuse.
        self.path_order_cache = {}
        

    def video_from_path(
        self, filepath, decode_video=True, decode_audio=False, decoder="pyav", fps=30
    ):
        try:
            is_file = g_pathmgr.isfile(filepath)
            is_dir = g_pathmgr.isdir(filepath)
        except NotImplementedError:

            # Not all PathManager handlers support is{file,dir} functions, when this is the
            # case, we default to assuming the path is a file.
            is_file = True
            is_dir = False

        if is_file:
            from pytorchvideo.data.encoded_video import EncodedVideo

            return EncodedVideo.from_path(
                filepath,
                #decode_video=decode_video,
                #decode_audio=decode_audio,
                decoder=decoder,
            )
        elif is_dir:
            from pytorchvideo.data.frame_video import FrameVideo

            assert not decode_audio, "decode_audio must be False when using FrameVideo"
            return FrameVideo.from_directory(
                filepath, fps, path_order_cache=self.path_order_cache
            )
        else:
            raise FileNotFoundError(f"{filepath} not found.")


class Video(ABC):
    """
    Video provides an interface to access clips from a video container.
    """

    @abstractmethod
    def __init__(
        self,
        file: BinaryIO,
        video_name: Optional[str] = None,
        decode_audio: bool = True,
    ) -> None:
        """
        Args:
            file (BinaryIO): a file-like object (e.g. io.BytesIO or io.StringIO) that
                contains the encoded video.
        """
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        """
        Returns:
            duration of the video in seconds
        """
        pass

    @abstractmethod
    def get_clip(
        self, start_sec: float, end_sec: float
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Retrieves frames from the internal video at the specified start and end times
        in seconds (the video always starts at 0 seconds).
        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            video_data_dictonary: A dictionary mapping strings to tensor of the clip's
                underlying data.
        """
        pass

    def close(self):
        pass

class LabeledVideoDataset(torch.utils.data.IterableDataset):
    """
    LabeledVideoDataset handles the storage, loading, decoding and clip sampling for a
    video dataset. It assumes each video is stored as either an encoded video
    (e.g. mp4, avi) or a frame video (e.g. a folder of jpg, or png)
    """

    _MAX_CONSECUTIVE_FAILURES = 10

    def __init__(
        self,
        labeled_video_paths: List[Tuple[str, Optional[dict]]],
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[dict], Any]] = None,
        decode_audio: bool = True,
        decode_video: bool = True,
        decoder: str = "pyav",
       
        train:bool = False
    ) -> None:
        """
        Args:
            labeled_video_paths (List[Tuple[str, Optional[dict]]]): List containing
                    video file paths and associated labels. If video paths are a folder
                    it's interpreted as a frame video, otherwise it must be an encoded
                    video.
            clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.
            video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.
            transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations on the clips. The clip output format is described in __next__().
            decode_audio (bool): If True, decode audio from video.
            decode_video (bool): If True, decode video frames from a video container.
            decoder (str): Defines what type of decoder used to decode a video. Not used for
                frame videos.
        """
        self._decode_audio = decode_audio
        self._decode_video = decode_video
        self._transform = transform
        self._clip_sampler = clip_sampler
        self._labeled_videos = labeled_video_paths
        #print(self._labeled_videos)
        
        self._decoder = decoder
        # If a RandomSampler is used we need to pass in a custom random generator that
        # ensures all PyTorch multiprocess workers have the same random seed.
        self._video_random_generator = None
        if video_sampler == torch.utils.data.RandomSampler:
            self._video_random_generator = torch.Generator()
            self._video_sampler = video_sampler(
                self._labeled_videos, generator=self._video_random_generator
            )
        else:
            self._video_sampler = video_sampler(self._labeled_videos)

        self._video_sampler_iter = None  # Initialized on first call to self.__next__()

        # Depending on the clip sampler type, we may want to sample multiple clips
        # from one video. In that case, we keep the store video, label and previous sampled
        # clip time in these variables.
        self._loaded_video_label = None
        self._loaded_clip = None
        self._last_clip_end_time = None
        self.video_path_handler = VideoPathHandler()
        self.train = train
        self.tr = Compose(
                  [
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    
                 ]
                )
        

    @property
    def video_sampler(self):
        """
        Returns:
            The video sampler that defines video sample order. Note that you'll need to
            use this property to set the epoch for a torch.utils.data.DistributedSampler.
        """
        return self._video_sampler

    @property
    def num_videos(self):
        """
        Returns:
            Number of videos in dataset.
        """
        return len(self.video_sampler)
    def vid_name(self,name):
        t=name.split("/")
        return t[-2],t[-1]
                
    def __next__(self) -> dict:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.
        Returns:
            A dictionary with the following format.
            .. code-block:: text
                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.
            if self._loaded_video_label:
                video, info_dict, video_index = self._loaded_video_label
            else:
                video_index = next(self._video_sampler_iter)
                try:
                    video_path, info_dict = self._labeled_videos[video_index]
                    video = self.video_path_handler.video_from_path(
                        video_path,
                        #decode_audio=self._decode_audio,
                        #decode_video=self._decode_video,
                        decoder=self._decoder,
                    )
                    self._loaded_video_label = (video, info_dict, video_index)
                except Exception as e:
                    logger.debug(
                        "Failed to load video with error: {}; trial {}".format(
                            e,
                            i_try,
                        )
                    )
                    logger.exception("Video load exception")
                    continue

            (
                clip_start,
                clip_end,
                clip_index,
                aug_index,
                is_last_clip,
            ) = self._clip_sampler(self._last_clip_end_time, video.duration, info_dict)

            if isinstance(clip_start, list):  # multi-clip in each sample

                # Only load the clips once and reuse previously stored clips if there are multiple
                # views for augmentations to perform on the same clips.
                if aug_index[0] == 0:
                    self._loaded_clip = {}
                    loaded_clip_list = []
                    for i in range(len(clip_start)):
                        clip_dict = video.get_clip(clip_start[i], clip_end[i])
                        if clip_dict is None or clip_dict["video"] is None:
                            self._loaded_clip = None
                            break
                        loaded_clip_list.append(clip_dict)

                    if self._loaded_clip is not None:
                        for key in loaded_clip_list[0].keys():
                            self._loaded_clip[key] = [x[key] for x in loaded_clip_list]

            else:  # single clip case

                # Only load the clip once and reuse previously stored clip if there are multiple
                # views for augmentations to perform on the same clip.
                if aug_index == 0:
                    self._loaded_clip = video.get_clip(clip_start, clip_end)

            self._last_clip_end_time = clip_end

            video_is_null = (
                self._loaded_clip is None or self._loaded_clip["video"] is None
            )
            if (
                is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip
            ) or video_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                self._loaded_video_label[0].close()
                self._loaded_video_label = None
                self._last_clip_end_time = None
                self._clip_sampler.reset()

                # Force garbage collection to release video container immediately
                # otherwise memory can spike.
                gc.collect()

                if video_is_null:
                    logger.debug(
                        "Failed to load clip {}; trial {}".format(video.name, i_try)
                    )
                    continue

            frames = self._loaded_clip["video"]
            #audio_samples = self._loaded_clip["audio"]
            sample_dict = {
                "video": frames,
                "video_name": video.name,
                "video_index": video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
                **info_dict,
                #**({"audio": audio_samples} if audio_samples is not None else {}),
            }
            #print(sample_dict["video"].shape)
            vid_name = sample_dict["video_name"]
            
            if self._transform is not None:
                
                sample_dict = self._transform(sample_dict)
                
                sample_dict["video"] = self.tr(sample_dict["video"])    
                if sample_dict is None:
                    continue

            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )

    def __iter__(self):
        self._video_sampler_iter = None  # Reset video sampler

        # If we're in a PyTorch DataLoader multiprocessing context, we need to use the
        # same seed for each worker's RandomSampler generator. The workers at each
        # __iter__ call are created from the unique value: worker_info.seed - worker_info.id,
        # which we can use for this seed.
        worker_info = torch.utils.data.get_worker_info()
        if self._video_random_generator is not None and worker_info is not None:
            base_seed = worker_info.seed - worker_info.id
            self._video_random_generator.manual_seed(base_seed)

        return self


def labeled_video_dataset(
    cl: int=None,
    data_path: str=None,
    clip_sampler: ClipSampler=None,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    video_path_prefix: str = "",
    decode_audio: bool = False,
    decoder: str = "pyav",
    
    train: bool = False
) -> LabeledVideoDataset:
    """
    A helper function to create ``LabeledVideoDataset`` object for Ucf101 and Kinetics datasets.
    Args:
        data_path (str): Path to the data. The path type defines how the data
            should be read:
            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).
        clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.
        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.
        transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.
        video_path_prefix (str): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. All the video paths before loading
                are prefixed with this path.
        decode_audio (bool): If True, also decode audio from video.
        decoder (str): Defines what type of decoder used to decode a video.
    """
    labeled_video_paths = LabeledVideoPaths.from_path(data = cl,data_path = data_path)
    labeled_video_paths.path_prefix = video_path_prefix
    dataset = LabeledVideoDataset(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
        train = train
    )
    return dataset
