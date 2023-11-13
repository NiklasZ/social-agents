from typing import Dict, Tuple

import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from utils.helpers import log_verbose


class ObservationRecorder:
    """
    This is a class that records observations from the environment. It is loosely based off of
    gymnasium/wrappers/monitoring/video_recorder.py, but doesn't have direct access to the environment.
    """

    def __init__(self,
                 video_folder: str,
                 video_prefix: str,
                 frame_shapes: Dict[str, Tuple[int, int]],
                 frames_per_sec: int,
                 verbose: bool = False):
        """
        @param video_folder: target folder to output videos to
        @param video_prefix: prefix for video file names
        @param frame_shapes: shape of each player's observation
        @param frames_per_sec: how many observations to render per second
        @param verbose: whether to print debug statements
        """
        self.video_folder = video_folder
        self.video_prefix = video_prefix
        self.frame_shapes = frame_shapes
        self.frames_per_sec = frames_per_sec
        self.recorded_frames = {p: [] for p in frame_shapes.keys()}
        self.verbose = verbose
        self.closed = False

    def capture_frame(self, observations: Dict[str, np.ndarray]):
        """
        @param observations: dictionary of observations from the environment
        """
        assert not self.closed, "Cannot capture frame from closed recorder"

        for p in observations.keys():
            assert observations[p].shape == self.frame_shapes[
                p], f"Observation shape mismatch for player {p}. Expected {self.frame_shapes[p]}, got {observations[p].shape}"
            log_verbose(f"Capturing video frame for player {p}", self.verbose)
            self.recorded_frames[p].append(observations[p])

    def close(self):
        """
        Closes recorder and saves videos to video_folder.
        @return:
        """

        assert not self.closed, "Cannot close already closed recorder"
        self.closed = True
        for player, frames in self.recorded_frames.items():
            if len(frames) > 0:
                clip = ImageSequenceClip(frames, fps=self.frames_per_sec)
                clip.write_videofile(f"{self.video_folder}/{self.video_prefix}_{player}_observation.mp4",
                                     logger=None if not self.verbose else 'bar')
