from functools import cached_property

import cv2


class FrameIterator:
    """
    Класс-генератор, позволяющий пройтись по всем фрэймам видео.
    """

    def __init__(self, video_name):
        self.video_name = video_name

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_name)
        _ = self.frame_count
        _ = self.output

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "cap"):
            self.cap.release()

        return False

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()

        if not ret:
            raise StopIteration

        return frame

    @cached_property
    def frame_count(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    @cached_property
    def output(self):
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

        return out
