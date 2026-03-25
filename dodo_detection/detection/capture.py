import cv2


class FrameIterator:
    def __init__(self, video_name):
        self.video_name = video_name

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_name)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, 'cap'):
            self.cap.release()

        return False

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()

        if not ret:
            raise StopIteration

        return frame
