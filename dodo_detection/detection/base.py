import cv2
from ultralytics import YOLO

from dodo_detection.detection.capture import FrameIterator
from dodo_detection.processing.base import Processor


class VideoDetector:
    YOLO_MODEL = "yolo26x.pt"
    PROCESSING_CLASS = Processor

    def __init__(self, video_name: str | int):
        self.video_name = video_name

    def run(self):
        self.init()

        with FrameIterator(self.video_name) as frame_iterator:
            for frame in frame_iterator:
                extracted = self.detect(frame)
                processed = self.processor.run(extracted)
                self.visualize(frame, processed)

    def detect(self, frame):
        extracted = self.model(frame, conf=0.15, stream=False)

        return extracted

    def visualize(self, frame, processed):
        for object_data in processed:
            (x1, y1), (x2, y2) = object_data['coords']
            color, label = object_data['color'], object_data['label']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Detection', frame)
        cv2.waitKey(1)

    def init(self):
        self.model = YOLO(self.YOLO_MODEL)
        self.processor = self.PROCESSING_CLASS()
        cv2.namedWindow('Detection', cv2.WINDOW_GUI_NORMAL)
