import cv2
from ultralytics import YOLO

from dodo_detection.detection.capture import FrameIterator
from dodo_detection.processing.base import Processor, BLUE_COLOR


class VideoDetector:
    YOLO_MODEL = "yolo26x.pt"
    PROCESSING_CLASS = Processor

    def __init__(
        self,
        video_name: str | int,
        need_visualize: bool = True,
        need_write_video: bool = True,
    ):
        self.video_name = video_name
        self.need_visualize = need_visualize
        self.need_write_video = need_write_video

    def run(self):
        self.init()

        with FrameIterator(self.video_name) as frame_iterator:
            for idx, frame in enumerate(frame_iterator):
                # extracted_walkings = self.detect_walking(frame)
                extracted = self.detect(frame)

                processed = self.processor.run(idx, extracted)

                self.visualize(frame, processed)

    def detect(self, frame):
        """
        Обнаруживаем объекты людей и столов
        """

        extracted = self.model.track(
            frame, conf=0.15, persist=True, stream=False, classes=[0, 60]
        )

        return extracted

    def detect_walking(self, frame):
        """
        Способ обнаруживать движения. Не получилось различить ходьбу и движение руками.
        """

        walkings = []

        fgmask = self.fgbg.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 3000:
                x, y, w, h = cv2.boundingRect(contour)

                walkings.append(
                    dict(
                        coords=((x, y), (x + w, y + h)),
                        color=BLUE_COLOR,
                        label="Walking",
                    )
                )

        return walkings

    def visualize(self, frame, processed):
        """
        Отрисовка результата.
        """

        for object_data in processed:
            (x1, y1), (x2, y2) = object_data["coords"]
            color, label = object_data["color"], object_data["label"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        cv2.imshow("Detection", frame)
        cv2.waitKey(1)

    def init(self):
        """
        Ленивая инициализация
        """

        self.model = YOLO(self.YOLO_MODEL)
        self.processor = self.PROCESSING_CLASS()

        cv2.namedWindow("Detection", cv2.WINDOW_GUI_NORMAL)

        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=36, detectShadows=True
        )
