from dodo_detection.detection.base import VideoDetector

detector = VideoDetector(
    "/home/user/Рабочий стол/Projects/DODO-Detection/input/input.mp4",
    need_visualize=False,
)
detector.run()
