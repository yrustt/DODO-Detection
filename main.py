import argparse

from dodo_detection.detection.base import VideoDetector


def parse_arguments():
    """
    Парсинг аргументов командной строки
    """
    parser = argparse.ArgumentParser(description="Распознавание занятости стола")

    parser.add_argument(
        "--video", "-v", type=str, required=True, help="Путь к видеофайлу"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    detector = VideoDetector(video_name=args.video, need_visualize=False)
    detector.run()


if __name__ == "__main__":
    main()
