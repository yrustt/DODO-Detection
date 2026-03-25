from enum import Enum
from operator import itemgetter

from shapely import box

from dodo_detection.processing.math import are_same


class ObjectTypes(Enum):
    PERSON = "person"
    TABLE = "dining table"


BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)


class Processor:
    def __init__(self):
        self._tables = {}

    def run(self, extracted):
        processed = []

        for object_data in extracted:
            # Проверяем, есть ли детекции
            if object_data.boxes is None:
                continue

            boxes = object_data.boxes.xyxy.cpu().numpy()  # координаты
            confs = object_data.boxes.conf.cpu().numpy()  # уверенность
            classes = object_data.boxes.cls.cpu().numpy()  # ID классов

            # Отрисовываем каждую детекцию
            for box, conf, cls_id in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box[:4])

                class_name = object_data.names[int(cls_id)]

                # Выбираем цвет в зависимости от класса
                if class_name == ObjectTypes.PERSON.value:
                    processed.append(dict(coords=((x1, y1), (x2, y2)), color=BLUE_COLOR, label="person"))
                elif class_name == ObjectTypes.TABLE.value:
                    self.add_table(x1, y1, x2, y2)
                else:
                    continue  # пропускаем другие объекты

            self.filter_tables()

            processed.extend(
                dict(coords=((key[0], key[1]), (key[2], key[3])), color=GREEN_COLOR, label="table")
                for key in self._tables.keys()
            )

        return processed

    def add_table(self, x1, y1, x2, y2):
        square = box(x1, y1, x2, y2)

        self._tables[(x1, y1, x2, y2)] = square.area

    def filter_tables(self):
        tables = list(sorted(self._tables.items(), key=itemgetter(1)))
        length = len(tables)
        keep = [True] * length

        for i in range(length):
            square = box(*tables[i][0])

            for j in range(i + 1, length):
                other_square = box(*tables[j][0])

                if are_same(square, other_square):
                    keep[j] = False

        for i, (coords, _) in enumerate(tables):
            if not keep[i]:
                self._tables.pop(coords)
