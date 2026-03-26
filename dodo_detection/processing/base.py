import time
from enum import Enum
from operator import itemgetter

import pandas as pd
from shapely import box

from dodo_detection.processing.math import are_same, close_each_other, is_walking


class ObjectTypes(Enum):
    PERSON = "person"
    TABLE = "dining table"


DARK_BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (0, 128, 255)


class Processor:
    def __init__(self):
        self._tables = {}
        self._previous_persons = []
        self._actions = pd.DataFrame(columns=["x1", "y1", "x2", "y2", "value", "time"])

    def run(self, extracted, extracted_walkings):
        processed, persons = [], []

        for data in extracted:
            # Проверяем, есть ли детекции
            if data.boxes is None:
                continue

            boxes = data.boxes.xyxy.cpu().numpy()  # координаты
            confs = data.boxes.conf.cpu().numpy()  # уверенность
            classes = data.boxes.cls.cpu().numpy()  # ID классов
            track_ids = data.boxes.id.cpu().numpy()

            # Отрисовываем каждую детекцию
            for box, conf, cls_id, track_id in zip(boxes, confs, classes, track_ids):
                x1, y1, x2, y2 = map(int, box[:4])

                class_name = data.names[int(cls_id)]

                # Выбираем цвет в зависимости от класса
                if class_name == ObjectTypes.PERSON.value:
                    persons.append(
                        dict(coords=((x1, y1), (x2, y2)), color=DARK_BLUE_COLOR, label=f"person", track=track_id)
                    )
                elif class_name == ObjectTypes.TABLE.value:
                    self.add_table(x1, y1, x2, y2)
                else:
                    continue  # пропускаем другие объекты

            self.filter_tables()
            self.find_walking(persons)

            processed.extend(self.process_tables(self.exclude_walking(persons)))
            processed.extend(persons)
            # processed.extend(extracted_walkings)

        return processed

    def find_walking(self, persons):
        for person in persons:
            person_square = box(*person["coords"][0], *person["coords"][1])

            for previous_person in self._previous_persons:
                if person["track"] is not None and person["track"] == previous_person["track"]:
                    previous_person_square = box(*previous_person["coords"][0], *previous_person["coords"][1])

                    if is_walking(person_square, previous_person_square):
                        person["color"] = BLUE_COLOR
                        person["is_walking"] = True

                        break
            else:
                person["is_walking"] = False

        self._previous_persons = persons

    def exclude_walking(self, persons):
        return [person for person in persons if not person["is_walking"]]

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

    def process_tables(self, persons):
        tables = [
            dict(coords=((key[0], key[1]), (key[2], key[3])), color=GREEN_COLOR, label="table")
            for key in self._tables.keys()
        ]

        for table in tables:
            table_square = box(*table["coords"][0], *table["coords"][1])
            table_actions = self._actions[
                (self._actions["x1"] == table["coords"][0][0])
                & (self._actions["y1"] == table["coords"][0][1])
                & (self._actions["x2"] == table["coords"][1][0])
                & (self._actions["y2"] == table["coords"][1][1])
            ]
            last_action = None
            if not table_actions.empty:
                last_action = table_actions.sort_values("time").iloc[-1]

            for person in persons:
                person_square = box(*person["coords"][0], *person["coords"][1])

                if close_each_other(person_square, table_square):
                    table["color"] = RED_COLOR

                    if last_action is None or last_action.value == 0:
                        new_row = pd.DataFrame([
                            {
                                "x1": table["coords"][0][0],
                                "y1": table["coords"][0][1],
                                "x2": table["coords"][1][0],
                                "y2": table["coords"][1][1],
                                "value": 1,
                                "time": time.time(),
                            }
                        ])
                        pd.concat([self._actions, new_row], ignore_index=True)
            else:
                if last_action is None or last_action.value == 1:
                    new_row = pd.DataFrame([
                        {
                            "x1": table["coords"][0][0],
                            "y1": table["coords"][0][1],
                            "x2": table["coords"][1][0],
                            "y2": table["coords"][1][1],
                            "value": 0,
                            "time": time.time(),
                        }
                    ])
                    pd.concat([self._actions, new_row], ignore_index=True)

        return tables
