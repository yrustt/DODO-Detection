import time
from enum import Enum
from operator import itemgetter
from typing import NoReturn

import pandas as pd
from shapely import box

from dodo_detection.processing.math import are_same, close_each_other, is_walking


class ObjectTypes(Enum):
    PERSON = "person"
    TABLE = "dining table"


DARK_BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (230, 216, 173)


class Processor:
    def __init__(self):
        self._tables = {}
        self._previous_persons = []
        self._actions = pd.DataFrame(columns=["x1", "y1", "x2", "y2", "value", "time"])

    def run(self, frame_id, extracted):
        self._frame_id = frame_id

        processed, persons = [], []

        for data in extracted:
            if data.boxes is None:
                continue

            boxes = data.boxes.xyxy.cpu().numpy()
            confs = data.boxes.conf.cpu().numpy()
            classes = data.boxes.cls.cpu().numpy()
            track_ids = data.boxes.id.cpu().numpy()

            for box_, conf, cls_id, track_id in zip(boxes, confs, classes, track_ids):
                x1, y1, x2, y2 = map(int, box_[:4])

                class_name = data.names[int(cls_id)]

                if class_name == ObjectTypes.PERSON.value:
                    persons.append(
                        dict(
                            coords=((x1, y1), (x2, y2)),
                            color=DARK_BLUE_COLOR,
                            label="person",
                            track=track_id,
                        )
                    )
                elif class_name == ObjectTypes.TABLE.value:
                    self.add_table(x1, y1, x2, y2)
                else:
                    continue

            self.filter_tables()
            self.find_walking(persons)

            processed.extend(self.process_tables(self.exclude_walking(persons)))
            processed.extend(persons)

        return processed

    def find_walking(self, persons: list[dict]) -> NoReturn:
        """
        Находим движущихся людей.

        Если в текущем frame'е объекты переместились относительно своих предыдущих версий
        (смотрим по идентификатору track), то значит что человек идёт.

        Это нужно, чтобы игнорировать идущих людей, когда они проходят рядом со столом.

        Функция устанавливает is_walking = true/false.

        :param persons: список людей в текущем фрэйме.
        :return: NoReturn
        """

        for person in persons:
            person_square = box(*person["coords"][0], *person["coords"][1])

            for previous_person in self._previous_persons:
                if (
                    person["track"] is not None
                    and person["track"] == previous_person["track"]
                ):
                    previous_person_square = box(
                        *previous_person["coords"][0], *previous_person["coords"][1]
                    )

                    if is_walking(person_square, previous_person_square):
                        person["color"] = BLUE_COLOR
                        person["is_walking"] = True

                        break
            else:
                person["is_walking"] = False

        self._previous_persons = persons

    def exclude_walking(self, persons: list[dict]) -> list[dict]:
        """
        Исключаем движущихся людей.

        :param persons: список людей в текущем фрэйме
        :return: отфильтрованный список людей
        """

        return [person for person in persons if not person["is_walking"]]

    def add_table(self, x1: int, y1: int, x2: int, y2: int) -> NoReturn:
        """
        Добавляем стол в список обнаруженных таблиц.

        Нужно для более точной работы, потому что бывают моменты, когда стол на фрэйме не обнаруживается.
        """

        square = box(x1, y1, x2, y2)

        self._tables[(x1, y1, x2, y2)] = square.area

    def filter_tables(self) -> NoReturn:
        """
        Отфильтровываем дубликаты столов, оставляя наименьший по площади вариант.
        """

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
        """
        Обнаруживаем занятые столы. Если человек находится близко к столу, то считаем, что стол занят.
        """

        tables = [
            dict(
                coords=((key[0], key[1]), (key[2], key[3])),
                color=GREEN_COLOR,
                label="table",
            )
            for key in self._tables.keys()
        ]

        for table in tables:
            table_square = box(*table["coords"][0], *table["coords"][1])

            # Берём последнее действие со столом. Если value = 1, то стол занят, если 0, то свободен.
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

                    # Если стол был свободен, а сейчас занят, то записываем время этого события.
                    if last_action is None or last_action.value == 0:
                        new_row = pd.DataFrame(
                            [
                                {
                                    "x1": table["coords"][0][0],
                                    "y1": table["coords"][0][1],
                                    "x2": table["coords"][1][0],
                                    "y2": table["coords"][1][1],
                                    "value": 1,
                                    "time": self._frame_id,
                                }
                            ]
                        )
                        self._actions = pd.concat([self._actions, new_row], ignore_index=True)

                    break
            else:
                # Если стол был занят, а сейчас свободен, то записываем время этого события.
                if last_action is None or last_action.value == 1:
                    new_row = pd.DataFrame(
                        [
                            {
                                "x1": table["coords"][0][0],
                                "y1": table["coords"][0][1],
                                "x2": table["coords"][1][0],
                                "y2": table["coords"][1][1],
                                "value": 0,
                                "time": self._frame_id,
                            }
                        ]
                    )
                    self._actions = pd.concat([self._actions, new_row], ignore_index=True)

        return tables

    @property
    def actions(self):
        return self._actions
