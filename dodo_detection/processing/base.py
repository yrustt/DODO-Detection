from enum import Enum


class ObjectTypes(Enum):
    PERSON = "person"
    TABLE = "dining table"


BLUE_COLOR = (255, 0, 0)
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)


class Processor:
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
                    color = BLUE_COLOR
                    label = f"person {conf:.2f}"
                elif class_name == ObjectTypes.TABLE.value:
                    color = GREEN_COLOR
                    label = f"table {conf:.2f}"
                else:
                    continue  # пропускаем другие объекты

                processed.append(((x1, y1), (x2, y2), color, label))

        return processed
