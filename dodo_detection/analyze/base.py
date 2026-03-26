from statistics import mean
from typing import NoReturn

import pandas as pd


class Analyzer:
    def run(self, actions: pd.DataFrame, frame_count: int) -> NoReturn:
        actions = self.filter_actions(actions)

        self.report(actions, frame_count)
        self.write(actions, frame_count)

    def filter_actions(self, actions: pd.DataFrame) -> pd.DataFrame:
        """
        Удаляем группы быстрых смен статуса.
        """

        remove_idx = set()

        for idx, group in actions.groupby(["x1", "y1", "x2", "y2"]):
            group = group.sort_values("time")

            for curr, next in zip(group[:-1].itertuples(), group[1:].itertuples()):
                if next.time - curr.time < 10:
                    remove_idx.add(curr.Index)
                    remove_idx.add(next.Index)

        return actions.drop(list(remove_idx))

    def report(self, actions: pd.DataFrame, frame_count) -> NoReturn:
        times = []

        for idx, group in actions.groupby(["x1", "y1", "x2", "y2"]):
            group = group.sort_values("time")
            start_time = None

            for row in group.itertuples():
                if row.value == 0 and start_time is None:
                    start_time = row.time

                if row.value == 1:
                    if start_time is not None:
                        times.append(row.time - start_time)

                    start_time = None

        print("Среднее время простоя: ", mean(times))

    def write(self, actions: pd.DataFrame, frame_count: int) -> NoReturn:
        actions.to_csv("output.csv")
