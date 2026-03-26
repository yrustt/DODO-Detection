from typing import NoReturn

import pandas as pd


class Reporter:
    def write(self, actions: pd.DataFrame, frame_count: int) -> NoReturn:
        actions.to_csv("output.csv")
