from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


class LeaderboardManager:
    """Stores and ranks model results for a single run."""

    def __init__(self, primary_metric: str, problem_type: str) -> None:
        self.primary_metric = primary_metric
        self.problem_type = problem_type
        self._rows: List[Dict[str, Any]] = []

    def add(self, model_name: str, metrics: Dict[str, Any], training_time: float) -> None:
        row = {
            "model": model_name,
            "training_time": round(training_time, 2),
        }
        row.update(metrics)
        self._rows.append(row)

    def dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=["model", "training_time"])

        df = pd.DataFrame(self._rows)
        metric = self.primary_metric
        if metric in df.columns:
            lower_is_better = metric in {"rmse", "mae", "mse"} and self.problem_type == "regression"
            df = df.sort_values(
                by=[metric, "training_time"],
                ascending=[lower_is_better, True],
            )
        return df.reset_index(drop=True)

    def as_records(self) -> List[Dict[str, Any]]:
        return self.dataframe().to_dict(orient="records")
