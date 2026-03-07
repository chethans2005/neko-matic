"""
Leaderboard
============
Builds and maintains a ranked DataFrame of training results.

Models are sorted by the primary metric; ties are broken by training time.
The best model sits at rank 1 (DataFrame index 1).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from flexautoml.core.trainer import TrainingResult


class Leaderboard:
    """
    Converts a list of :class:`~flexautoml.core.trainer.TrainingResult`
    objects into a sortable leaderboard DataFrame.

    Parameters
    ----------
    problem_type : str
        ``"classification"`` or ``"regression"``.
    primary_metric : str
        The metric column used to rank models.
    """

    def __init__(
        self,
        problem_type: str,
        primary_metric: str = "accuracy",
    ) -> None:
        self.problem_type = problem_type
        self.primary_metric = primary_metric
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, results: List[TrainingResult]) -> pd.DataFrame:
        """
        Builds the leaderboard DataFrame from training results.

        Rows are sorted so that the best model appears first (rank 1).
        For regression metrics ``rmse`` / ``mae`` / ``mse`` lower is better;
        for all other metrics higher is better.

        Parameters
        ----------
        results : list[TrainingResult]

        Returns
        -------
        pd.DataFrame
            Columns: model, training_time_s, <metric columns…>
        """
        rows: List[Dict[str, Any]] = []
        for r in results:
            row: Dict[str, Any] = {
                "model": r.model_name,
                "training_time_s": round(r.training_time, 2),
            }
            for k, v in r.metrics.items():
                row[k] = round(v, 4) if isinstance(v, float) else v
            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by primary metric if present
        if self.primary_metric in df.columns:
            lower_better = (
                self.problem_type == "regression"
                and self.primary_metric in ("rmse", "mae", "mse")
            )
            df = df.sort_values(
                [self.primary_metric, "training_time_s"],
                ascending=[lower_better, True],
            ).reset_index(drop=True)
            df.index += 1  # Rank starts at 1

        self._df = df
        return df

    def get_best_model_name(self) -> Optional[str]:
        """Returns the model name at rank 1, or ``None`` if empty."""
        if self._df is not None and not self._df.empty:
            return str(self._df.iloc[0]["model"])
        return None

    def to_dict(self) -> List[Dict[str, Any]]:
        """Returns the leaderboard as a list of row dicts."""
        if self._df is not None:
            return self._df.to_dict("records")
        return []

    def to_csv(self, path: str) -> None:
        """Saves the leaderboard to a CSV file."""
        if self._df is not None:
            self._df.to_csv(path, index=True)

    def display(self) -> None:
        """Pretty-prints the leaderboard to stdout."""
        if self._df is not None and not self._df.empty:
            print(self._df.to_string())
        else:
            print("Leaderboard is empty. Call build() first.")

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """The raw leaderboard DataFrame (``None`` if not yet built)."""
        return self._df
