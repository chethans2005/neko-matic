from __future__ import annotations

from typing import List

import pandas as pd

from backend.core.model_registry import ModelRegistry
from backend.meta_learning.dataset_difficulty import DatasetDifficultyAnalyzer


class ModelRecommender:
    """Recommend models using lightweight dataset heuristics."""

    def recommend(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        problem_type: str,
        top_k: int = 5,
    ) -> List[str]:
        models = ModelRegistry.list_models(problem_type)
        if not models:
            return []

        analyzer = DatasetDifficultyAnalyzer()
        meta = analyzer.analyze(dataframe, target_column)
        difficulty = float(meta.get("difficulty_score", 0.5))
        n_rows = int(meta.get("n_samples", len(dataframe)))

        priority: List[str] = []

        if problem_type == "classification":
            if difficulty >= 0.6:
                priority.extend(["XGBClassifier", "LGBMClassifier", "RandomForestClassifier"])
            elif n_rows < 1000:
                priority.extend(["LogisticRegression", "RandomForestClassifier", "SVC"])
            else:
                priority.extend(["RandomForestClassifier", "LGBMClassifier", "GradientBoostingClassifier"])
        else:
            if difficulty >= 0.6:
                priority.extend(["XGBRegressor", "LGBMRegressor", "RandomForestRegressor"])
            elif n_rows < 1000:
                priority.extend(["Ridge", "Lasso", "RandomForestRegressor"])
            else:
                priority.extend(["RandomForestRegressor", "LGBMRegressor", "GradientBoostingRegressor"])

        ordered = []
        seen = set()
        for name in priority + models:
            if name in models and name not in seen:
                ordered.append(name)
                seen.add(name)

        return ordered[:top_k]
