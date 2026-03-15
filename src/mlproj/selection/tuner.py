from __future__ import annotations

from dataclasses import dataclass

from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold


@dataclass
class Tuner:
    random_state: int = 42

    def tune(self, estimator, X, y, config: dict):
        method = config.get("method", "grid")
        param_grid = config.get("param_grid", {})
        n_iter = int(config.get("n_iter", 20))
        task = config.get("task", "classification")
        cv = self._build_cv(task, config.get("cv_folds", 5))
        scoring = config.get("scoring")

        if task in {"classification", "regression"} and y is None:
            raise ValueError(f"task '{task}' tuning requires y labels")

        if task in {"clustering", "pca_reduction", "anomaly_detection", "topic_modeling"}:
            if scoring is None and not hasattr(estimator, "score"):
                raise ValueError(
                    f"task '{task}' tuning requires scoring or estimator.score support"
                )

        if method == "grid":
            search = GridSearchCV(estimator, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        elif method == "random":
            search = RandomizedSearchCV(
                estimator,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=self.random_state,
            )
        elif method == "halving_grid":
            from sklearn.experimental import enable_halving_search_cv  # noqa: F401
            from sklearn.model_selection import HalvingGridSearchCV

            search = HalvingGridSearchCV(
                estimator,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )
        elif method == "halving_random":
            from sklearn.experimental import enable_halving_search_cv  # noqa: F401
            from sklearn.model_selection import HalvingRandomSearchCV

            search = HalvingRandomSearchCV(
                estimator,
                param_distributions=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unsupported tune method: {method}")

        if y is None:
            search.fit(X)
        else:
            search.fit(X, y)
        return search

    def _build_cv(self, task: str, folds: int):
        if task == "classification":
            return StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        return KFold(n_splits=folds, shuffle=True, random_state=self.random_state)
