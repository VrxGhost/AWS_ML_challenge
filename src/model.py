"""Gradient-boosted tree wrapper: LightGBM (MIT) when installed, otherwise
scikit-learn's HistGradientBoostingClassifier (BSD). Both handle NaN natively."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:  # pragma: no cover
    HAVE_LGB = False


class GBM:
    def __init__(self, seed: int = 42, n_jobs: int = 4):
        self.seed, self.n_jobs = seed, n_jobs
        self.model = None
        self.cols = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, Xv: pd.DataFrame = None, yv: np.ndarray = None):
        self.cols = list(X.columns)
        if HAVE_LGB:
            self.model = lgb.LGBMClassifier(
                n_estimators=3000, learning_rate=0.04, num_leaves=127, min_child_samples=40,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=self.seed, n_jobs=self.n_jobs, verbose=-1)
            if Xv is not None:
                self.model.fit(X, y, eval_set=[(Xv[self.cols], yv)],
                               callbacks=[lgb.early_stopping(100, verbose=False)])
            else:
                self.model.set_params(n_estimators=800)
                self.model.fit(X, y)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.model = HistGradientBoostingClassifier(
                learning_rate=0.06, max_iter=600, max_leaf_nodes=63, min_samples_leaf=40,
                l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=40, random_state=self.seed)
            self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.cols])[:, 1].astype(np.float32)

    def importance(self) -> pd.Series:
        if HAVE_LGB:
            return pd.Series(self.model.booster_.feature_importance("gain"), index=self.cols).sort_values(ascending=False)
        return pd.Series(dtype=float)
