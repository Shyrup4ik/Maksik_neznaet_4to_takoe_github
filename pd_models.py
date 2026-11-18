"""Классы моделей (вынесены в модуль, чтобы обученные модели открывались из интерфейса)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from features import FIN_FEATS, MKT_FEATS, RAT_FEATS, ALTMAN_X
from mars import MARSClassifier


class Prep:
    """Отсечение выбросов по квантилям обучающей выборки + заполнение пропусков медианой
    + флаги пропуска групп признаков (нет отчётности / нет котировок / нет рейтинга)."""

    def __init__(self, feats):
        self.feats = list(feats)

    def fit(self, df):
        X = df[self.feats].astype(float)
        self.lo, self.hi = X.quantile(0.01), X.quantile(0.99)
        Xc = X.clip(self.lo, self.hi, axis=1)
        self.med = Xc.median().fillna(0)
        self.groups = {g: [f for f in fs if f in self.feats] for g, fs in
                       (("miss_fin", FIN_FEATS), ("miss_mkt", MKT_FEATS), ("miss_rating", RAT_FEATS))}
        self.groups = {g: fs for g, fs in self.groups.items() if fs and X[fs].isna().all(axis=1).any()}
        self.names = self.feats + list(self.groups)
        return self

    def transform(self, df):
        X = df.reindex(columns=self.feats).astype(float).clip(self.lo, self.hi, axis=1)
        out = X.fillna(self.med)
        for g, fs in self.groups.items():
            out[g] = X[fs].isna().all(axis=1).astype(float)
        return out[self.names].to_numpy()


class AltmanFixed:
    name = "Altman Z'' (классика)"

    def fit(self, df):
        z = df[["z_altman"]].to_numpy()
        self.lr = LogisticRegression(C=1e6, max_iter=2000).fit(z, df.y.astype(int))
        return self

    def predict(self, df):
        z = df["z_altman"].astype(float)
        p = pd.Series(np.nan, index=df.index)
        ok = z.notna()
        if ok.any():
            p[ok] = self.lr.predict_proba(z[ok].to_numpy()[:, None])[:, 1]
        return p.to_numpy()


class AltmanLogit:
    name = "Altman-логит (переоценка весов)"

    def fit(self, df):
        self.prep = Prep(ALTMAN_X).fit(df)
        self.prep.groups = {}
        self.prep.names = list(ALTMAN_X)
        self.lr = LogisticRegression(C=1.0, max_iter=5000).fit(self.prep.transform(df), df.y.astype(int))
        return self

    def predict(self, df):
        p = self.lr.predict_proba(self.prep.transform(df))[:, 1]
        p[df[ALTMAN_X].isna().any(axis=1).to_numpy()] = np.nan
        return p


class MarsModel:
    def __init__(self, name, feats, need_fin=False, **kw):
        self.name, self.feats, self.need_fin, self.kw = name, feats, need_fin, kw

    def fit(self, df):
        self.prep = Prep(self.feats).fit(df)
        self.m = MARSClassifier(**self.kw).fit(self.prep.transform(df), df.y.to_numpy(), self.prep.names)
        return self

    def predict(self, df):
        p = self.m.predict_proba(self.prep.transform(df))[:, 1]
        if self.need_fin:
            p[df["current_ratio"].isna().to_numpy() & df["equity_ratio"].isna().to_numpy()] = np.nan
        return p
