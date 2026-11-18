"""ТОЛЬКО ДЛЯ ТЕСТА КОДА: генерирует синтетические сырые файлы в формате сборщиков,
чтобы проверить сборку базы, обучение и интерфейс без доступа к MOEX/ФНС.
Цифры выдуманные, выводы по ним делать нельзя. Запуск: PD_DATA_DIR=<папка> python tests/make_fake_raw.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA, RAW, ROOT  # noqa: E402

rng = np.random.default_rng(1)
src = ROOT / "data"
for f in ("universe.csv", "defaults_registry.csv"):
    (DATA / f).write_bytes((src / f).read_bytes())
(RAW / "history").mkdir(parents=True, exist_ok=True)

uni = pd.read_csv(DATA / "universe.csv")
reg = pd.read_csv(DATA / "defaults_registry.csv", parse_dates=["event_date"])
d = reg.groupby("company").agg(sector=("sector", "first")).reset_index()
d = d[~d.company.isin(uni.company)]
d["segment"], d["mandatory"] = "Дефолт-реестр", 0
em = pd.concat([uni, d], ignore_index=True)
em["status"], em["emitter_id"], em["inn"] = "ok", range(1000, 1000 + len(em)), [str(7700000000 + i) for i in range(len(em))]
em.to_csv(RAW / "emitters.csv", index=False)

first_ev = reg.groupby("company").event_date.min()
days = pd.bdate_range("2019-01-01", "2026-11-06")
bonds, fins = [], []
for e in em.itertuples():
    base = {"IG": 1.5, "HY": 0.3, "ВДО": -0.3}.get(e.segment, -0.5) + rng.normal(0, 0.5)
    fe = first_ev.get(e.company, pd.NaT)

    def health(t):
        h = base
        if pd.notna(fe):
            yrs = (fe - t).days / 365
            h -= 2.5 * np.exp(-max(yrs, 0) / 0.8) if yrs > -1 else 2.5
        return h + rng.normal(0, 0.3)

    for y in range(2018, 2026):
        h = health(pd.Timestamp(f"{y}-12-31"))
        ta = np.exp(rng.normal(16 if e.segment == "IG" else 13, 1))
        eqr = np.clip(0.25 + 0.1 * h + rng.normal(0, 0.05), -0.3, 0.8)
        ca = ta * rng.uniform(0.3, 0.7)
        cl = ca / np.clip(1.1 + 0.3 * h + rng.normal(0, 0.1), 0.3, 4)
        debt = ta * np.clip(0.45 - 0.08 * h, 0.05, 0.9)
        rev = ta * rng.uniform(0.3, 1.5)
        ebit = ta * (0.06 + 0.03 * h + rng.normal(0, 0.02))
        intr = debt * 0.12
        fins.append({"company": e.company, "inn": e.inn, "year": y, "c1600": ta, "c1200": ca, "c1210": ca * 0.3,
                     "c1250": ca * 0.1, "c1240": 0, "c1300": ta * eqr, "c1370": ta * (eqr - 0.05),
                     "c1400": debt * 0.6, "c1410": debt * 0.6, "c1500": cl, "c1510": debt * 0.4,
                     "c2110": rev, "c2200": ebit, "c2300": ebit - intr, "c2330": intr,
                     "c2400": (ebit - intr) * 0.8, "source": "FAKE"})
    for k in range(rng.integers(1, 4)):
        secid = f"FAKE{e.emitter_id}{k}"
        iss = pd.Timestamp("2019-01-01") + pd.Timedelta(days=int(rng.integers(0, 1500)))
        mat = iss + pd.Timedelta(days=int(rng.integers(700, 2500)))
        bonds.append({"secid": secid, "isin": f"RU000{secid}", "shortname": f"{e.company[:10]} {k + 1}",
                      "issuedate": iss.date(), "matdate": mat.date(), "facevalue": 1000,
                      "coupon_percent": 12 + 5 * (e.segment != "IG"), "company": e.company,
                      "emitter_id": e.emitter_id})
        dd = days[(days >= iss) & (days <= mat)]
        if pd.notna(fe):
            dd = dd[dd <= fe + pd.Timedelta(days=60)]
        hs = np.array([health(t) for t in dd[::20]])
        hs = np.interp(np.arange(len(dd)), np.arange(0, len(dd), 20)[:len(hs)], hs) if len(hs) else np.array([])
        price = np.clip(100 - 25 * np.exp(-hs) + rng.normal(0, 0.5, len(dd)), 15, 105)
        yld = np.clip(14 + 60 * np.exp(-1.5 * hs), 5, 400)
        pd.DataFrame({"tradedate": dd, "price": price.round(2), "yield": yld.round(2),
                      "value": rng.lognormal(14, 1, len(dd)).round(0)}).to_csv(RAW / "history" / f"{secid}.csv", index=False)
pd.DataFrame(bonds).to_csv(RAW / "bonds.csv", index=False)
pd.DataFrame(fins).to_csv(RAW / "financials.csv", index=False)
print("fake raw ok:", len(em), "компаний,", len(bonds), "облигаций")
