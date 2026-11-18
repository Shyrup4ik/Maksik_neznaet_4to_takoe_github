"""Шаг 4. Обучение и сравнение моделей вероятности дефолта (PD на 6 месяцев).

Модели:
  1. Altman Z'' (классика)      — Z'' с исходными коэффициентами Альтмана (1995),
                                   переведённый в вероятность логит-калибровкой  PD = σ(a + b·Z'').
  2. Altman-логит (переоценка)   — те же 4 коэффициента X1..X4, но веса оценены на российских данных.
  3. MARS (отчётность)           — MARS-логит на коэффициентах РСБУ и их годовой динамике.
  4. MARS (полная)               — + котировки облигаций, рейтинги, история техдефолтов.

Проверка out-of-time: обучаем на срезах до TRAIN_END, тестируем на более поздних;
компании из FORCE_TEST_COMPANIES (Самолёт) в обучение не попадают вообще.

Выход: models/models.pkl, models/metrics.csv, models/predictions.csv, models/roc.json, models/report.txt
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, average_precision_score

from config import DATA, MODELS, TRAIN_END, TEST_START, FORCE_TEST_COMPANIES, TARGET_MODE, HORIZON_DAYS
from features import FIN_FEATS, DYN_FEATS, MKT_FEATS, RAT_FEATS, HIST_FEATS, ALTMAN_X
from pd_models import AltmanFixed, AltmanLogit, MarsModel


def ks_stat(y, p):
    f, t, _ = roc_curve(y, p)
    return float(np.max(t - f))


def metrics(y, p):
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return {"n": int(len(y)), "дефолтов": int(y.sum())}
    auc = roc_auc_score(y, p)
    return {"n": int(len(y)), "дефолтов": int(y.sum()), "ROC AUC": auc, "Gini": 2 * auc - 1,
            "KS": ks_stat(y, p), "PR AUC": average_precision_score(y, p), "Brier": brier_score_loss(y, p)}


def boot_auc_diff(y, p1, p2, n=1000, seed=0):
    ok = ~np.isnan(p1) & ~np.isnan(p2)
    y, p1, p2 = y[ok], p1[ok], p2[ok]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    d = np.array(d)
    return float(np.mean(d)), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), float((d <= 0).mean())


def main():
    df = pd.read_csv(DATA / "dataset.csv", parse_dates=["date", "next_event", "fin_period_end"])
    forced = df.company.isin(FORCE_TEST_COMPANIES)
    lab = df.label_known == 1
    df["split"] = "score"
    df.loc[lab & (df.date <= TRAIN_END) & ~forced, "split"] = "train"
    df.loc[lab & ((df.date >= TEST_START) | forced), "split"] = "test"
    tr, te = df[df.split == "train"], df[df.split == "test"]
    print(f"Обучение: {len(tr)} строк, {int(tr.y.sum())} дефолтов | Тест: {len(te)} строк, {int(te.y.sum())} дефолтов")
    if tr.y.sum() < 5:
        print("ВНИМАНИЕ: в обучении меньше 5 дефолтов — сдвиньте TRAIN_END в config.py позже "
              "или расширьте реестр дефолтов.")

    tr_fin = tr[tr.z_altman.notna()]
    full_feats = FIN_FEATS + DYN_FEATS + MKT_FEATS + HIST_FEATS + \
        [f for f in RAT_FEATS if df[f].notna().any()]
    models = [
        AltmanFixed().fit(tr_fin),
        AltmanLogit().fit(tr_fin),
        MarsModel("MARS (отчётность)", FIN_FEATS + DYN_FEATS, need_fin=True,
                  max_terms=15, max_degree=1).fit(tr_fin),
        MarsModel("MARS (отчётность + рынок + рейтинги)", full_feats,
                  max_terms=21, max_degree=1, logit_C=0.5).fit(tr),
    ]

    preds = df[["company", "date", "sector", "segment", "split", "y", "is_snapshot", "z_altman"]].copy()
    for m in models:
        preds[m.name] = m.predict(df)

    rows = []
    roc = {}
    te_idx = preds.split == "test"
    common = te_idx & preds[[m.name for m in models]].notna().all(axis=1)
    for sample, mask in (("тест (все строки модели)", te_idx), ("тест (общая выборка с отчётностью)", common),
                         ("обучение", preds.split == "train")):
        for m in models:
            r = metrics(preds.loc[mask, "y"].to_numpy(float), preds.loc[mask, m.name].to_numpy(float))
            rows.append({"выборка": sample, "модель": m.name} | r)
    met = pd.DataFrame(rows)
    for m in models:
        y = preds.loc[common, "y"].to_numpy(float)
        p = preds.loc[common, m.name].to_numpy(float)
        if len(np.unique(y)) == 2:
            f, t, _ = roc_curve(y, p)
            roc[m.name] = {"fpr": f.round(4).tolist(), "tpr": t.round(4).tolist()}

    yc = preds.loc[common, "y"].to_numpy(float)
    comp = []
    if len(np.unique(yc)) == 2:
        for a, b in ((models[3], models[0]), (models[2], models[0]), (models[3], models[2])):
            d, lo, hi, pv = boot_auc_diff(yc, preds.loc[common, a.name].to_numpy(float),
                                          preds.loc[common, b.name].to_numpy(float))
            comp.append(f"ΔAUC [{a.name}] − [{b.name}] = {d:+.3f} (95% ДИ {lo:+.3f}…{hi:+.3f}, "
                        f"доля бутстрэп-выборок с Δ≤0: {pv:.3f})")

    MODELS.mkdir(exist_ok=True)
    with open(MODELS / "models.pkl", "wb") as f:
        pickle.dump({"models": models, "full_feats": full_feats}, f)
    met.to_csv(MODELS / "metrics.csv", index=False)
    preds.to_csv(MODELS / "predictions.csv", index=False)
    (MODELS / "roc.json").write_text(json.dumps(roc), "utf-8")

    imp_lines = []
    for m in models[2:]:
        imp_lines.append(f"\n{m.name}: важность признаков (доля прироста GCV)")
        imp_lines += [f"  {k:28s} {v:6.1%}" for k, v in list(m.m.importance_.items())[:12]]
        imp_lines.append(f"  Уравнение (логит):\n  " + m.m.summary().replace("\n", "\n  "))
    alt = models[0].lr
    report = "\n".join([
        f"Цель: дефолт (режим '{TARGET_MODE}') в течение {HORIZON_DAYS} дней после даты среза",
        f"Обучение: срезы ≤ {TRAIN_END}, n={len(tr)}, дефолтов={int(tr.y.sum())}",
        f"Тест: срезы ≥ {TEST_START} + {', '.join(FORCE_TEST_COMPANIES)}, n={len(te)}, дефолтов={int(te.y.sum())}",
        "", met.round(3).to_string(index=False), "", *comp,
        f"\nКалибровка Альтмана: PD = σ({alt.intercept_[0]:+.3f} {alt.coef_[0][0]:+.3f}·Z'')",
        *imp_lines,
    ])
    (MODELS / "report.txt").write_text(report, "utf-8")
    print(report)


if __name__ == "__main__":
    main()
