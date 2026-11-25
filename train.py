"""Шаг 4. Обучение и сравнение моделей PD на 6 месяцев (out-of-time).

Разбиение (даты — в config.py):
  train            — метка известна, окно (t, t+6м] закончилось не позже TRAIN_DATE, эмитент не из holdout;
  gap              — метка известна, но окно пересекает TRAIN_DATE (исход на дату обучения неизвестен) —
                     не используется;
  test             — общий тест: срезы строго после TRAIN_DATE с завершённым окном (все эмитенты,
                     включая Самолёт в этом периоде);
  holdout_backtest — срезы Самолёта до TRAIN_DATE: вне обучения по эмитенту, но не по времени —
                     показываются отдельно, в общую метрику не входят;
  score            — метки нет (окно открыто, нет покрытия реестра, эмитент в дефолте): только прогноз.

Модели:
  1. Altman Z'' (классика)          — исходные веса 1995 г., в вероятность — логит-калибровкой на train;
  2. Altman-логит                    — X1..X4, веса переоценены на train;
  3. MARS (отчётность)               — коэффициенты РСБУ/МСФО и их годовая динамика;
  4. Логит (те же входы, что у 5)    — контроль: отделяет вклад формы MARS от вклада признаков;
  5. MARS (отчётность+рынок+рейтинг) — основная модель.
Вся предобработка (квантили, медианы) и гиперпараметры фиксируются по train, тест не используется.

Выход (models/): models.pkl, metrics.csv, predictions.csv, split_export.csv, periods.csv,
                 holdout_samolet.csv, roc.json, report.txt
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, average_precision_score

from config import (DATA, MODELS, TRAIN_DATE, TODAY, HOLDOUT_COMPANIES_INN, TARGET_MODE, TARGET_NAME,
                    HORIZON_MONTHS, REGISTRY_COVERAGE_START, POPULATION)
from features import FIN_FEATS, DYN_FEATS, MKT_FEATS, RAT_FEATS, HIST_FEATS
from pd_models import AltmanFixed, AltmanLogit, MarsModel, LogitModel


def assign_splits(df, train_date=TRAIN_DATE, holdout=HOLDOUT_COMPANIES_INN):
    td = pd.Timestamp(train_date)
    known = df.label_status.eq("known")
    hold = df.inn.astype(str).isin([str(x) for x in holdout])
    split = pd.Series("score", index=df.index)
    split[known & (df.window_end <= td) & ~hold] = "train"
    split[known & (df.date <= td) & (df.window_end > td) & ~hold] = "gap"
    split[known & (df.date > td)] = "test"
    split[known & (df.date <= td) & hold] = "holdout_backtest"
    return split


def check_splits(df, train_date=TRAIN_DATE, holdout=HOLDOUT_COMPANIES_INN):
    tr, te = df[df.split == "train"], df[df.split == "test"]
    td = pd.Timestamp(train_date)
    checks = {
        "в обучении нет эмитентов holdout (Самолёт)": not tr.inn.astype(str).isin(holdout).any(),
        "у всех обучающих строк окно метки закончилось к дате обучения": bool((tr.window_end <= td).all()),
        "все обучающие метки известны": bool(tr.label_status.eq("known").all()),
        "все срезы теста позже даты обучения": bool((te.date > td).all()),
        "у всех тестовых строк окно завершилось к дате выгрузки": bool((te.window_end <= TODAY).all()),
    }
    return checks


def ks_stat(y, p):
    f, t, _ = roc_curve(y, p)
    return float(np.max(t - f))


def metrics(y, p):
    n_all = len(y)
    ok = ~np.isnan(p)
    y, p = y[ok], p[ok]
    r = {"строк": n_all, "покрытие": ok.mean() if n_all else np.nan, "n": int(len(y)), "событий": int(y.sum()),
         "доля событий": y.mean() if len(y) else np.nan, "средний PD": p.mean() if len(p) else np.nan}
    if len(np.unique(y)) < 2:
        return r
    auc = roc_auc_score(y, p)
    return r | {"ROC AUC": auc, "Gini": 2 * auc - 1, "KS": ks_stat(y, p),
                "AP (average precision)": average_precision_score(y, p), "Brier": brier_score_loss(y, p)}


def cluster_boot_auc_diff(y, p1, p2, groups, n=1000, seed=0):
    """Бутстрэп по эмитентам: пересэмплируются эмитенты целиком со всеми их кварталами."""
    ok = ~np.isnan(p1) & ~np.isnan(p2)
    y, p1, p2, groups = y[ok], p1[ok], p2[ok], np.asarray(groups)[ok]
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        if y[i].min() == y[i].max():
            continue
        d.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    if not d:
        return None
    d = np.array(d)
    return float(roc_auc_score(y, p1) - roc_auc_score(y, p2)), float(np.quantile(d, 0.025)), \
        float(np.quantile(d, 0.975)), float((d <= 0).mean()), len(ug)


def period_table(df):
    x = df[df.split.isin(["train", "gap", "test", "holdout_backtest"])]
    pos = x[x.y == 1]
    g = x.groupby(["date", "split"]).agg(эмитентов=("inn", "nunique"), строк=("inn", "size"),
                                         положительных=("y", "sum"), доля=("y", "mean"))
    ev = pos.groupby(["date", "split"]).apply(lambda z: z[["inn", "next_event"]].drop_duplicates().shape[0],
                                              include_groups=False)
    g["уникальных_событий"] = ev.reindex(g.index).fillna(0).astype(int)
    by_origin = x.groupby(["date", "split", "origin"]).y.mean().unstack()
    by_origin.columns = [f"доля ({c})" for c in by_origin.columns]
    return g.join(by_origin).reset_index()


def main():
    df = pd.read_csv(DATA / "dataset.csv", dtype={"inn": str},
                     parse_dates=["date", "next_event", "window_end", "fin_period_end", "quote_date"])
    df["split"] = assign_splits(df)
    checks = check_splits(df)
    tr, te = df[df.split == "train"], df[df.split == "test"]
    print(f"Обучение: {len(tr)} строк, {int(tr.y.sum())} положит. | Тест: {len(te)} строк, {int(te.y.sum())} положит.")
    for k, v in checks.items():
        print(("  ✓ " if v else "  ✗ ") + k)
    if not all(checks.values()):
        raise SystemExit("Нарушены правила разбиения — см. выше")
    if tr.y.sum() < 5 or len(np.unique(tr.y)) < 2:
        raise SystemExit("В обучении меньше 5 событий — сдвиньте TRAIN_DATE позже или расширьте реестр/выборку.")

    tr_fin = tr[tr.z_altman.notna()]
    if tr_fin.empty or tr_fin.y.nunique() < 2:
        raise SystemExit("В обучении нет строк с отчётностью (или в них нет обоих классов): модели Альтмана и MARS "
                         "по отчётности обучить нельзя. Запустите collect_girbo.py или заполните "
                         "data/financials_manual.csv, затем повторите build_dataset.py.")
    rat = [f for f in RAT_FEATS if tr[f].notna().any()]
    full_feats = FIN_FEATS + DYN_FEATS + MKT_FEATS + HIST_FEATS + rat
    models = [
        AltmanFixed().fit(tr_fin),
        AltmanLogit().fit(tr_fin),
        MarsModel("MARS (отчётность)", FIN_FEATS + DYN_FEATS, need_fin=True, max_terms=15, max_degree=1).fit(tr_fin),
        LogitModel("Логит (входы полного MARS)", full_feats).fit(tr),
        MarsModel("MARS (отчётность + рынок + рейтинги)", full_feats, max_terms=21, max_degree=1, logit_C=0.5).fit(tr),
    ]
    names = [m.name for m in models]

    keep = ["inn", "company", "date", "window_end", "label_status", "sector", "segment", "origin", "split", "y",
            "is_snapshot", "z_altman", "next_event"]
    preds = df[keep].copy()
    for m in models:
        preds[m.name] = m.predict(df)
    preds["imputed_share"] = models[-1].prep.imputed_share(df)

    rows = []
    te_mask = preds.split == "test"
    common = te_mask & preds[names].notna().all(axis=1)
    samples = [("тест: все строки", te_mask), ("тест: общая выборка (у всех моделей есть оценка)", common),
               ("тест: только отобранные компании (universe)", common & preds.origin.eq("universe")),
               ("обучение", preds.split == "train")]
    for sname, mask in samples:
        for m in models:
            rows.append({"выборка": sname, "модель": m.name} |
                        metrics(preds.loc[mask, "y"].to_numpy(float), preds.loc[mask, m.name].to_numpy(float)))
    met = pd.DataFrame(rows)

    roc = {}
    yc = preds.loc[common, "y"].to_numpy(float)
    if len(np.unique(yc)) == 2:
        for m in models:
            f, t, _ = roc_curve(yc, preds.loc[common, m.name].to_numpy(float))
            roc[m.name] = {"fpr": f.round(4).tolist(), "tpr": t.round(4).tolist()}
    comp = []
    if len(np.unique(yc)) == 2:
        pairs = [(4, 0, "полный MARS vs Альтман"), (2, 0, "MARS на той же отчётности vs Альтман"),
                 (4, 3, "форма MARS vs логит на тех же входах"), (4, 2, "вклад рынка/рейтингов в MARS")]
        for a, b, label in pairs:
            r = cluster_boot_auc_diff(yc, preds.loc[common, names[a]].to_numpy(float),
                                      preds.loc[common, names[b]].to_numpy(float), preds.loc[common, "inn"])
            if r:
                d, lo, hi, pv, ng = r
                comp.append(f"{label}: ΔAUC = {d:+.3f} (95% ДИ {lo:+.3f}…{hi:+.3f}; бутстрэп по {ng} эмитентам; "
                            f"доля Δ≤0: {pv:.2f}) → " +
                            ("различие значимо" if lo > 0 or hi < 0 else "убедительной разницы нет"))

    per = period_table(preds.assign(y=df.y))
    hold = preds[preds.inn.isin(HOLDOUT_COMPANIES_INN)].merge(
        df[["inn", "date", "fin_period_end", "fin_source", "mkt_price_min", "mkt_yield_med", "quote_date"]],
        on=["inn", "date"], how="left")
    exp = preds[["inn", "company", "date", "window_end", "label_status", "y", "next_event", "split", "origin"]]

    MODELS.mkdir(exist_ok=True)
    meta = {"train_date": TRAIN_DATE, "as_of": str(TODAY.date()), "target_mode": TARGET_MODE,
            "target_name": TARGET_NAME, "horizon_months": HORIZON_MONTHS,
            "coverage_start": REGISTRY_COVERAGE_START, "population": POPULATION,
            "n_train": int(len(tr)), "events_train": int(tr.y.sum()), "checks": checks}
    with open(MODELS / "models.pkl", "wb") as f:
        pickle.dump({"models": models, "full_feats": full_feats, "meta": meta}, f)
    met.to_csv(MODELS / "metrics.csv", index=False)
    preds.to_csv(MODELS / "predictions.csv", index=False)
    exp.to_csv(MODELS / "split_export.csv", index=False)
    per.to_csv(MODELS / "periods.csv", index=False)
    hold.to_csv(MODELS / "holdout_samolet.csv", index=False)
    (MODELS / "roc.json").write_text(json.dumps(roc), "utf-8")

    imp = []
    for m in models:
        if m.kind == "MARS":
            imp.append(f"\n{m.name}: важность признаков (доля прироста GCV), ранг прямого прохода "
                       f"{m.m.forward_rank_}")
            imp += [f"  {k:28s} {v:6.1%}" for k, v in list(m.m.importance_.items())[:12]]
            imp.append("  Уравнение (логит):\n  " + m.m.summary().replace("\n", "\n  "))
    alt = models[0].lr
    sc = ["date", "split", "label_status", "y"] + names
    report = "\n".join([
        f"Цель: {TARGET_NAME} (режим '{TARGET_MODE}')",
        f"Дата выгрузки: {TODAY.date()} | Модель обучена на дату: {TRAIN_DATE} | Покрытие реестра с {REGISTRY_COVERAGE_START} "
        f"| Совокупность: {POPULATION}",
        f"Обучение: n={len(tr)}, положительных={int(tr.y.sum())}, эмитентов={tr.inn.nunique()}",
        f"Тест (срезы {te.date.min():%d.%m.%Y}–{te.date.max():%d.%m.%Y}): n={len(te)}, положительных={int(te.y.sum())}, "
        f"эмитентов={te.inn.nunique()}" if len(te) else "Тест: пусто",
        "Проверки разбиения: " + "; ".join(f"{k}: {'да' if v else 'НЕТ'}" for k, v in checks.items()),
        "", "Метрики (AP — average precision; Brier — ошибка вероятностей; покрытие — доля строк с оценкой):",
        met.round(3).to_string(index=False), "", "Сравнение моделей (кластерный бутстрэп):", *comp,
        f"\nКалибровка Альтмана: PD = σ({alt.intercept_[0]:+.3f} {alt.coef_[0][0]:+.3f}·Z'')",
        "\nПо периодам (строки ≠ события: один дефолт даёт до двух положительных кварталов):",
        per.assign(date=per.date.dt.strftime('%Y-%m-%d')).round(3).to_string(index=False),
        f"\nСамолёт (вне обучения; модель обучена на {TRAIN_DATE}):",
        hold[sc].assign(date=hold.date.dt.strftime("%Y-%m-%d")).round(4).to_string(index=False)
        if len(hold) else "нет строк",
        *imp,
    ])
    (MODELS / "report.txt").write_text(report, "utf-8")
    print(report)


if __name__ == "__main__":
    main()
