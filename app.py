"""Шаг 5. Интерфейс: вероятность события неплатежа на 6 месяцев по облигации / эмитенту.

Запуск:  streamlit run app.py
"""
import json
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import (DATA, RAW, MODELS, LGD, RISK_FREE_ANNUAL, MAX_WEIGHT, KELLY_FRACTION, MAX_STALE_PRICE_DAYS,
                    TARGET_MODE)
from features import ratios, dynamics, altman_zone, missing_lines, LABELS_RU, FIN_FEATS, DYN_FEATS, MKT_FEATS

st.set_page_config(page_title="PD облигаций РФ", page_icon="📉", layout="wide")
MANDATORY = {"9731004688": "Самолет", "7826705374": "Балтийский лизинг", "7838360491": "ЛСР",
             "7724451748": "ДжетЛенд", "7705619586": "Лидер-Инвест"}


# ============================================================================= данные
@st.cache_data
def load():
    need = [MODELS / "predictions.csv", DATA / "dataset.csv", MODELS / "models.pkl"]
    if not all(p.exists() for p in need):
        return None
    dt = ["date", "window_end", "next_event"]
    pr = pd.read_csv(MODELS / "predictions.csv", dtype={"inn": str}, parse_dates=dt)
    ds = pd.read_csv(DATA / "dataset.csv", dtype={"inn": str},
                     parse_dates=dt + ["fin_period_end", "fin_publish_date", "quote_date"])
    bf = RAW / "bonds.csv"
    bonds = pd.read_csv(bf, dtype={"inn": str}, parse_dates=["issuedate", "matdate"]) \
        if bf.exists() and bf.stat().st_size > 10 else pd.DataFrame(columns=["secid", "inn", "isin", "shortname",
                                                                              "issuedate", "matdate", "company"])
    met = pd.read_csv(MODELS / "metrics.csv")
    roc = json.loads((MODELS / "roc.json").read_text("utf-8"))
    rep = (MODELS / "report.txt").read_text("utf-8")
    reg = pd.read_csv(DATA / "defaults_registry.csv", dtype={"inn": str}, keep_default_na=False)
    reg["event_date"] = pd.to_datetime(reg.event_date)
    per = pd.read_csv(MODELS / "periods.csv") if (MODELS / "periods.csv").exists() else pd.DataFrame()
    return pr, ds, bonds, met, roc, rep, reg, per


@st.cache_resource
def load_models():
    with open(MODELS / "models.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def bond_history(secid):
    f = RAW / "history" / f"{secid}.csv"
    return pd.read_csv(f, parse_dates=["tradedate"]) if f.exists() else pd.DataFrame(
        columns=["tradedate", "price", "yield", "value"])


@st.cache_data
def reports_for(inn):
    """Отчёт, который модель использовала бы на последний срез (с потоками за 12 мес.), и отчёт годом ранее."""
    import build_dataset as bd
    em = pd.read_csv(RAW / "emitters.csv", dtype={"inn": str}) if (RAW / "emitters.csv").exists() else pd.DataFrame()
    n2i = dict(zip(em.get("company", []), em.get("inn", [])))
    try:
        fin = bd.load_financials(n2i)
    except Exception:  # noqa: BLE001
        return None, None
    return bd.pick_report(fin.get(inn), pd.Timestamp.today())


data = load()
if data is None:
    st.error("Модели ещё не обучены. Выполните по порядку: `python collect_moex.py`, `python collect_girbo.py`, "
             "`python build_dataset.py`, `python train.py`, затем перезапустите интерфейс.")
    st.stop()
pr, ds, bonds, met, roc, rep, reg, per = data
bundle = load_models()
META = bundle.get("meta", {})
MODELS_ = bundle["models"]
MODEL_NAMES = [m.name for m in MODELS_]
MAIN, ALT = MODEL_NAMES[-1], MODEL_NAMES[0]
FIN_MODEL = next((n for n in MODEL_NAMES if n.startswith("MARS (отч") and "рынок" not in n), MODEL_NAMES[0])
latest = pr.sort_values("date").groupby("inn").tail(1).set_index("inn")
latest_ds = ds.sort_values("date").groupby("inn").tail(1).set_index("inn")


# ============================================================================= инвест-правило
def allocation(pd6, ytm_annual, quote_age_days=0, in_default=False):
    """Доля капитала: дробный Келли для бинарного исхода за 6 мес.
    Нет события: доходность за полгода r6 (из годовой YTM по сложной ставке), сверх ставки сравнения rf6.
    Событие: потеря LGD. Упрощение: цена реализации через полгода и денежные потоки выпуска не моделируются.
    → dict: status (ok / unavailable), verdict, weight, шаги расчёта."""
    rf6 = (1 + RISK_FREE_ANNUAL) ** 0.5 - 1
    out = {"status": "unavailable", "weight": np.nan, "verdict": "", "PD": pd6, "YTM": ytm_annual,
           "r6": np.nan, "rf6": rf6, "LGD": LGD, "exp_ret": np.nan, "kelly": np.nan}
    if in_default:
        return out | {"status": "ok", "weight": 0.0, "verdict": "Не инвестировать: эмитент в дефолте"}
    if pd6 is None or not np.isfinite(pd6):
        return out | {"verdict": "Расчёт недоступен: нет оценки PD (не хватает данных)"}
    if ytm_annual is None or not np.isfinite(ytm_annual):
        return out | {"verdict": "Расчёт недоступен: нет доходности к погашению"}
    if quote_age_days is not None and quote_age_days > MAX_STALE_PRICE_DAYS:
        return out | {"verdict": f"Расчёт недоступен: котировка устарела ({quote_age_days} дн.)"}
    r6 = (1 + ytm_annual / 100) ** 0.5 - 1
    exp_ret = (1 - pd6) * r6 - pd6 * LGD
    out |= {"status": "ok", "r6": r6, "exp_ret": exp_ret}
    if pd6 >= 0.5:
        return out | {"weight": 0.0, "verdict": "Не инвестировать: PD ≥ 50%"}
    if r6 <= rf6:
        return out | {"weight": 0.0, "verdict": "Не инвестировать: доходность не выше ставки сравнения"}
    b, a = r6 - rf6, LGD + rf6
    kelly = (1 - pd6) / a - pd6 / b
    w = float(np.clip(KELLY_FRACTION * kelly, 0, MAX_WEIGHT))
    verdict = ("Не инвестировать: премия не покрывает ожидаемые потери" if w <= 0 else
               "Инвестировать" if pd6 < 0.02 else "Инвестировать ограниченно" if pd6 < 0.10 else
               "Только спекулятивно, малой долей")
    return out | {"kelly": kelly, "weight": w, "verdict": verdict}


def alloc_table(a, pd_label, pd_date, ytm_date):
    f = lambda x, d=2: "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"  # noqa: E731
    rows = [("PD за 6 мес.", f(a["PD"]), f"{pd_label}, срез {pd_date}"),
            ("Доходность к погашению (YTM)", "—" if not np.isfinite(a["YTM"] if a["YTM"] is not None else np.nan)
             else f"{a['YTM']:.2f}%", f"дата котировки {ytm_date}"),
            ("r6 = (1+YTM)^0.5 − 1", f(a["r6"]), "доходность за полгода без события"),
            ("rf6", f(a["rf6"]), f"ставка сравнения {RISK_FREE_ANNUAL:.1%} годовых за полгода"),
            ("LGD", f(a["LGD"], 0), "потери при событии (консервативно — как при дефолте)" if TARGET_MODE == "broad"
             else "потери при дефолте"),
            ("E[доходность] = (1−PD)·r6 − PD·LGD", f(a["exp_ret"]), ""),
            ("Келли = (1−PD)/(LGD+rf6) − PD/(r6−rf6)", f(a["kelly"], 1), ""),
            ("Доля = min(½·Келли, потолок)", f(a["weight"], 1), f"потолок {MAX_WEIGHT:.0%}, ½ Келли")]
    return pd.DataFrame(rows, columns=["Шаг", "Значение", "Откуда"])


def fmt_pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


def fmt_d(x):
    return "—" if x is None or pd.isna(x) else pd.Timestamp(x).strftime("%d.%m.%Y")


def last_quote(h, asof):
    """Последняя сделка не позже даты asof; доходность берётся с той же даты (или «нет»)."""
    h = h[h.tradedate <= asof]
    if h.empty:
        return np.nan, np.nan, pd.NaT
    r = h.iloc[-1]
    return float(r.price), (float(r["yield"]) if pd.notna(r["yield"]) else np.nan), r.tradedate


# ============================================================================= шапка
st.title("Вероятность неплатежа эмитентов облигаций РФ на горизонте 6 месяцев")
st.caption(f"Цель: {META.get('target_name', '—')}. Модель обучена на дату {META.get('train_date', '—')} "
           f"(в обучении только срезы с завершённым к этой дате окном; Самолёт вне обучения). "
           f"Дата выгрузки данных: {META.get('as_of', '—')}. Основная модель — «{MAIN}».")

tab1, tab2, tab3, tab4 = st.tabs(["🔎 Облигация / эмитент", "📋 Скринер", "⚖️ MARS vs Альтман", "✍️ Своя отчётность"])

# ============================================================================= 1. облигация
with tab1:
    show_reg = st.toggle("Показывать эмитентов, добавленных из реестра событий", value=False)
    inns = latest.index if show_reg else latest.index[latest.origin != "registry"]
    b = bonds[bonds.inn.isin(inns)].copy()
    name = latest.company.to_dict()
    b["company"] = b.inn.map(name)
    b["label"] = b.company + " — " + b.shortname.astype(str) + " (" + b["isin"].astype(str) + ")"
    b["active"] = b.matdate.isna() | (b.matdate >= pd.Timestamp.today())
    c0, c1, c2 = st.columns([1, 1, 2])
    must = c0.selectbox("Обязательные компании", ["—"] + [v for k, v in MANDATORY.items() if k in latest.index])
    sectors = ["Все"] + sorted(latest.loc[inns, "sector"].dropna().unique())
    sector = c1.selectbox("Отрасль", sectors)
    only_active = c1.checkbox("Только непогашенные выпуски", value=True)
    if must != "—":
        b = b[b.inn == next(k for k, v in MANDATORY.items() if v == must)]
    elif sector != "Все":
        b = b[b.inn.map(latest.sector) == sector]
    if only_active and b.active.any():
        b = b[b.active]
    opts = b.sort_values(["company", "matdate"]).label.tolist()
    dflt = next((i for i, o in enumerate(opts) if o.startswith("Самолет")), 0)
    sel = c2.selectbox(f"Облигация ({len(opts)} выпусков)", opts, index=dflt if opts else None)
    if not opts:
        st.warning("По фильтру нет выпусков с данными. Проверьте журнал сборщика data/raw/collect_log.csv.")
    if sel:
        bond = b[b.label == sel].iloc[0]
        inn = bond.inn
        L, Ld = latest.loc[inn], latest_ds.loc[inn]
        eval_date = Ld.date
        h = bond_history(bond.secid)
        price, ytm, qdate = last_quote(h, eval_date)
        qage = (eval_date - qdate).days if pd.notna(qdate) else None
        in_def = Ld.label_status == "in_default"

        st.subheader(f"{L.company} · {bond.shortname}")
        st.caption(f"{L.sector} · {L.segment} · ИНН {inn} · ISIN {bond.isin} · погашение {fmt_d(bond.matdate)}")
        if in_def:
            last_d = reg[(reg.inn == inn) | (reg.company == L.company)]
            st.error(f"Эмитент в дефолте (последнее событие {fmt_d(last_d.event_date.max())}). PD не оценивается, "
                     "вложение — 0%.")
        d1, d2, d3 = st.columns(3)
        d1.markdown(f"**Дата оценки (срез):** {fmt_d(eval_date)}  \n"
                    f"Статус метки: `{Ld.label_status}`; окно до {fmt_d(Ld.window_end)}")
        d2.markdown(f"**Отчётность:** период до {fmt_d(Ld.get('fin_period_end'))}, "
                    f"опубликована {fmt_d(Ld.get('fin_publish_date'))}  \n"
                    f"{Ld.get('fin_source', '—')} · {Ld.get('fin_standard', '')} {Ld.get('fin_perimeter', '')} · "
                    f"потоки: {Ld.get('fin_flow_basis', '—')}" if pd.notna(Ld.get("fin_period_end"))
                    else "**Отчётность:** нет опубликованной к дате оценки")
        d3.markdown(f"**Котировка выпуска:** {fmt_d(qdate)}"
                    + (f" ({qage} дн. до даты оценки)" if qage else "") + "  \n"
                    + (f"цена {price:.2f}%, " if np.isfinite(price) else "нет сделок, ")
                    + (f"YTM {ytm:.2f}%" if np.isfinite(ytm) else "доходность на эту дату не рассчитана"))

        imp = float(L.get("imputed_share", np.nan))
        miss = str(Ld.get("fin_missing_lines", "") or "")
        warn = []
        if np.isfinite(imp) and imp > 0:
            warn.append(f"{imp:.0%} входов основной модели нет в данных и подставлены медианой обучающей выборки")
        if miss and miss != "nan":
            warn.append(f"в отчёте нет строк: {miss}")
        if pd.isna(Ld.get("mkt_price_med")):
            warn.append("нет свежих котировок облигаций эмитента — рыночные признаки пустые")
        if warn:
            (st.warning if (np.isfinite(imp) and imp > 0.3) else st.info)("Данные неполные: " + "; ".join(warn) + ".")

        m = st.columns(5)
        m[0].metric(f"PD 6м — {MAIN}", fmt_pct(L[MAIN]))
        m[1].metric(f"PD 6м — {FIN_MODEL}", fmt_pct(L[FIN_MODEL]))
        m[2].metric("PD 6м — Альтман Z''", fmt_pct(L[ALT]))
        z = L.z_altman
        m[3].metric("Z''-счёт / зона", f"{z:.2f}" if pd.notna(z) else "—", altman_zone(z), delta_color="off")
        m[4].metric("Цена / YTM выпуска", f"{price:.1f}%" if np.isfinite(price) else "—",
                    f"YTM {ytm:.1f}%" if np.isfinite(ytm) else None, delta_color="off")

        model_for_alloc = st.radio("Модель для инвест-решения", MODEL_NAMES, index=len(MODEL_NAMES) - 1,
                                   horizontal=True)
        a = allocation(L[model_for_alloc], ytm, qage, in_def)
        (st.info if a["status"] == "ok" else st.warning)(
            f"**{a['verdict']}.** Доля капитала: **{fmt_pct(a['weight'])}**")
        with st.expander("Как получена доля"):
            st.dataframe(alloc_table(a, model_for_alloc, fmt_d(eval_date), fmt_d(qdate)), hide_index=True,
                         width="stretch")
            st.caption("Упрощение: годовая YTM переводится в полугодовую по сложной ставке; цена продажи через "
                       "полгода и график платежей выпуска не моделируются. Параметры — в config.py.")

        g1, g2 = st.columns(2)
        hist = pr[pr.inn == inn].sort_values("date")
        fig = go.Figure()
        for nm in MODEL_NAMES:
            fig.add_trace(go.Scatter(x=hist.date, y=hist[nm] * 100, name=nm, mode="lines+markers"))
        for ev in reg[(reg.inn == inn) | (reg.company == L.company)].itertuples():
            fig.add_vline(x=ev.event_date, line_dash="dot", line_color="red")
            fig.add_annotation(x=ev.event_date, y=1, yref="paper", text=ev.event_type, showarrow=False,
                               font=dict(color="red"))
        fig.add_vline(x=pd.Timestamp(META.get("train_date", "2025-06-30")), line_dash="dash", line_color="grey")
        fig.update_layout(title="PD на 6 мес. по срезам (красное — TD/D; серое — дата обучения модели)",
                          yaxis_title="PD, %", height=380, legend=dict(orientation="h", y=-0.3))
        g1.plotly_chart(fig, width="stretch")
        if len(h):
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=h.tradedate, y=h.price, name="Цена, % номинала"))
            fig2.add_trace(go.Scatter(x=h.tradedate, y=h["yield"], name="Доходность, %", yaxis="y2"))
            fig2.update_layout(title=f"Котировки {bond.shortname}", height=380,
                               yaxis2=dict(overlaying="y", side="right", title="YTM, %"),
                               legend=dict(orientation="h", y=-0.3))
            g2.plotly_chart(fig2, width="stretch")
        else:
            g2.info("По выпуску нет истории торгов.")

        st.markdown("**Признаки на дату оценки** (— значит нет в данных)")
        feats = FIN_FEATS + DYN_FEATS + MKT_FEATS + ["rating_num", "rating_chg_6m", "had_event_12m"]
        tbl = pd.DataFrame({"Признак": [LABELS_RU.get(f, f) for f in feats],
                            "Значение": [Ld.get(f, np.nan) for f in feats]})
        tbl["Значение"] = tbl["Значение"].map(lambda v: "—" if pd.isna(v) else f"{v:,.3f}")
        c3, c4 = st.columns(2)
        c3.dataframe(tbl, hide_index=True, width="stretch", height=420)
        mm = MODELS_[-1]
        x = mm.prep.transform(ds[ds.inn == inn].sort_values("date").tail(1))
        icpt, contr = mm.m.contributions(x[0])
        cdf = pd.DataFrame(contr, columns=["Базисная функция MARS", "Вклад в логит"])
        cdf = cdf[cdf["Вклад в логит"].abs() > 1e-6].sort_values("Вклад в логит", key=abs, ascending=False)
        c4.markdown(f"Разложение PD основной модели: свободный член {icpt:+.2f}; положительный вклад повышает PD. "
                    "Признаки `miss_*` — флаги отсутствия группы данных.")
        c4.dataframe(cdf, hide_index=True, width="stretch", height=380)

# ============================================================================= 2. скринер
with tab2:
    scr = latest.copy()
    scr["Цена мин., %"] = latest_ds.mkt_price_min
    scr["YTM медиан., %"] = latest_ds.mkt_yield_med
    scr["Котировка"] = latest_ds.quote_date
    scr["Отчётность до"] = latest_ds.fin_period_end
    scr["Зона Альтмана"] = scr.z_altman.map(altman_zone)
    scr["Подставлено"] = scr.imputed_share
    al = [allocation(r[MAIN], r["YTM медиан., %"], 0, r.label_status == "in_default") for _, r in scr.iterrows()]
    scr["Доля капитала"] = [x["weight"] for x in al]
    scr["Решение"] = [x["verdict"] for x in al]
    f1, f2, f3 = st.columns(3)
    segs = f1.multiselect("Сегмент", sorted(scr.segment.dropna().unique()),
                          default=[s for s in scr.segment.dropna().unique() if s not in ("Реестр событий", "Дефолт-реестр")])
    secs = f2.multiselect("Отрасль", sorted(scr.sector.dropna().unique()))
    sort_by = f3.selectbox("Сортировка", [MAIN] + MODEL_NAMES[:-1])
    v = scr[scr.segment.isin(segs)]
    if secs:
        v = v[v.sector.isin(secs)]
    v = v.sort_values(sort_by, ascending=False)
    cols = ["company", "sector", "segment", "date", "Отчётность до", "Котировка"] + MODEL_NAMES + \
        ["Подставлено", "z_altman", "Зона Альтмана", "Цена мин., %", "YTM медиан., %", "Доля капитала", "Решение"]
    show = v[cols].rename(columns={"company": "Эмитент", "sector": "Отрасль", "segment": "Сегмент",
                                   "date": "Срез", "z_altman": "Z''"})
    st.dataframe(show.style.format({**{n: "{:.1%}" for n in MODEL_NAMES}, "Доля капитала": "{:.1%}",
                                    "Подставлено": "{:.0%}", "Z''": "{:.2f}", "Цена мин., %": "{:.1f}",
                                    "YTM медиан., %": "{:.1f}", "Срез": "{:%d.%m.%Y}",
                                    "Отчётность до": "{:%d.%m.%Y}", "Котировка": "{:%d.%m.%Y}"}, na_rep="—")
                 .background_gradient(subset=[MAIN], cmap="Reds", vmin=0, vmax=0.5),
                 width="stretch", height=600)
    st.caption("«Подставлено» — доля входов основной модели, которых нет в данных. Доля капитала в скринере "
               "считается по медианной YTM эмитента; точный расчёт по выпуску — на первой вкладке.")
    pick = st.multiselect("Сравнить динамику PD", v.company.tolist(),
                          default=[c for c in MANDATORY.values() if c in set(v.company)])
    if pick:
        fig = go.Figure()
        for c in pick:
            hh = pr[pr.company == c].sort_values("date")
            fig.add_trace(go.Scatter(x=hh.date, y=hh[MAIN] * 100, name=c, mode="lines+markers"))
        fig.update_layout(yaxis_title=f"PD 6м, % ({MAIN})", height=420)
        st.plotly_chart(fig, width="stretch")

# ============================================================================= 3. сравнение
with tab3:
    st.markdown(f"**Проверка во времени.** Модели обучены на срезах, у которых окно метки закончилось не позже "
                f"{META.get('train_date')}; общий тест — срезы после этой даты с завершённым окном "
                f"(все эмитенты, включая Самолёт в этом периоде). «Общая выборка» — строки, где оценка есть у всех "
                f"моделей. Логит на тех же входах показывает, что даёт сама форма MARS. "
                f"Интервалы ΔAUC — бутстрэп по эмитентам (в отчёте ниже).")
    num = [c for c in met.columns if met[c].dtype.kind == "f"]
    st.dataframe(met.style.format({c: "{:.3f}" for c in num}, na_rep="—"), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    fig = go.Figure()
    for nm, r in roc.items():
        fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], name=nm, mode="lines"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), showlegend=False))
    fig.update_layout(title="ROC-кривые на тесте (общая выборка)", xaxis_title="FPR", yaxis_title="TPR",
                      height=450, legend=dict(orientation="h", y=-0.25))
    c1.plotly_chart(fig, width="stretch")
    te = pr[pr.split == "test"]
    fig = go.Figure()
    for nm in MODEL_NAMES:
        t = te[[nm, "y"]].dropna()
        if len(t) < 20:
            continue
        t["bin"] = pd.qcut(t[nm].rank(method="first"), 5, labels=False)
        g = t.groupby("bin").agg(p=(nm, "mean"), y=("y", "mean"))
        fig.add_trace(go.Scatter(x=g.p * 100, y=g.y * 100, name=nm, mode="lines+markers"))
    mx = float(np.nanmax(te[MODEL_NAMES].to_numpy()) * 100) if len(te) else 1
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], line=dict(dash="dash", color="grey"), showlegend=False))
    fig.update_layout(title="Калибровка на тесте: прогноз vs факт (квинтили)", xaxis_title="Средний PD, %",
                      yaxis_title="Доля событий, %", height=450, legend=dict(orientation="h", y=-0.25))
    c2.plotly_chart(fig, width="stretch")
    s = pr[pr.inn == "9731004688"].sort_values("date")
    if len(s):
        st.markdown(f"**Самолёт — отдельный пример вне обучения** (модель обучена на {META.get('train_date')}; "
                    "срезы до этой даты — вне обучения по эмитенту, после — и по времени)")
        st.dataframe(s[["date", "window_end", "split", "label_status", "y"] + MODEL_NAMES]
                     .style.format({**{n: "{:.1%}" for n in MODEL_NAMES}, "date": "{:%d.%m.%Y}",
                                    "window_end": "{:%d.%m.%Y}"}, na_rep="—"), hide_index=True, width="stretch")
    if len(per):
        with st.expander("Состав выборки по периодам (эмитенты, строки, уникальные события)"):
            st.dataframe(per, hide_index=True, width="stretch")
    with st.expander("Полный отчёт обучения"):
        st.code(rep)

# ============================================================================= 4. своя отчётность
with tab4:
    st.markdown("Введите строки отчёта (тыс. руб.; потоки — за 12 месяцев). **Пустое поле — нет данных**, "
                "ноль вводите явно. Можно подставить отчёт эмитента из базы и изменить его.")
    lines = [("1600", "Баланс (итого активы)"), ("1200", "Оборотные активы"), ("1210", "Запасы"),
             ("1250", "Денежные средства"), ("1240", "Краткоср. фин. вложения"), ("1300", "Капитал и резервы"),
             ("1370", "Нераспределённая прибыль"), ("1400", "Долгосрочные обязательства"),
             ("1410", "Долгосрочные заёмные средства"), ("1500", "Краткосрочные обязательства"),
             ("1510", "Краткосрочные заёмные средства"), ("2110", "Выручка"), ("2200", "Прибыль от продаж"),
             ("2300", "Прибыль до налогообложения"), ("2330", "Проценты к уплате"), ("2400", "Чистая прибыль")]
    opts = {"—": None} | {f"{r.company} (ИНН {i})": i for i, r in latest.sort_values("company").iterrows()}
    base_lbl = st.selectbox("Подставить отчётность эмитента из базы", list(opts))
    base = opts[base_lbl]
    pre_c, pre_p, base_row = {}, {}, {}
    if base:
        cur, prev = reports_for(base)
        pre_c, pre_p = (cur or {}), (prev or {})
        base_row = latest_ds.loc[base].to_dict()
        if cur:
            st.caption(f"Подставлен отчёт за период до {fmt_d(cur.get('period_end'))} "
                       f"({cur.get('standard')}, {cur.get('perimeter')}, потоки: {cur.get('fin_flow_basis')}); "
                       "рынок и рейтинг — с последнего среза эмитента.")
        else:
            st.caption("В базе нет отчётности этого эмитента — заполните поля вручную.")

    def _v(d, code):
        x = d.get(f"c{code}")
        return float(x) if x is not None and pd.notna(x) else None

    with st.form("manual"):
        cA, cB, cC = st.columns(3)
        curv, prevv = {}, {}
        cA.markdown("**Отчётный период**")
        cB.markdown("**Тот же период годом ранее** (для динамики)")
        for code, title in lines:
            curv[f"c{code}"] = cA.number_input(f"{code} {title}", value=_v(pre_c, code), key=f"c{code}_{base}",
                                               format="%.0f", placeholder="нет данных")
            prevv[f"c{code}"] = cB.number_input(f"{code} {title} ", value=_v(pre_p, code), key=f"p{code}_{base}",
                                                format="%.0f", placeholder="нет данных")
        cC.markdown("**Рынок (необязательно)**")
        pm = base_row.get("mkt_price_min")
        ym = base_row.get("mkt_yield_med")
        price = cC.number_input("Мин. цена облигаций, % номинала", value=float(pm) if pd.notna(pm) else None,
                                placeholder="нет данных", key=f"pr_{base}")
        ytm = cC.number_input("Доходность к погашению, %", value=float(ym) if pd.notna(ym) else None,
                              placeholder="нет данных", key=f"y_{base}")
        had = cC.checkbox("Были техдефолты за последние 12 мес.", value=bool(base_row.get("had_event_12m", 0) == 1),
                          key=f"h_{base}")
        go_btn = st.form_submit_button("Рассчитать PD")
    if go_btn:
        cur = {k: (np.nan if v is None else v) for k, v in curv.items()}
        prev = {k: (np.nan if v is None else v) for k, v in prevv.items()}
        has_prev = any(np.isfinite(v) for v in prev.values())
        rc = ratios(cur)
        rp = ratios(prev) if has_prev else None
        row = {k: base_row.get(k, np.nan) for k in MKT_FEATS + ["rating_num", "rating_chg_6m"]}
        row |= {k: rc[k] for k in FIN_FEATS} | {"z_altman": rc["z_altman"]}
        row |= dynamics(rc, rp, cur, prev if has_prev else None)
        if price is not None:
            row["mkt_price_min"] = price
            if not base:
                row["mkt_price_med"] = price
        elif not base:
            row["mkt_price_min"] = np.nan
        if ytm is not None:
            if base and pd.notna(base_row.get("mkt_yield_med")) and ytm != base_row.get("mkt_yield_med"):
                row["mkt_spread"] = (base_row.get("mkt_spread", 0) or 0) + ytm - base_row["mkt_yield_med"]
            row["mkt_yield_med"] = ytm
            if not base:
                row["mkt_yield_max"] = ytm
                med = ds[ds.is_snapshot == 1].mkt_yield_med.median()
                row["mkt_spread"] = ytm - med if pd.notna(med) else np.nan
        row["had_event_12m"] = float(had)
        one = pd.DataFrame([row])
        out = {m.name: float(m.predict(one)[0]) for m in MODELS_}
        miss = missing_lines(cur)
        if miss:
            st.warning("Нет строк: " + "; ".join(miss) + ". Зависящие от них коэффициенты — пропуски, модель "
                       "заменяет их медианой обучающей выборки.")
        imp = float(MODELS_[-1].prep.imputed_share(one)[0])
        st.caption(f"Доля входов основной модели, подставленных медианой: {imp:.0%}")
        k = st.columns(len(out) + 1)
        for i, (nm, p) in enumerate(out.items()):
            k[i].metric(nm, fmt_pct(p))
        k[-1].metric("Z''-счёт", f"{rc['z_altman']:.2f}" if np.isfinite(rc["z_altman"]) else "—",
                     altman_zone(rc["z_altman"]), delta_color="off")
        if base:
            st.info(f"Для сверки: в базе на срез {fmt_d(latest.loc[base].date)} основная модель дала "
                    f"{fmt_pct(latest.loc[base][MAIN])}. Если отчёт не менялся, расхождение возникает из-за "
                    f"отчёта, опубликованного позже среза, или изменённых рыночных полей.")
        a = allocation(out[MAIN], ytm if ytm is not None else np.nan)
        (st.info if a["status"] == "ok" else st.warning)(f"**{a['verdict']}.** Доля капитала: {fmt_pct(a['weight'])}")
        with st.expander("Как получена доля"):
            st.dataframe(alloc_table(a, MAIN, "ручной ввод", "ручной ввод"), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame({"Коэффициент": [LABELS_RU.get(k2, k2) for k2 in row],
                                   "Значение": [f"{v:,.3f}" if pd.notna(v) else "—" for v in row.values()]}),
                     hide_index=True, width="stretch")

st.caption("PD — вероятность события, заданного целью модели, в течение 6 месяцев. Учебная модель, "
           "не является индивидуальной инвестиционной рекомендацией.")
