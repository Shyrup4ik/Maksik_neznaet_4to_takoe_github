"""Шаг 5. Интерфейс: вероятность дефолта на 6 месяцев по облигации / компании.

Запуск:  streamlit run app.py
"""
import json
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import DATA, RAW, MODELS, LGD, RISK_FREE_ANNUAL, MAX_WEIGHT, KELLY_FRACTION, HORIZON_DAYS
from features import ratios, dynamics, altman_zone, LABELS_RU, FIN_FEATS, DYN_FEATS, MKT_FEATS

st.set_page_config(page_title="PD облигаций РФ", page_icon="📉", layout="wide")


# ----------------------------------------------------------------------------- данные
@st.cache_data
def load():
    need = [MODELS / "predictions.csv", DATA / "dataset.csv", MODELS / "models.pkl"]
    if not all(p.exists() for p in need):
        return None
    pr = pd.read_csv(MODELS / "predictions.csv", parse_dates=["date"])
    ds = pd.read_csv(DATA / "dataset.csv", parse_dates=["date", "next_event"])
    bonds = pd.read_csv(RAW / "bonds.csv", parse_dates=["issuedate", "matdate"])
    met = pd.read_csv(MODELS / "metrics.csv")
    roc = json.loads((MODELS / "roc.json").read_text("utf-8"))
    rep = (MODELS / "report.txt").read_text("utf-8")
    reg = pd.read_csv(DATA / "defaults_registry.csv", parse_dates=["event_date"])
    return pr, ds, bonds, met, roc, rep, reg


@st.cache_resource
def load_models():
    with open(MODELS / "models.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def bond_history(secid):
    f = RAW / "history" / f"{secid}.csv"
    return pd.read_csv(f, parse_dates=["tradedate"]) if f.exists() else pd.DataFrame()


data = load()
if data is None:
    st.error("Модели ещё не обучены. Выполните по порядку: `python collect_moex.py`, `python collect_girbo.py`, "
             "`python build_dataset.py`, `python train.py`, затем перезапустите интерфейс.")
    st.stop()
pr, ds, bonds, met, roc, rep, reg = data
bundle = load_models()
MODEL_NAMES = [m.name for m in bundle["models"]]
MAIN = MODEL_NAMES[-1]
ALT = MODEL_NAMES[0]
latest = pr.sort_values("date").groupby("company").tail(1).set_index("company")
latest_ds = ds.sort_values("date").groupby("company").tail(1).set_index("company")


# ----------------------------------------------------------------------------- инвест-правило
def allocation(pd6, ytm_annual):
    """Доля капитала: дробный Келли для бинарного исхода на горизонте 6 мес.
    Выигрыш (нет дефолта): доходность за полгода сверх безрисковой. Проигрыш: LGD + безрисковая."""
    if pd6 is None or not np.isfinite(pd6):
        return np.nan, np.nan, "нет оценки"
    r6 = (1 + ytm_annual / 100) ** 0.5 - 1 if np.isfinite(ytm_annual) else np.nan
    rf6 = (1 + RISK_FREE_ANNUAL) ** 0.5 - 1
    exp_ret = (1 - pd6) * r6 - pd6 * LGD if np.isfinite(r6) else np.nan
    if pd6 >= 0.5:
        return 0.0, exp_ret, "Не инвестировать"
    if not np.isfinite(r6) or r6 <= rf6:
        return 0.0, exp_ret, "Не инвестировать: доходность не выше безрисковой"
    b, a = r6 - rf6, LGD + rf6
    kelly = (1 - pd6) / a - pd6 / b
    w = float(np.clip(KELLY_FRACTION * kelly, 0, MAX_WEIGHT))
    if w <= 0:
        verdict = "Не инвестировать: премия не покрывает ожидаемые потери"
    elif pd6 < 0.02:
        verdict = "Инвестировать"
    elif pd6 < 0.10:
        verdict = "Инвестировать ограниченно"
    else:
        verdict = "Только спекулятивно, малой долей"
    return w, exp_ret, verdict


def fmt_pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.{d}f}%"


# ----------------------------------------------------------------------------- шапка
st.title("Вероятность дефолта эмитентов облигаций РФ на горизонте 6 месяцев")
st.caption(f"Модели: MARS-логит и Z''-счёт Альтмана. Данные: MOEX ISS, ГИР БО ФНС, реестр дефолтов. "
           f"Последний срез: {pr.date.max():%d.%m.%Y}. Основная модель — «{MAIN}».")

tab1, tab2, tab3, tab4 = st.tabs(["🔎 Облигация / компания", "📋 Скринер", "⚖️ MARS vs Альтман",
                                  "✍️ Своя отчётность"])

# ============================================================================= 1. облигация
with tab1:
    show_def = st.toggle("Показывать эмитентов из реестра дефолтов", value=False)
    comp_list = latest.index[latest.segment != "Дефолт-реестр"] if not show_def else latest.index
    b = bonds[bonds.company.isin(comp_list)].copy()
    b["label"] = b.company + " — " + b.shortname.astype(str) + " (" + b["isin"].astype(str) + ")"
    b["active"] = b.matdate.isna() | (b.matdate >= pd.Timestamp.today())
    c1, c2 = st.columns([1, 2])
    sectors = ["Все"] + sorted(latest.loc[comp_list, "sector"].dropna().unique())
    sector = c1.selectbox("Отрасль", sectors)
    if sector != "Все":
        b = b[b.company.map(latest.sector) == sector]
    only_active = c1.checkbox("Только непогашенные выпуски", value=True)
    if only_active:
        b = b[b.active]
    default_idx = int(np.argmax(b.company.values == "Самолет")) if (b.company == "Самолет").any() else 0
    sel = c2.selectbox(f"Облигация ({len(b)} выпусков)", b.label.tolist(), index=default_idx if len(b) else None)
    if sel:
        bond = b[b.label == sel].iloc[0]
        comp = bond.company
        L, Ld = latest.loc[comp], latest_ds.loc[comp]
        h = bond_history(bond.secid)
        last_y = h["yield"].dropna().iloc[-1] if len(h) and h["yield"].notna().any() else np.nan
        last_p = h.price.iloc[-1] if len(h) else np.nan

        st.subheader(f"{comp} · {bond.shortname}")
        st.caption(f"{L.sector} · сегмент {L.segment} · ИНН {Ld.inn} · погашение "
                   f"{bond.matdate:%d.%m.%Y}" if pd.notna(bond.matdate) else "")
        m = st.columns(5)
        m[0].metric(f"PD 6м — {MAIN}", fmt_pct(L[MAIN]))
        m[1].metric("PD 6м — MARS (отчётность)", fmt_pct(L[MODEL_NAMES[2]]))
        m[2].metric("PD 6м — Альтман Z''", fmt_pct(L[ALT]))
        m[3].metric("Z''-счёт / зона", f"{L.z_altman:.2f}" if np.isfinite(L.z_altman) else "—",
                    altman_zone(L.z_altman), delta_color="off")
        m[4].metric("Цена / доходность", f"{last_p:.1f}%" if np.isfinite(last_p) else "—",
                    f"YTM {last_y:.1f}%" if np.isfinite(last_y) else None, delta_color="off")

        model_for_alloc = st.radio("Модель для инвест-решения", MODEL_NAMES, index=len(MODEL_NAMES) - 1,
                                   horizontal=True)
        w, er, verdict = allocation(L[model_for_alloc], last_y)
        st.info(f"**Рекомендация: {verdict}.** Доля капитала: **{fmt_pct(w)}** "
                f"(половина Келли, потолок {MAX_WEIGHT:.0%}, LGD {LGD:.0%}, безрисковая {RISK_FREE_ANNUAL:.1%} годовых). "
                f"Ожидаемая доходность за 6 мес. с учётом риска дефолта: {fmt_pct(er)}.")

        g1, g2 = st.columns(2)
        hist = pr[pr.company == comp].sort_values("date")
        fig = go.Figure()
        for name in MODEL_NAMES:
            fig.add_trace(go.Scatter(x=hist.date, y=hist[name] * 100, name=name, mode="lines+markers"))
        for ev in reg[reg.company == comp].itertuples():
            fig.add_vline(x=ev.event_date, line_dash="dot", line_color="red")
            fig.add_annotation(x=ev.event_date, y=1, yref="paper", text=ev.event_type, showarrow=False,
                               font=dict(color="red"))
        fig.update_layout(title="PD на 6 мес. по срезам (красные линии — техдефолт TD / дефолт D)",
                          yaxis_title="PD, %", height=380, legend=dict(orientation="h", y=-0.25))
        g1.plotly_chart(fig, width="stretch")
        if len(h):
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=h.tradedate, y=h.price, name="Цена, % номинала"))
            fig2.add_trace(go.Scatter(x=h.tradedate, y=h["yield"], name="Доходность, %", yaxis="y2"))
            fig2.update_layout(title=f"Котировки {bond.shortname}", height=380,
                               yaxis2=dict(overlaying="y", side="right", title="YTM, %"),
                               legend=dict(orientation="h", y=-0.25))
            g2.plotly_chart(fig2, width="stretch")

        st.markdown("**Признаки на последнюю дату среза**")
        feats = FIN_FEATS + DYN_FEATS + MKT_FEATS + ["rating_num", "had_event_12m"]
        tbl = pd.DataFrame({"Признак": [LABELS_RU.get(f, f) for f in feats],
                            "Значение": [Ld.get(f, np.nan) for f in feats]})
        tbl["Значение"] = tbl["Значение"].map(lambda v: "—" if pd.isna(v) else f"{v:,.3f}")
        st.caption(f"Отчётность: {Ld.get('fin_source', '—')}, период {str(Ld.get('fin_period_end', '—'))[:10]}")
        c3, c4 = st.columns(2)
        c3.dataframe(tbl, hide_index=True, width="stretch", height=420)
        mm = bundle["models"][-1]
        x = mm.prep.transform(ds[(ds.company == comp)].sort_values("date").tail(1))
        icpt, contr = mm.m.contributions(x[0])
        cdf = pd.DataFrame(contr, columns=["Базисная функция MARS", "Вклад в логит"])
        cdf = cdf[cdf["Вклад в логит"].abs() > 1e-6].sort_values("Вклад в логит", key=abs, ascending=False)
        c4.markdown(f"Разложение PD основной модели: свободный член {icpt:+.2f}; "
                    f"положительный вклад повышает вероятность дефолта")
        c4.dataframe(cdf, hide_index=True, width="stretch", height=380)

# ============================================================================= 2. скринер
with tab2:
    scr = latest.copy()
    scr["Цена мин., %"] = latest_ds.mkt_price_min
    scr["YTM медиан., %"] = latest_ds.mkt_yield_med
    scr["Зона Альтмана"] = scr.z_altman.map(altman_zone)
    al = scr.apply(lambda r: allocation(r[MAIN], r["YTM медиан., %"]), axis=1, result_type="expand")
    scr["Доля капитала"], scr["Решение"] = al[0], al[2]
    f1, f2, f3 = st.columns(3)
    segs = f1.multiselect("Сегмент", sorted(scr.segment.dropna().unique()),
                          default=[s for s in scr.segment.dropna().unique() if s != "Дефолт-реестр"])
    secs = f2.multiselect("Отрасль", sorted(scr.sector.dropna().unique()))
    sort_by = f3.selectbox("Сортировка", [MAIN] + MODEL_NAMES[:-1])
    v = scr[scr.segment.isin(segs)]
    if secs:
        v = v[v.sector.isin(secs)]
    v = v.sort_values(sort_by, ascending=False)
    cols = ["sector", "segment", "date"] + MODEL_NAMES + ["z_altman", "Зона Альтмана", "Цена мин., %",
                                                          "YTM медиан., %", "Доля капитала", "Решение"]
    show = v[cols].rename(columns={"sector": "Отрасль", "segment": "Сегмент", "date": "Срез", "z_altman": "Z''"})
    st.dataframe(show.style.format({**{n: "{:.1%}" for n in MODEL_NAMES}, "Доля капитала": "{:.1%}",
                                    "Z''": "{:.2f}", "Цена мин., %": "{:.1f}", "YTM медиан., %": "{:.1f}",
                                    "Срез": "{:%d.%m.%Y}"}, na_rep="—")
                 .background_gradient(subset=[MAIN], cmap="Reds", vmin=0, vmax=0.5),
                 width="stretch", height=600)
    pick = st.multiselect("Сравнить динамику PD компаний", v.index.tolist(),
                          default=[c for c in ["Самолет", "ЛСР", "Балтийский лизинг", "ДжетЛенд", "Лидер-Инвест"]
                                   if c in v.index])
    if pick:
        fig = go.Figure()
        for c in pick:
            hh = pr[pr.company == c].sort_values("date")
            fig.add_trace(go.Scatter(x=hh.date, y=hh[MAIN] * 100, name=c, mode="lines+markers"))
        fig.update_layout(yaxis_title=f"PD 6м, % ({MAIN})", height=420)
        st.plotly_chart(fig, width="stretch")

# ============================================================================= 3. сравнение
with tab3:
    st.markdown("Out-of-time проверка: модели обучены на срезах до конца 2024 г., тест — на более поздних "
                "срезах; Самолёт целиком в тесте. «Общая выборка» — строки, где у всех моделей есть оценка "
                "(есть отчётность), — честное сравнение.")
    st.dataframe(met.style.format({c: "{:.3f}" for c in ["ROC AUC", "Gini", "KS", "PR AUC", "Brier"]},
                                  na_rep="—"), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    fig = go.Figure()
    for name, r in roc.items():
        fig.add_trace(go.Scatter(x=r["fpr"], y=r["tpr"], name=name, mode="lines"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="grey"), showlegend=False))
    fig.update_layout(title="ROC-кривые на тесте (общая выборка)", xaxis_title="FPR", yaxis_title="TPR",
                      height=450, legend=dict(orientation="h", y=-0.2))
    c1.plotly_chart(fig, width="stretch")
    te = pr[pr.split == "test"]
    fig = go.Figure()
    for name in MODEL_NAMES:
        t = te[[name, "y"]].dropna()
        if len(t) < 20:
            continue
        t["bin"] = pd.qcut(t[name].rank(method="first"), 10, labels=False)
        g = t.groupby("bin").agg(p=(name, "mean"), y=("y", "mean"))
        fig.add_trace(go.Scatter(x=g.p * 100, y=g.y * 100, name=name, mode="lines+markers"))
    mx = float(te[MODEL_NAMES].max().max() * 100) if len(te) else 1
    fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], line=dict(dash="dash", color="grey"), showlegend=False))
    fig.update_layout(title="Калибровка на тесте: прогноз vs факт по децилям", xaxis_title="Средний PD, %",
                      yaxis_title="Доля дефолтов, %", height=450, legend=dict(orientation="h", y=-0.2))
    c2.plotly_chart(fig, width="stretch")
    with st.expander("Полный отчёт обучения (уравнения MARS, важность признаков, бутстрэп ΔAUC)"):
        st.code(rep)

# ============================================================================= 4. своя отчётность
with tab4:
    st.markdown("Введите строки РСБУ (тыс. руб.) за отчётный и предыдущий год — модель посчитает "
                "коэффициенты и вероятность дефолта. Рыночные поля необязательны.")
    lines = [("1600", "Баланс (итого активы)"), ("1200", "Оборотные активы"), ("1210", "Запасы"),
             ("1250", "Денежные средства"), ("1240", "Краткоср. фин. вложения"), ("1300", "Капитал и резервы"),
             ("1370", "Нераспределённая прибыль"), ("1400", "Долгосрочные обязательства"),
             ("1410", "Долгосрочные заёмные средства"), ("1500", "Краткосрочные обязательства"),
             ("1510", "Краткосрочные заёмные средства"), ("2110", "Выручка"), ("2200", "Прибыль от продаж"),
             ("2300", "Прибыль до налогообложения"), ("2330", "Проценты к уплате"), ("2400", "Чистая прибыль")]
    base = st.selectbox("Подставить отчётность компании из базы (можно отредактировать)",
                        ["—"] + sorted(latest.index))
    fin = pd.read_csv(RAW / "financials.csv") if (RAW / "financials.csv").exists() else pd.DataFrame()
    pre_c, pre_p = {}, {}
    if base != "—" and len(fin):
        ff = fin[fin.company == base].sort_values("year")
        if len(ff):
            pre_c = ff.iloc[-1].to_dict()
            pre_p = ff.iloc[-2].to_dict() if len(ff) > 1 else {}
    with st.form("manual"):
        cA, cB, cC = st.columns(3)
        cur, prev = {}, {}
        cA.markdown("**Отчётный год**")
        cB.markdown("**Предыдущий год**")
        for code, title in lines:
            dv = pre_c.get(f"c{code}")
            cur[f"c{code}"] = cA.number_input(f"{code} {title}", value=float(dv) if pd.notna(dv) else 0.0,
                                              key=f"c{code}_{base}", format="%.0f")
            dp = pre_p.get(f"c{code}")
            prev[f"c{code}"] = cB.number_input(f"{code} {title} ", value=float(dp) if pd.notna(dp) else 0.0,
                                               key=f"p{code}_{base}", format="%.0f")
        cC.markdown("**Рынок (необязательно)**")
        price = cC.number_input("Мин. цена облигаций, % номинала (0 — нет)", value=0.0)
        ytm = cC.number_input("Доходность к погашению, % (0 — нет)", value=0.0)
        chg = cC.number_input("Изменение цены за 3 мес., %", value=0.0)
        had = cC.checkbox("Были техдефолты за последние 12 мес.")
        go_btn = st.form_submit_button("Рассчитать PD")
    if go_btn:
        rc = ratios(cur)
        has_prev = any(v for v in prev.values())
        rp = ratios(prev) if has_prev else None
        row = {**{k: rc[k] for k in FIN_FEATS}, "z_altman": rc["z_altman"],
               **dynamics(rc, rp, cur, prev if has_prev else None)}
        mkt_med = ds.groupby("date").mkt_yield_med.median().iloc[-1] if ds.mkt_yield_med.notna().any() else np.nan
        row.update({"mkt_price_min": price or np.nan, "mkt_price_med": price or np.nan,
                    "mkt_yield_max": ytm or np.nan, "mkt_yield_med": ytm or np.nan,
                    "mkt_spread": (ytm - mkt_med) if ytm else np.nan,
                    "mkt_price_chg_3m": chg / 100 if price else np.nan, "had_event_12m": float(had)})
        one = pd.DataFrame([row])
        out = {m.name: float(m.predict(one)[0]) for m in bundle["models"]}
        k = st.columns(len(out) + 1)
        for i, (n, p) in enumerate(out.items()):
            k[i].metric(n, fmt_pct(p))
        k[-1].metric("Z''-счёт", f"{rc['z_altman']:.2f}" if np.isfinite(rc["z_altman"]) else "—",
                     altman_zone(rc["z_altman"]), delta_color="off")
        w, er, verdict = allocation(out[MAIN], ytm if ytm else np.nan)
        st.info(f"**{verdict}.** Доля капитала: {fmt_pct(w)}; ожидаемая доходность за 6 мес.: {fmt_pct(er)}")
        st.dataframe(pd.DataFrame({"Коэффициент": [LABELS_RU.get(k2, k2) for k2 in row],
                                   "Значение": [f"{v:,.3f}" if pd.notna(v) else "—" for v in row.values()]}),
                     hide_index=True, width="stretch")

st.caption(f"PD — вероятность дефолта в течение {HORIZON_DAYS} дней. Учебная модель, не является "
           "индивидуальной инвестиционной рекомендацией.")
