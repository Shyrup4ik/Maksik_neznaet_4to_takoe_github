"""Шаг 3. Панель «эмитент × квартал»: признаки на дату среза и целевая переменная.

Ключ эмитента — ИНН. Для каждой строки сохраняются даты, по которым её можно проверить:
  date            — дата среза t (конец квартала; плюс «текущий» срез по последним торгам);
  window_end      — конец окна метки t + 6 мес.;
  fin_period_end / fin_publish_date / fin_source — какой отчёт использован и когда он стал доступен;
  quote_date      — дата последней использованной котировки;
  label_status    — known / unknown_open_window / unknown_no_coverage / in_default.

Отчётность:
  * берётся только отчёт, опубликованный не позже t; из нескольких версий одного периода —
    последняя опубликованная к t (исправления учитываются, будущие версии — нет);
  * промежуточная отчётность (3/6/9 мес. нарастающим итогом) переводится в «последние 12 мес.»:
    LTM = текущий период + прошлый год − тот же период прошлого года; если нужных отчётов нет —
    годовая оценка ×12/m (fin_flow_basis = annualized);
  * динамика считается к отчёту того же стандарта, периметра и длины периода годом ранее.

Целевая переменная (TARGET_MODE в config.py):
  * broad  — дата первого события эпизода (техпросрочка или дефолт);
  * strict — дата перехода в полноценный дефолт (event_type = D);
  y = 1, если такое событие попадает в (t, t + 6 мес.]. Метка ставится, только если всё окно
  лежит в периоде полного покрытия реестра (REGISTRY_COVERAGE_START) и уже завершилось к TODAY.
Эмитент в активном дефолте (после D и до даты исправления cure_date) из обычных срезов
исключается; в текущем срезе он остаётся со статусом in_default (PD не оценивается).

Выход: data/dataset.csv, data/coverage_report.csv, data/registry_summary.csv
"""
import numpy as np
import pandas as pd

from config import (DATA, RAW, START_DATE, TODAY, TARGET_MODE, REGISTRY_COVERAGE_START, POPULATION,
                    ANNUAL_REPORT_LAG_DAYS, INTERIM_REPORT_LAG_DAYS, IFRS_ANNUAL_LAG_DAYS,
                    MARKET_WINDOW_DAYS, MAX_STALE_PRICE_DAYS, horizon_end)
from features import (ratios, dynamics, rating_to_num, missing_lines, FIN_FEATS, DYN_FEATS, MKT_FEATS,
                      RAT_FEATS, FLOW_CODES)

LINE_COLS = [f"c{c}" for c in ["1100", "1150", "1200", "1210", "1230", "1240", "1250", "1300", "1370", "1400",
                                "1410", "1500", "1510", "1520", "1600"] + FLOW_CODES]


def _inn(x):
    s = "".join(ch for ch in str(x if x is not None else "").split(".")[0] if ch.isdigit())
    return s if len(s) in (10, 12) else ""


# =============================================================================== отчётность
def normalize_reports(df, kind, name_to_inn=None):
    """Приводит таблицу отчётов (ГИР БО или ручную) к единому виду. Понятно падает,
    если нет обязательных полей; отсутствующие необязательные — заполняет по правилам."""
    df = df.copy()
    if "period_end" not in df and "year" in df:
        df["period_end"] = df.year.astype(int).astype(str) + "-12-31"
    missing = [c for c in ("period_end",) if c not in df]
    if missing or not ({"inn", "company"} & set(df.columns)):
        raise ValueError(f"{kind}: нет обязательных колонок {missing or ['inn или company']}")
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    bad = df.period_end.isna()
    if bad.any():
        print(f"! {kind}: {bad.sum()} строк без корректной period_end — пропущены")
        df = df[~bad]
    df["inn"] = df.get("inn", pd.Series("", index=df.index)).map(_inn)
    if name_to_inn is not None and "company" in df:
        df.loc[df.inn == "", "inn"] = df.loc[df.inn == "", "company"].map(name_to_inn).fillna("")
    if (df.inn == "").any():
        print(f"! {kind}: {int((df.inn == '').sum())} строк без ИНН (компания не найдена в emitters.csv) — пропущены")
        df = df[df.inn != ""]
    df["period_months"] = pd.to_numeric(df.get("period_months"), errors="coerce") \
        if "period_months" in df else np.nan
    df["period_months"] = df.period_months.fillna(df.period_end.dt.month).astype(int)
    df["standard"] = df.get("standard", pd.Series("РСБУ", index=df.index)).fillna("РСБУ").astype(str).str.upper() \
        .replace({"IFRS": "МСФО", "RAS": "РСБУ"})
    df["perimeter"] = df.get("perimeter", pd.Series(np.nan, index=df.index))
    df["perimeter"] = df.perimeter.fillna(df.standard.map({"МСФО": "группа"}).fillna("юрлицо"))
    lag = np.where(df.period_months < 12, INTERIM_REPORT_LAG_DAYS,
                   np.where(df.standard == "МСФО", IFRS_ANNUAL_LAG_DAYS, ANNUAL_REPORT_LAG_DAYS))
    pub = pd.to_datetime(df["publish_date"], errors="coerce") if "publish_date" in df \
        else pd.Series(pd.NaT, index=df.index)
    est = df.period_end + pd.to_timedelta(lag, unit="D")
    df["publish_date_source"] = np.where(pub.notna(), df.get("publish_date_source", "указана в источнике"),
                                         "оценка по сроку раскрытия")
    df["publish_date"] = pub.fillna(est)
    for c in LINE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    df["source"] = df.get("source", pd.Series(kind, index=df.index)).fillna(kind)
    df["source_url"] = df.get("source_url", pd.Series("", index=df.index))
    return df[["inn", "period_end", "period_months", "standard", "perimeter", "publish_date",
               "publish_date_source", "source", "source_url"] + LINE_COLS]


def load_financials(name_to_inn):
    frames = []
    f = RAW / "financials.csv"
    if f.exists() and f.stat().st_size > 10:
        frames.append(normalize_reports(pd.read_csv(f, dtype={"inn": str}), "ГИР БО", name_to_inn))
    m = DATA / "financials_manual.csv"
    if m.exists():
        b = pd.read_csv(m, dtype={"inn": str})
        if len(b):
            frames.append(normalize_reports(b, "ручной ввод", name_to_inn))
    if not frames:
        print("! Отчётности нет: финансовые признаки будут пустыми (запустите collect_girbo.py "
              "или заполните data/financials_manual.csv)")
        return {}
    fin = pd.concat(frames, ignore_index=True)
    return {k: g.sort_values(["period_end", "publish_date"]) for k, g in fin.groupby("inn")}


def _same_kind(av, rep):
    return av[(av.standard == rep.standard) & (av.perimeter == rep.perimeter)]


def _latest_version(av, period_end, months):
    x = av[(av.period_end == period_end) & (av.period_months == months)]
    return x.iloc[-1] if len(x) else None


def ltm(rep, av):
    """Потоки за 12 мес. для отчёта rep по отчётам, доступным на дату среза (av)."""
    out = rep.to_dict()
    m = int(rep.period_months)
    if m >= 12:
        out["fin_flow_basis"] = "12 мес."
        return out
    same = _same_kind(av, rep)
    fy = _latest_version(same, pd.Timestamp(f"{rep.period_end.year - 1}-12-31"), 12)
    py = _latest_version(same, rep.period_end - pd.DateOffset(years=1), m)
    if fy is not None and py is not None:
        for c in FLOW_CODES:
            out[f"c{c}"] = rep[f"c{c}"] + fy[f"c{c}"] - py[f"c{c}"]
        out["fin_flow_basis"] = "LTM"
    else:
        for c in FLOW_CODES:
            out[f"c{c}"] = rep[f"c{c}"] * 12 / m
        out["fin_flow_basis"] = f"×12/{m} (нет отчётов для LTM)"
    return out


PRIORITY = {("МСФО", "группа"): 0, ("РСБУ", "юрлицо"): 1}


def pick_report(g, t):
    """→ (текущий отчёт с потоками за 12 мес., отчёт годом ранее того же вида) по данным на t."""
    if g is None:
        return None, None
    av = g[g.publish_date <= t]
    if av.empty:
        return None, None
    last_pe = av.period_end.max()
    cand = av[av.period_end == last_pe].copy()
    cand["prio"] = [PRIORITY.get((s, p), 2) for s, p in zip(cand.standard, cand.perimeter)]
    # предпочтение — консолидированная МСФО группы; внутри вида — последняя опубликованная версия
    rep = cand.sort_values(["prio", "publish_date"], ascending=[True, False]).iloc[0]
    same = _same_kind(av, rep)
    prev = _latest_version(same, rep.period_end - pd.DateOffset(years=1), int(rep.period_months))
    return ltm(rep, av), (ltm(prev, av) if prev is not None else None)


# =============================================================================== рынок
def load_market():
    f = RAW / "bonds.csv"
    if not f.exists() or f.stat().st_size < 10:
        return pd.DataFrame(columns=["secid", "inn", "issuedate", "matdate"]), {}
    bonds = pd.read_csv(f, dtype={"inn": str})
    bonds["inn"] = bonds.inn.map(_inn)
    for c in ("issuedate", "matdate"):
        bonds[c] = pd.to_datetime(bonds[c], errors="coerce")
    hist = {}
    for secid in bonds.secid.unique():
        h = RAW / "history" / f"{secid}.csv"
        if h.exists():
            x = pd.read_csv(h, parse_dates=["tradedate"])
            if len(x):
                hist[secid] = x.sort_values("tradedate")
    return bonds, hist


def market_feats(bonds_c, hist, t):
    out = {k: np.nan for k in MKT_FEATS} | {"quote_date": pd.NaT}
    live = bonds_c[(bonds_c.issuedate.isna() | (bonds_c.issuedate <= t)) &
                   (bonds_c.matdate.isna() | (bonds_c.matdate >= t))]
    out["n_bonds_outstanding"] = len(live)
    prices, ylds, chg, vol, turn, qd = [], [], [], [], 0.0, []
    for b in live.itertuples():
        h = hist.get(b.secid)
        if h is None:
            continue
        w = h[(h.tradedate <= t) & (h.tradedate > t - pd.Timedelta(days=MARKET_WINDOW_DAYS))]
        if w.empty or (t - w.tradedate.iloc[-1]).days > MAX_STALE_PRICE_DAYS:
            continue
        if pd.notna(b.matdate) and (b.matdate - t).days < 30:
            continue
        prices.append(w.price.iloc[-1])
        qd.append(w.tradedate.iloc[-1])
        y = w["yield"].iloc[-1]
        if pd.notna(y) and y > 0:
            ylds.append(min(y, 500.0))
        if len(w) >= 5:
            chg.append(w.price.iloc[-1] / w.price.iloc[0] - 1)
            vol.append(w.price.pct_change().std())
        turn += np.nansum(w["value"])
    if prices:
        out.update(mkt_price_min=min(prices), mkt_price_med=float(np.median(prices)),
                   mkt_yield_max=max(ylds) if ylds else np.nan,
                   mkt_yield_med=float(np.median(ylds)) if ylds else np.nan,
                   mkt_price_chg_3m=float(np.median(chg)) if chg else np.nan,
                   mkt_vol_3m=float(np.median(vol)) if vol else np.nan,
                   mkt_log_turnover=np.log1p(turn), quote_date=max(qd))
    return out


# =============================================================================== рейтинги
def load_ratings(name_to_inn):
    f = DATA / "ratings_manual.csv"
    if not f.exists():
        return {}
    r = pd.read_csv(f, dtype={"inn": str})
    if r.empty:
        return {}
    r["inn"] = r.get("inn", pd.Series("", index=r.index)).map(_inn)
    r.loc[r.inn == "", "inn"] = r.loc[r.inn == "", "company"].map(name_to_inn).fillna("")
    r["date"] = pd.to_datetime(r.date)
    r["num"] = r.rating.map(rating_to_num)      # WD (отзыв) → NaN, отменяет прежний рейтинг агентства
    return {k: g.sort_values("date") for k, g in r.groupby("inn") if k}


def rating_feats(g, t):
    if g is None:
        return {"rating_num": np.nan, "rating_chg_6m": np.nan}

    def worst_at(d):
        a = g[g.date <= d]
        if a.empty:
            return np.nan
        last = a.groupby("agency").tail(1)
        last = last[last.date > d - pd.Timedelta(days=550)]
        return last.num.max() if last.num.notna().any() else np.nan
    now, before = worst_at(t), worst_at(t - pd.Timedelta(days=182))
    return {"rating_num": now, "rating_chg_6m": (now - before) if np.isfinite(before) and np.isfinite(now) else
            (0.0 if np.isfinite(now) else np.nan)}


# =============================================================================== события и метки
def load_events(name_to_inn):
    reg = pd.read_csv(DATA / "defaults_registry.csv", dtype={"inn": str}, keep_default_na=False)
    reg["inn"] = reg.inn.map(_inn)
    reg.loc[reg.inn == "", "inn"] = reg.loc[reg.inn == "", "company"].map(name_to_inn).fillna("")
    for c in ("event_date", "due_date", "cure_date"):
        reg[c] = pd.to_datetime(reg[c].replace("", None), errors="coerce")
    reg["cured"] = pd.to_numeric(reg.cured, errors="coerce").fillna(0).astype(int)
    if "episode_id" not in reg or (reg.episode_id == "").any():
        reg["episode_id"] = reg.company + "#" + reg.event_date.dt.strftime("%Y%m%d")
    return reg


def target_events(reg, mode=TARGET_MODE):
    """Даты событий, которые считаются целевыми: по одной на эпизод."""
    if mode == "strict":
        e = reg[reg.event_type == "D"].groupby("episode_id").agg(inn=("inn", "first"), date=("event_date", "min"))
    else:
        e = reg.groupby("episode_id").agg(inn=("inn", "first"), date=("event_date", "min"))
    return e.reset_index()


def default_intervals(reg, mode=TARGET_MODE):
    """Интервалы «эмитент в неплатеже» [начало, дата исправления) по эмитенту; срезы внутри не используются.

    Начало интервала: strict — дата D; broad — дата ПЕРВОГО события эпизода (TD), если в эпизоде
    потом наступил D. Иначе срез между TD и D получал бы y=0 (первое событие эпизода уже в прошлом),
    хотя полноценный дефолт этого же эпизода ещё впереди."""
    d = reg[reg.event_type == "D"]
    first = reg.groupby("episode_id").event_date.min() if "episode_id" in reg else None
    out = {}
    for r in d.itertuples():
        end = r.cure_date if pd.notna(r.cure_date) else pd.Timestamp.max
        start = r.event_date
        if mode == "broad" and first is not None:
            start = min(start, first.get(r.episode_id, start))
        out.setdefault(r.inn, []).append((start, end))
    return out


def in_default(intervals, t):
    return any(a <= t < b for a, b in intervals)


def label(t, ev_dates, coverage_start=REGISTRY_COVERAGE_START, today=TODAY):
    """→ (label_status, y, window_end, первое событие в окне)."""
    we = horizon_end(t)
    fut = sorted(d for d in ev_dates if t < d <= we)
    nxt = fut[0] if fut else pd.NaT
    if t < pd.Timestamp(coverage_start):
        return "unknown_no_coverage", np.nan, we, nxt
    if we > today:
        return "unknown_open_window", np.nan, we, nxt
    return "known", int(bool(fut)), we, nxt


# =============================================================================== сборка
def snapshot_dates(hist):
    q = list(pd.date_range(START_DATE, TODAY, freq="QE"))
    last_trade = max((h.tradedate.max() for h in hist.values()), default=None)
    snap = pd.Timestamp(last_trade).normalize() if last_trade is not None else TODAY
    snap = min(snap, TODAY)
    if not q or snap > q[-1]:
        q.append(snap)
    return q, snap


def main():
    em = pd.read_csv(RAW / "emitters.csv", dtype={"inn": str})
    em = em[em.status.isin(["ok", "ok_name_match"])].copy()
    em["inn"] = em.inn.map(_inn)
    if "origin" not in em:
        em["origin"] = np.where(em.segment.eq("Реестр событий") | em.segment.eq("Дефолт-реестр"), "registry", "universe")
    keep = {"universe": ["universe"], "universe+registry": ["universe", "registry"],
            "all": ["universe", "registry", "population"]}[POPULATION]
    em = em[em.origin.isin(keep)].drop_duplicates("inn")
    name_to_inn = dict(zip(em.company, em.inn))

    fin = load_financials(name_to_inn)
    bonds, hist = load_market()
    rats = load_ratings(name_to_inn)
    reg = load_events(name_to_inn)
    tev = target_events(reg)
    dint = default_intervals(reg)
    dates, snap = snapshot_dates(hist)

    rows = []
    for e in em.itertuples():
        inn = e.inn
        bc = bonds[bonds.inn == inn]
        evd = list(tev.loc[tev.inn == inn, "date"])
        reg_c = reg[reg.inn == inn]
        ints = dint.get(inn, [])
        for t in dates:
            is_snap = t == snap
            dflt = in_default(ints, t)
            if dflt and not is_snap:
                continue
            mk = market_feats(bc, hist, t)
            if mk["n_bonds_outstanding"] == 0 and not is_snap:
                continue
            cur, prev = pick_report(fin.get(inn), t)
            rec = {"inn": inn, "company": e.company, "date": t, "sector": e.sector, "segment": e.segment,
                   "origin": e.origin, "is_snapshot": int(is_snap)}
            if cur is not None:
                rc = ratios(cur)
                rp = ratios(prev) if prev is not None else None
                rec |= {k: rc[k] for k in FIN_FEATS} | {"z_altman": rc["z_altman"]}
                rec |= dynamics(rc, rp, cur, prev)
                rec |= {"fin_period_end": cur["period_end"], "fin_publish_date": cur["publish_date"],
                        "fin_publish_date_source": cur["publish_date_source"], "fin_months": cur["period_months"],
                        "fin_standard": cur["standard"], "fin_perimeter": cur["perimeter"],
                        "fin_flow_basis": cur["fin_flow_basis"], "fin_source": cur["source"],
                        "fin_source_url": cur.get("source_url", ""),
                        "fin_age_days": (t - cur["period_end"]).days,
                        "fin_missing_lines": "; ".join(missing_lines(cur))}
            rec |= mk
            rec |= rating_feats(rats.get(inn), t)
            past = reg_c[(reg_c.event_date <= t) & (reg_c.event_date > t - pd.Timedelta(days=365))]
            rec["had_event_12m"] = float(len(past) > 0)
            st, y, we, nxt = label(t, evd)
            if dflt:
                st, y = "in_default", np.nan
            rec |= {"label_status": st, "y": y, "window_end": we, "next_event": nxt}
            rows.append(rec)

    df = pd.DataFrame(rows)
    for col in FIN_FEATS + DYN_FEATS + MKT_FEATS + RAT_FEATS + ["z_altman", "fin_period_end", "fin_months",
                                                                 "quote_date", "fin_missing_lines", "fin_age_days",
                                                                 "fin_publish_date", "fin_source", "fin_standard",
                                                                 "fin_perimeter", "fin_flow_basis"]:
        if col not in df:
            df[col] = np.nan
    df["mkt_spread"] = df.mkt_yield_med - df.groupby("date").mkt_yield_med.transform("median")
    df = df.sort_values(["company", "date"])
    df.to_csv(DATA / "dataset.csv", index=False)

    # --- сводка покрытия по компаниям ---
    reg_rows = df[df.is_snapshot == 0]
    cov = reg_rows.groupby(["inn", "company"]).agg(
        срезов=("date", "size"),
        доля_с_отчётностью=("fin_period_end", lambda s: s.notna().mean()),
        доля_квартальной=("fin_months", lambda s: (s < 12).mean()),
        доля_свежей_отчётности=("fin_age_days", lambda s: (s <= 183).mean()),
        доля_с_котировками=("mkt_price_med", lambda s: s.notna().mean()),
        доля_с_рейтингом=("rating_num", lambda s: s.notna().mean()),
        меток_известно=("label_status", lambda s: (s == "known").sum()),
    ).reset_index()
    cov.to_csv(DATA / "coverage_report.csv", index=False)

    # --- реестр по годам ---
    reg["year"] = reg.event_date.dt.year
    reg["в_панели"] = reg.inn.isin(set(df.inn))
    rs = reg.groupby("year").agg(событий=("event_date", "size"), эпизодов=("episode_id", "nunique"),
                                 эмитентов=("company", "nunique"), из_них_в_панели=("в_панели", "mean"),
                                 проверено_по_первоисточнику=("verified", lambda s: (s.astype(str) == "1").mean()))
    pan = reg_rows.assign(year=reg_rows.date.dt.year).groupby("year").agg(
        эмитентов_в_панели=("inn", "nunique"), строк=("date", "size"))
    nb = bonds.copy()
    yrs = range(int(pd.Timestamp(START_DATE).year), TODAY.year + 1)
    pan["выпусков_в_обращении"] = [((nb.issuedate.isna() | (nb.issuedate <= f"{y}-12-31")) &
                                     (nb.matdate.isna() | (nb.matdate >= f"{y}-01-01"))).sum() for y in pan.index] \
        if len(pan) else []
    summ = pan.join(rs, how="outer").reindex(yrs)
    summ.to_csv(DATA / "registry_summary.csv")

    lab = df[df.label_status == "known"]
    print(f"Строк: {len(df)}; эмитентов: {df.inn.nunique()}; меток известно: {len(lab)} "
          f"(y=1: {int(lab.y.sum())}); статусы: {df.label_status.value_counts().to_dict()}")
    print(f"Доля срезов с отчётностью: {reg_rows.fin_period_end.notna().mean():.0%}, "
          f"из них с квартальной: {(reg_rows.fin_months < 12).mean():.0%}; "
          f"с котировками: {reg_rows.mkt_price_med.notna().mean():.0%}")
    print("Покрытие реестра и панели по годам → data/registry_summary.csv; по компаниям → data/coverage_report.csv")
    if (reg_rows.fin_months < 12).mean() < 0.5:
        print("ВНИМАНИЕ: квартальная отчётность есть менее чем у половины срезов — внутри года признаки "
              "отчётности повторяют годовой отчёт. Дополните data/financials_manual.csv или согласуйте "
              "использование годовой РСБУ как отступление от ТЗ.")


if __name__ == "__main__":
    main()
