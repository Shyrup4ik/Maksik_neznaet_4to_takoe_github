"""Шаг 3. Сборка панели «компания × квартал» с признаками и целевой переменной.

Логика без заглядывания в будущее:
  * дата среза t — конец каждого квартала (+ «сегодняшний» срез по последним торгам);
  * отчётность берётся только та, что была опубликована к дате t
    (годовая РСБУ за год Y — с 1 апреля Y+1; ручная — по publish_date);
  * рыночные признаки — только по сделкам до t включительно;
  * целевая переменная y = 1, если в интервале (t, t + 6 мес.] у эмитента случился дефолт
    (по TARGET_MODE в config.py), иначе 0. Если t + 6 мес. ещё не наступило — метка неизвестна
    (такие строки только скорингуются, в обучении/тесте не участвуют);
  * строки после реального дефолта эмитента отбрасываются (компания уже в дефолте).

Выход: data/dataset.csv
"""
import numpy as np
import pandas as pd

from config import (DATA, RAW, START_DATE, TODAY, HORIZON_DAYS, TARGET_MODE, ANNUAL_REPORT_LAG_DAYS,
                    MANUAL_REPORT_LAG_DAYS, MARKET_WINDOW_DAYS, MAX_STALE_PRICE_DAYS)
from features import ratios, dynamics, rating_to_num, FIN_FEATS, DYN_FEATS


def load_financials():
    frames = []
    f = RAW / "financials.csv"
    if f.exists():
        a = pd.read_csv(f)
        a["period_end"] = pd.to_datetime(a.year.astype(int).astype(str) + "-12-31")
        a["publish_date"] = a.period_end + pd.Timedelta(days=ANNUAL_REPORT_LAG_DAYS)
        a["freq"] = "Y"
        frames.append(a)
    m = DATA / "financials_manual.csv"
    if m.exists():
        b = pd.read_csv(m)
        if len(b):
            b["period_end"] = pd.to_datetime(b.period_end)
            pub = pd.to_datetime(b.get("publish_date"), errors="coerce")
            b["publish_date"] = pub.fillna(b.period_end + pd.Timedelta(days=MANUAL_REPORT_LAG_DAYS))
            b["freq"] = "M"
            b["source"] = b.get("source", "ручной ввод")
            frames.append(b)
    if not frames:
        raise SystemExit("Нет отчётности: запустите collect_girbo.py или заполните data/financials_manual.csv")
    fin = pd.concat(frames, ignore_index=True)
    return {c: g.sort_values("period_end") for c, g in fin.groupby("company")}


def pick_report(g, t):
    """Последний опубликованный к t отчёт и отчёт годом ранее (для динамики)."""
    if g is None:
        return None, None
    avail = g[g.publish_date <= t]
    if avail.empty:
        return None, None
    cur = avail.iloc[-1]
    target = cur.period_end - pd.Timedelta(days=365)
    prev = avail[(avail.period_end - target).abs() <= pd.Timedelta(days=45)]
    return cur, (prev.iloc[-1] if len(prev) else None)


def load_market():
    bonds = pd.read_csv(RAW / "bonds.csv")
    for c in ("issuedate", "matdate"):
        bonds[c] = pd.to_datetime(bonds[c], errors="coerce")
    hist = {}
    for secid in bonds.secid.unique():
        f = RAW / "history" / f"{secid}.csv"
        if f.exists():
            h = pd.read_csv(f, parse_dates=["tradedate"])
            if len(h):
                hist[secid] = h.sort_values("tradedate")
    return bonds, hist


def market_feats(bonds_c, hist, t):
    out = {"n_bonds_outstanding": 0}
    live = bonds_c[(bonds_c.issuedate.isna() | (bonds_c.issuedate <= t)) &
                   (bonds_c.matdate.isna() | (bonds_c.matdate >= t))]
    out["n_bonds_outstanding"] = len(live)
    prices, ylds, chg, vol, turn = [], [], [], [], 0.0
    for b in live.itertuples():
        h = hist.get(b.secid)
        if h is None:
            continue
        w = h[(h.tradedate <= t) & (h.tradedate > t - pd.Timedelta(days=MARKET_WINDOW_DAYS))]
        if w.empty or (t - w.tradedate.iloc[-1]).days > MAX_STALE_PRICE_DAYS:
            continue
        # бумаги за месяц до погашения искажают доходность — пропускаем
        if pd.notna(b.matdate) and (b.matdate - t).days < 30:
            continue
        prices.append(w.price.iloc[-1])
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
                   mkt_log_turnover=np.log1p(turn))
    return out


def load_ratings():
    f = DATA / "ratings_manual.csv"
    if not f.exists():
        return {}
    r = pd.read_csv(f)
    if r.empty:
        return {}
    r["date"] = pd.to_datetime(r.date)
    r["num"] = r.rating.map(rating_to_num)
    return {c: g.sort_values("date") for c, g in r.groupby("company")}


def rating_feats(g, t):
    if g is None:
        return {"rating_num": np.nan, "rating_chg_6m": np.nan}

    def worst_at(d):
        a = g[g.date <= d]
        if a.empty:
            return np.nan
        last = a.groupby("agency").tail(1)        # последний рейтинг от каждого агентства
        last = last[last.date > d - pd.Timedelta(days=550)]  # рейтинг отозван/устарел
        return last.num.max() if len(last) else np.nan
    now, before = worst_at(t), worst_at(t - pd.Timedelta(days=182))
    return {"rating_num": now, "rating_chg_6m": (now - before) if np.isfinite(before) else 0.0}


def main():
    em = pd.read_csv(RAW / "emitters.csv")
    em = em[em.status == "ok"]
    fin = load_financials()
    bonds, hist = load_market()
    rats = load_ratings()
    reg = pd.read_csv(DATA / "defaults_registry.csv", parse_dates=["event_date"])
    target_events = reg if TARGET_MODE == "broad" else reg[reg.event_type == "D"]

    dates = list(pd.date_range(START_DATE, TODAY, freq="QE"))
    last_trade = max((h.tradedate.max() for h in hist.values()), default=dates[-1])
    snapshot = pd.Timestamp(last_trade).normalize()
    if snapshot > dates[-1]:
        dates.append(snapshot)

    rows = []
    for e in em.itertuples():
        c = e.company
        bc = bonds[bonds.company == c]
        ev_all = reg[reg.company == c]
        ev_t = target_events[target_events.company == c]
        first_D = ev_all[ev_all.event_type == "D"].event_date.min()
        for t in dates:
            is_snap = t == snapshot and snapshot not in pd.date_range(START_DATE, TODAY, freq="QE")
            if pd.notna(first_D) and t >= first_D and not is_snap:
                continue
            mk = market_feats(bc, hist, t)
            if mk["n_bonds_outstanding"] == 0 and not is_snap:
                continue
            cur, prev = pick_report(fin.get(c), t)
            rec = {"company": c, "date": t, "sector": e.sector, "segment": e.segment,
                   "inn": e.inn, "is_snapshot": int(is_snap)}
            if cur is not None:
                rc = ratios(cur.to_dict())
                rp = ratios(prev.to_dict()) if prev is not None else None
                rec |= {k: rc[k] for k in FIN_FEATS} | {"z_altman": rc["z_altman"]}
                rec |= dynamics(rc, rp, cur.to_dict(), prev.to_dict() if prev is not None else None)
                rec |= {"fin_period_end": cur.period_end, "fin_source": cur.get("source"),
                        "fin_age_days": (t - cur.period_end).days}
            rec |= mk
            rec |= rating_feats(rats.get(c), t)
            past = ev_all[(ev_all.event_date <= t) & (ev_all.event_date > t - pd.Timedelta(days=365))]
            rec["had_event_12m"] = float(len(past) > 0)
            horizon_end = t + pd.Timedelta(days=HORIZON_DAYS)
            fut = ev_t[(ev_t.event_date > t) & (ev_t.event_date <= horizon_end)]
            rec["label_known"] = int(horizon_end <= TODAY and not is_snap)
            rec["y"] = int(len(fut) > 0) if rec["label_known"] else np.nan
            rec["next_event"] = fut.event_date.min() if len(fut) else pd.NaT
            rows.append(rec)

    df = pd.DataFrame(rows)
    for col in FIN_FEATS + DYN_FEATS:
        if col not in df:
            df[col] = np.nan
    # спред доходности к медиане всех облигаций в базе на ту же дату
    df["mkt_spread"] = df.mkt_yield_med - df.groupby("date").mkt_yield_med.transform("median")
    df.sort_values(["company", "date"]).to_csv(DATA / "dataset.csv", index=False)
    lab = df[df.label_known == 1]
    print(f"Строк: {len(df)}; с известной меткой: {len(lab)}; дефолтных (y=1): {int(lab.y.sum())}; "
          f"компаний: {df.company.nunique()}; доля с отчётностью: {df.current_ratio.notna().mean():.0%}; "
          f"с рыночными данными: {df.mkt_price_med.notna().mean():.0%}")


if __name__ == "__main__":
    main()
