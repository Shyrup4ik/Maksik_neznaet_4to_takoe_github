"""Шаг 1. Эмитенты, выпуски (включая погашенные) и котировки с Московской биржи (MOEX ISS).

Ключ эмитента — ИНН. Название используется только для поиска и показа.

Что делает:
  1) находит эмитента: если в universe.csv задан ИНН — берётся ТОЛЬКО кандидат с этим ИНН
     (иначе статус not_found_inn и список кандидатов в журнале); без ИНН — по названию,
     со статусом ok_name_match (ИНН нужно проверить);
  2) собирает выпуски из трёх источников и объединяет их:
       a) /emitters/{id}/securities — текущий список эмитента;
       b) поиск /securities?q=… по названию и по стеблям кратких названий бумаг;
       c) полный исторический листинг рынка облигаций (/history/…/listing) —
          там есть и погашенные, и снятые с торгов бумаги;
     каждый кандидат подтверждается по ИНН эмитента из карточки поиска;
     плюс ручные дополнения data/bonds_extra.csv (контрольный пример: БО-П20 Самолёта);
  3) скачивает дневную историю цен и доходностей.

Журнал data/raw/collect_log.csv различает: ok / no_trades (бумага есть, сделок нет) /
not_found (бумаги или эмитента нет) / inn_mismatch / error (сбой выгрузки — повторите запуск).

Запуск:
  python collect_moex.py                     # universe.csv + эмитенты из реестра событий
  python collect_moex.py --refresh           # перекачать историю целиком
  python collect_moex.py --population all    # + все эмитенты корп. облигаций (долго, ~1 ч)
"""
import argparse
import re
import time

import pandas as pd
import requests

from config import DATA, RAW, ISS, START_DATE

HIST = RAW / "history"
HIST.mkdir(parents=True, exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "pd-model-student-project/2.0"
LOG = []
CORP_TYPES = ("exchange_bond", "corporate_bond", "ifi_bond", "non_exchange_bond")


class IssError(RuntimeError):
    pass


def iss(path, **params):
    params.setdefault("iss.meta", "off")
    last = None
    for attempt in range(5):
        try:
            r = S.get(f"{ISS}{path}", params=params, timeout=30)
            r.raise_for_status()
            time.sleep(0.12)
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
    raise IssError(f"{path}: {last}")


def block(js, name):
    """Блок ISS → DataFrame. Имена колонок приводятся к нижнему регистру:
    разные методы ISS отдают SECID/TYPE или secid/type."""
    b = (js or {}).get(name) or {}
    df = pd.DataFrame(b.get("data", []), columns=b.get("columns", []))
    df.columns = [str(c).lower() for c in df.columns]
    return df


def log(company, inn, step, status, secid="", msg=""):
    LOG.append({"company": company, "inn": inn, "step": step, "secid": secid, "status": status,
                "message": str(msg)[:300], "ts": pd.Timestamp.now().isoformat(timespec="seconds")})


def norm(s):
    s = str(s).lower().replace("ё", "е")
    s = re.sub(r"[\"«»'()]", " ", s)
    s = re.sub(r"\b(пао|ао|оао|зао|ооо|нао|мкпао|мфк|гк|группа компаний|специализированный застройщик)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_inn(x):
    s = re.sub(r"\D", "", str(x if x is not None else "").split(".")[0])
    return s if len(s) in (10, 12) else ""


def search(query, pages=5):
    out = []
    for p in range(pages):
        df = block(iss("/securities.json", q=query, engine="stock", market="bonds",
                       start=p * 100, limit=100), "securities")
        if df.empty:
            break
        out.append(df)
        if len(df) < 100:
            break
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    if "group" in df:
        df = df[df["group"].astype(str).str.contains("bond")]
    if "emitent_inn" in df:
        df["emitent_inn"] = df["emitent_inn"].map(clean_inn)
    return df


def resolve(query, inn=""):
    """→ (строка кандидата или None, статус, текст со всеми кандидатами).
    При заданном ИНН кандидат с другим ИНН не выбирается никогда."""
    df = search(query)
    if inn and (df.empty or inn not in set(df.get("emitent_inn", []))):
        df = pd.concat([df, search(inn)], ignore_index=True) if not df.empty else search(inn)
    if df.empty or "emitent_id" not in df:
        return None, "not_found", ""
    g = (df.dropna(subset=["emitent_id"])
           .groupby(["emitent_id", "emitent_title", "emitent_inn"], dropna=False)
           .size().reset_index(name="n_bonds"))
    cands = "; ".join(f"{r.emitent_title} (ИНН {r.emitent_inn}, {r.n_bonds} обл.)" for r in g.itertuples())
    if inn:
        m = g[g.emitent_inn == inn]
        return (m.iloc[0], "ok", cands) if len(m) else (None, "not_found_inn", cands)
    q = norm(query).split()
    g["match"] = g.emitent_title.map(lambda t: sum(w in norm(t) for w in q) / max(len(q), 1))
    g = g.sort_values(["match", "n_bonds"], ascending=False)
    best = g.iloc[0]
    return (best, "ok_name_match", cands) if best.match >= 0.5 else (None, "not_found", cands)


def emitter_list(emitter_id):
    df = block(iss(f"/emitters/{int(emitter_id)}/securities.json"), "securities")
    if df.empty or "secid" not in df:
        return []
    typ = df["type"].astype(str) if "type" in df else pd.Series("", index=df.index)
    return df.loc[typ.str.contains("bond") & ~typ.str.contains("ofz"), "secid"].astype(str).tolist()


def full_listing(refresh=False):
    """Все бумаги, когда-либо торговавшиеся на рынке облигаций (с датами истории)."""
    f = RAW / "iss_bond_listing.csv"
    if f.exists() and not refresh:
        return pd.read_csv(f, dtype=str)
    rows, start = [], 0
    while True:
        df = block(iss("/history/engines/stock/markets/bonds/listing.json", start=start), "securities")
        if df.empty:
            break
        rows.append(df)
        start += len(df)
    lst = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["secid"])
    lst = (lst.sort_values("history_till", na_position="first")
              .drop_duplicates("secid", keep="last") if "history_till" in lst else lst.drop_duplicates("secid"))
    lst.to_csv(f, index=False)
    print(f"Исторический листинг: {len(lst)} бумаг")
    return lst.astype(str)


def card(secid):
    """Строка поиска ISS по коду бумаги: эмитент, ИНН, тип, торгуется ли."""
    df = search(secid, pages=1)
    if df.empty:
        return None
    m = df[(df.get("secid", "") == secid) | (df.get("isin", "") == secid)]
    return (m if len(m) else df).iloc[0]


def describe(secid):
    d = block(iss(f"/securities/{secid}.json"), "description")
    if d.empty:
        return {}
    kv = dict(zip(d["name"].astype(str).str.upper(), d["value"]))
    return {"secid": kv.get("SECID", secid), "isin": kv.get("ISIN"),
            "shortname": kv.get("SHORTNAME") or kv.get("NAME"), "issuedate": kv.get("ISSUEDATE"),
            "matdate": kv.get("MATDATE"), "offerdate": kv.get("OFFERDATE"),
            "facevalue": kv.get("FACEVALUE"), "faceunit": kv.get("FACEUNIT"),
            "coupon_percent": kv.get("COUPONPERCENT"), "coupon_frequency": kv.get("COUPONFREQUENCY")}


def stems(shortnames):
    out = set()
    for s in shortnames:
        m = re.match(r"[A-Za-zА-Яа-яЁё\-]{3,}", str(s))
        if m:
            out.add(m.group(0).lower()[:8])
    return out


def bond_candidates(company, inn, emitter_id, title, listing, extra):
    """Объединяет выпуски эмитента из всех источников и подтверждает ИНН."""
    found = {}
    for s in emitter_list(emitter_id):
        found[s] = "emitter_list"
    srch = search(title)
    if not srch.empty and "secid" in srch:
        for r in srch[srch.emitent_inn == inn].itertuples():
            found.setdefault(r.secid, "search")
    # погашенные/снятые: по стеблю краткого названия и по названию эмитента в полном имени бумаги
    if len(listing):
        st = stems(srch.loc[srch.emitent_inn == inn, "shortname"]) if "shortname" in srch else set()
        core = [w for w in norm(title).split() if len(w) >= 4]
        sn = listing.get("shortname", pd.Series(dtype=str)).fillna("").str.lower()
        nm = listing.get("name", pd.Series(dtype=str)).fillna("").map(norm)
        hit = sn.map(lambda x: any(x.startswith(s) for s in st)) if st else pd.Series(False, index=listing.index)
        if core:
            hit |= nm.map(lambda x: all(w in x for w in core))
        for s in listing.loc[hit, "secid"]:
            if s not in found:
                c = card(s)
                if c is not None and clean_inn(c.get("emitent_inn")) == inn:
                    found[s] = "listing"
                else:
                    log(company, inn, "bond_match", "inn_mismatch", s,
                        f"ИНН в карточке: {None if c is None else c.get('emitent_inn')}")
    for s in extra:
        if s not in found:
            c = card(s)
            if c is not None and clean_inn(c.get("emitent_inn")) == inn:
                found[c.get("secid", s)] = "bonds_extra"
            else:
                log(company, inn, "bond_extra", "inn_mismatch" if c is not None else "not_found", s)
    return found


def history(secid, refresh=False):
    """→ (число дней истории, статус). При сбое файл не создаётся, чтобы повторный запуск докачал."""
    f = HIST / f"{secid}.csv"
    old = pd.DataFrame()
    start_from = START_DATE
    if f.exists() and not refresh:
        old = pd.read_csv(f, parse_dates=["tradedate"])
        if len(old):
            start_from = (old.tradedate.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    rows, start = [], 0
    while True:
        js = iss(f"/history/engines/stock/markets/bonds/securities/{secid}.json",
                 **{"from": start_from, "start": start, "marketprice_board": 1})
        h = block(js, "history")
        if h.empty:
            break
        rows.append(h)
        cur = block(js, "history.cursor")
        if cur.empty:
            start += len(h)
            if len(h) < 100:
                break
            continue
        idx, total, size = (int(cur.iloc[0][k]) for k in ("index", "total", "pagesize"))
        start = idx + size
        if start >= total:
            break
    if rows:
        h = pd.concat(rows, ignore_index=True)
        price = None
        for c in ("legalcloseprice", "close", "waprice", "marketprice2", "marketprice3"):
            if c in h:
                price = h[c] if price is None else price.fillna(h[c])
        yld = None
        for c in ("yieldclose", "yieldatwap"):
            if c in h:
                v = pd.to_numeric(h[c], errors="coerce").replace(0, pd.NA)
                yld = v if yld is None else yld.fillna(v)
        new = pd.DataFrame({"tradedate": pd.to_datetime(h["tradedate"]),
                            "price": pd.to_numeric(price, errors="coerce"),
                            "yield": pd.to_numeric(yld, errors="coerce"),
                            "value": pd.to_numeric(h.get("value"), errors="coerce")}).dropna(subset=["price"])
        new = new[new.price > 0]
        old = pd.concat([old, new]).drop_duplicates("tradedate", keep="last").sort_values("tradedate")
    old = old if len(old) else pd.DataFrame(columns=["tradedate", "price", "yield", "value"])
    old.to_csv(f, index=False)
    return len(old), ("ok" if len(old) else "no_trades")


def population_all(listing, known_inns):
    """Все эмитенты корпоративных облигаций из исторического листинга (по карточкам бумаг)."""
    f = RAW / "population_cards.csv"
    done = pd.read_csv(f, dtype=str) if f.exists() else pd.DataFrame(columns=["secid"])
    todo = [s for s in listing.secid if s not in set(done.secid)]
    print(f"Карточек к проверке: {len(todo)}")
    rows = []
    for i, s in enumerate(todo, 1):
        try:
            c = card(s)
        except IssError as e:
            log("", "", "population", "error", s, e)
            continue
        rows.append({"secid": s, "type": None if c is None else c.get("type"),
                     "emitent_id": None if c is None else c.get("emitent_id"),
                     "emitent_title": None if c is None else c.get("emitent_title"),
                     "emitent_inn": None if c is None else clean_inn(c.get("emitent_inn"))})
        if i % 200 == 0:
            done = pd.concat([done, pd.DataFrame(rows)]); rows = []
            done.to_csv(f, index=False); print(f"  {i}/{len(todo)}")
    done = pd.concat([done, pd.DataFrame(rows)])
    done.to_csv(f, index=False)
    corp = done[done.type.isin(CORP_TYPES) & done.emitent_inn.fillna("").str.len().gt(0)]
    corp = corp[~corp.emitent_inn.isin(known_inns)]
    em = corp.groupby("emitent_inn").agg(company=("emitent_title", "first")).reset_index()
    return pd.DataFrame({"company": em.company, "query": em.company, "sector": "н/д", "segment": "Рынок",
                         "mandatory": 0, "inn": em.emitent_inn, "origin": "population"})


def main(refresh=False, population="default", use_listing=True):
    uni = pd.read_csv(DATA / "universe.csv", dtype={"inn": str})
    uni["inn"] = uni["inn"].map(clean_inn)
    uni["origin"] = "universe"
    reg = pd.read_csv(DATA / "defaults_registry.csv", dtype={"inn": str})
    reg["inn"] = reg["inn"].map(clean_inn)
    d = reg.groupby("company").agg(query=("query", "first"), sector=("sector", "first"),
                                   inn=("inn", "first")).reset_index()
    d = d[~d.company.isin(uni.company) & ~(d.inn.ne("") & d.inn.isin(uni.inn))]
    d["segment"], d["mandatory"], d["origin"] = "Реестр событий", 0, "registry"
    allc = pd.concat([uni, d], ignore_index=True)
    extra = pd.read_csv(DATA / "bonds_extra.csv", dtype=str) if (DATA / "bonds_extra.csv").exists() else pd.DataFrame(columns=["inn", "secid"])

    listing = pd.DataFrame()
    if use_listing:
        try:
            listing = full_listing(refresh)
        except IssError as e:
            print("! Исторический листинг недоступен — погашенные выпуски будут неполными:", e)
            log("", "", "listing", "error", "", e)
    if population == "all":
        allc = pd.concat([allc, population_all(listing, set(allc.inn) - {""})], ignore_index=True)

    emitters, bonds = [], []
    for i, r in allc.iterrows():
        inn = r.inn or ""
        print(f"[{i + 1}/{len(allc)}] {r.company}")
        rec = r.to_dict()
        try:
            best, status, cands = resolve(r.query, inn)
        except IssError as e:
            log(r.company, inn, "resolve", "error", "", e)
            emitters.append(rec | {"status": "error", "candidates": ""})
            continue
        rec |= {"candidates": cands, "status": status}
        if best is None:
            print(f"   ✗ {status}; кандидаты: {cands[:200]}")
            log(r.company, inn, "resolve", status, "", cands)
            emitters.append(rec)
            continue
        inn = clean_inn(best.emitent_inn)
        rec |= {"emitter_id": int(best.emitent_id), "inn": inn, "emitter_title": best.emitent_title,
                "inn_verified": int(bool(r.inn))}
        emitters.append(rec)
        log(r.company, inn, "resolve", status, "", best.emitent_title)
        try:
            ex = extra.loc[extra.inn.map(clean_inn) == inn, "secid"].tolist()
            found = bond_candidates(r.company, inn, best.emitent_id, best.emitent_title, listing, ex)
        except IssError as e:
            log(r.company, inn, "bonds", "error", "", e)
            continue
        print(f"   ✓ {best.emitent_title} (ИНН {inn}); выпусков: {len(found)} "
              f"({', '.join(f'{k}:{v}' for k, v in pd.Series(found).value_counts().items())})")
        lst = listing.set_index("secid") if len(listing) and "secid" in listing else None
        for secid, src in found.items():
            try:
                dsc = describe(secid)
                if not dsc:
                    log(r.company, inn, "describe", "not_found", secid)
                    continue
                n, st = history(secid, refresh)
            except IssError as e:
                log(r.company, inn, "history", "error", secid, e)
                continue
            log(r.company, inn, "history", st, secid, f"дней: {n}")
            hf = ht = None
            if lst is not None and secid in lst.index:
                hf, ht = lst.loc[secid].get("history_from"), lst.loc[secid].get("history_till")
            bonds.append(dsc | {"company": r.company, "inn": inn, "emitter_id": int(best.emitent_id),
                                "source": src, "history_from": hf, "history_till": ht, "n_days": n})
        pd.DataFrame(emitters).to_csv(RAW / "emitters.csv", index=False)
        pd.DataFrame(bonds).to_csv(RAW / "bonds.csv", index=False)
        pd.DataFrame(LOG).to_csv(RAW / "collect_log.csv", index=False)

    em = pd.DataFrame(emitters)
    pd.DataFrame(LOG).to_csv(RAW / "collect_log.csv", index=False)
    em.to_csv(RAW / "emitters.csv", index=False)
    ok = em.status.isin(["ok", "ok_name_match"])
    print(f"\nЭмитентов найдено {ok.sum()} из {len(em)} (по ИНН: {(em.status == 'ok').sum()}, "
          f"по названию — проверить: {(em.status == 'ok_name_match').sum()}); выпусков {len(bonds)}")
    for s in ("not_found_inn", "not_found", "error"):
        x = em[em.status == s]
        if len(x):
            print(f"{s}: {', '.join(x.company)}")
    if (em.status == "error").any() or any(l["status"] == "error" for l in LOG):
        print("Были сбои выгрузки (status=error в collect_log.csv) — перезапустите скрипт.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--population", choices=["default", "all"], default="default")
    ap.add_argument("--no-listing", action="store_true", help="не использовать исторический листинг")
    a = ap.parse_args()
    main(a.refresh, a.population, not a.no_listing)
