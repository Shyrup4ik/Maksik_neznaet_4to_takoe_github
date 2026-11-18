"""Шаг 1. Сбор рыночных данных с Московской биржи (MOEX ISS, публичный API без ключа).

Что делает:
  1) по названию компании находит эмитента (ID и ИНН) — для universe.csv и для всех
     компаний из реестра дефолтов (defaults_registry.csv);
  2) собирает все его биржевые облигации и их параметры (даты выпуска/погашения, купон);
  3) скачивает дневную историю цен (% от номинала) и доходностей.

Результат: data/raw/emitters.csv, data/raw/bonds.csv, data/raw/history/<SECID>.csv

Запуск:  python collect_moex.py            (докачивает только то, чего нет)
         python collect_moex.py --refresh  (перекачать историю целиком)

Если компания не нашлась или нашлась не та — откройте data/raw/emitters.csv,
посмотрите колонку candidates и впишите правильный ИНН в колонку inn файла
data/universe.csv (для дефолтников — поправьте query в defaults_registry.csv).
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
S.headers["User-Agent"] = "pd-model-student-project/1.0"


def iss(path, **params):
    params.setdefault("iss.meta", "off")
    for attempt in range(5):
        try:
            r = S.get(f"{ISS}{path}", params=params, timeout=30)
            r.raise_for_status()
            time.sleep(0.12)
            return r.json()
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"   ! {path}: {e}; повтор через {wait} c")
            time.sleep(wait)
    raise RuntimeError(f"ISS недоступен: {path}")


def block(js, name):
    b = js.get(name) or {}
    return pd.DataFrame(b.get("data", []), columns=b.get("columns", []))


def norm(s):
    s = str(s).lower().replace("ё", "е")
    s = re.sub(r"[\"«»'()]", " ", s)
    s = re.sub(r"\b(пао|ао|оао|зао|ооо|нао|мкпао|гк|группа компаний)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def search_bonds(query):
    out = []
    for start in range(0, 500, 100):
        df = block(iss("/securities.json", q=query, engine="stock", market="bonds",
                       start=start, limit=100), "securities")
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
    return df


def resolve(company, query, inn=None):
    df = search_bonds(query)
    if (df.empty or "emitent_id" not in df) and inn:
        df = search_bonds(inn)
    if df.empty or "emitent_id" not in df:
        return None, ""
    g = (df.groupby(["emitent_id", "emitent_title", "emitent_inn"], dropna=False)
           .size().reset_index(name="n_bonds"))
    cands = "; ".join(f"{r.emitent_title} (ИНН {r.emitent_inn}, {r.n_bonds} обл.)"
                      for r in g.itertuples())
    if inn and str(inn).strip() not in ("", "nan"):
        m = g[g.emitent_inn.astype(str) == str(inn).split(".")[0]]
        if not m.empty:
            return m.iloc[0], cands
    q = norm(query).split()
    g["match"] = g.emitent_title.map(lambda t: sum(w in norm(t) for w in q) / max(len(q), 1))
    g = g.sort_values(["match", "n_bonds"], ascending=False)
    best = g.iloc[0]
    return (best if best.match >= 0.5 else None), cands


def emitter_bonds(emitter_id):
    df = block(iss(f"/emitters/{int(emitter_id)}/securities.json"), "securities")
    if df.empty:
        return df
    typ = df.get("type", pd.Series("", index=df.index)).astype(str)
    df = df[typ.str.contains("bond") & ~typ.str.contains("ofz")]
    return df


def describe(secid):
    d = block(iss(f"/securities/{secid}.json"), "description")
    if d.empty:
        return {}
    kv = dict(zip(d["name"], d["value"]))
    return {
        "secid": secid,
        "isin": kv.get("ISIN"),
        "shortname": kv.get("SHORTNAME") or kv.get("NAME"),
        "issuedate": kv.get("ISSUEDATE"),
        "matdate": kv.get("MATDATE"),
        "offerdate": kv.get("OFFERDATE"),
        "facevalue": kv.get("FACEVALUE"),
        "faceunit": kv.get("FACEUNIT"),
        "coupon_percent": kv.get("COUPONPERCENT"),
        "coupon_frequency": kv.get("COUPONFREQUENCY"),
    }


def history(secid, refresh=False):
    f = HIST / f"{secid}.csv"
    if f.exists() and not refresh:
        old = pd.read_csv(f, parse_dates=["tradedate"])
        start_from = (old.tradedate.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if len(old) else START_DATE
    else:
        old, start_from = pd.DataFrame(), START_DATE
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
        idx, total, size = cur.iloc[0][["INDEX", "TOTAL", "PAGESIZE"]]
        start = int(idx) + int(size)
        if start >= int(total):
            break
    if rows:
        h = pd.concat(rows, ignore_index=True)
        price = h.get("LEGALCLOSEPRICE")
        for alt in ("CLOSE", "WAPRICE", "MARKETPRICE2", "MARKETPRICE3"):
            if alt in h:
                price = h[alt] if price is None else price.fillna(h[alt])
        yld = h.get("YIELDCLOSE")
        if "YIELDATWAP" in h:
            yld = h["YIELDATWAP"] if yld is None else yld.replace(0, pd.NA).fillna(h["YIELDATWAP"])
        new = pd.DataFrame({
            "tradedate": pd.to_datetime(h["TRADEDATE"]),
            "price": pd.to_numeric(price, errors="coerce"),
            "yield": pd.to_numeric(yld, errors="coerce"),
            "value": pd.to_numeric(h.get("VALUE"), errors="coerce"),
        }).dropna(subset=["price"])
        new = new[new.price > 0]
        out = pd.concat([old, new]).drop_duplicates("tradedate", keep="last").sort_values("tradedate")
        out.to_csv(f, index=False)
        return len(out)
    if not old.empty:
        return len(old)
    pd.DataFrame(columns=["tradedate", "price", "yield", "value"]).to_csv(f, index=False)
    return 0


def main(refresh=False):
    uni = pd.read_csv(DATA / "universe.csv", dtype={"inn": str})
    uni["is_defaulter_registry"] = 0
    reg = pd.read_csv(DATA / "defaults_registry.csv")
    dflt = (reg.groupby("company").agg(query=("query", "first"), sector=("sector", "first"))
               .reset_index())
    dflt = dflt[~dflt.company.isin(uni.company)]
    dflt["segment"], dflt["mandatory"], dflt["inn"], dflt["is_defaulter_registry"] = "Дефолт-реестр", 0, None, 1
    allc = pd.concat([uni, dflt], ignore_index=True)

    emitters, bonds = [], []
    for i, r in allc.iterrows():
        print(f"[{i + 1}/{len(allc)}] {r.company}")
        best, cands = resolve(r.company, r.query, r.get("inn"))
        rec = r.to_dict() | {"candidates": cands}
        if best is None:
            print("   ✗ эмитент не найден — поправьте query/inn")
            emitters.append(rec | {"status": "not_found"})
            continue
        rec |= {"emitter_id": int(best.emitent_id), "inn": str(best.emitent_inn),
                "emitter_title": best.emitent_title, "status": "ok"}
        emitters.append(rec)
        eb = emitter_bonds(best.emitent_id)
        print(f"   ✓ {best.emitent_title} (ИНН {best.emitent_inn}); облигаций: {len(eb)}")
        for secid in eb.get("secid", []):
            d = describe(secid)
            if not d:
                continue
            d |= {"company": r.company, "emitter_id": int(best.emitent_id)}
            d["n_days"] = history(secid, refresh=refresh)
            bonds.append(d)
        pd.DataFrame(emitters).to_csv(RAW / "emitters.csv", index=False)
        pd.DataFrame(bonds).to_csv(RAW / "bonds.csv", index=False)

    em = pd.DataFrame(emitters)
    print(f"\nГотово: эмитентов найдено {(em.status == 'ok').sum()} из {len(em)}, облигаций {len(bonds)}")
    nf = em[em.status != "ok"]
    if len(nf):
        print("Не найдены:", ", ".join(nf.company))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    main(ap.parse_args().refresh)
