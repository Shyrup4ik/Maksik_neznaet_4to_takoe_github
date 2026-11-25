"""Шаг 2. Годовая бухгалтерская отчётность (РСБУ) из ГИР БО ФНС (bo.nalog.gov.ru).

Ключ — ИНН из data/raw/emitters.csv. Значения в тысячах рублей, коды строк форм 0710001/0710002.

Правила:
  * адреса пробуются по очереди из GIRBO_BASES; ответ HTML вместо JSON (переадресация на главную
    или защита от ботов) — это статус html_not_json, а не «нет данных»;
  * в кеш (data/raw/girbo_cache) пишется только успешно и полностью полученный ответ;
    сбой не кешируется, повторный запуск его перезапросит;
  * строка за год Y берётся из отчёта за год Y (колонка «current»). Сравнительные данные
    («previous») из отчёта за Y+1 используются, только если собственного отчёта за Y нет, и тогда
    им ставится дата публикации отчёта Y+1 — так в прошлое не попадает информация из будущего;
  * дата публикации — из метаданных отчёта, если ГИР БО её вернул, иначе 31.12.Y + ANNUAL_REPORT_LAG_DAYS
    (колонка publish_date_source показывает, какой вариант использован);
  * отсутствующая строка отчёта остаётся пустой (NaN), а не нулём.

Журнал: data/raw/girbo_log.csv (ok / no_reports / org_not_found / html_not_json / error).
Квартальную и консолидированную (МСФО) отчётность вносите в data/financials_manual.csv.
"""
import json
import re
import time

import pandas as pd
import requests

from config import RAW, GIRBO_BASES, ANNUAL_REPORT_LAG_DAYS

CACHE = RAW / "girbo_cache"
CACHE.mkdir(parents=True, exist_ok=True)
S = requests.Session()
S.headers.update({"User-Agent": "Mozilla/5.0 (pd-model student project)",
                  "Accept": "application/json, text/plain, */*"})
CODES = ["1100", "1150", "1200", "1210", "1230", "1240", "1250", "1300", "1370",
         "1400", "1410", "1500", "1510", "1520", "1600",
         "2110", "2120", "2200", "2300", "2320", "2330", "2400"]
DATE_KEYS = ("datePresent", "actualBfoDate", "publicationDate", "dateOfPublication", "uploadDate", "receiptDate")


class GirboError(RuntimeError):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status = status


def get(path, **params):
    """JSON или None (404). Исключение GirboError со статусом при HTML/сбое."""
    last = ("error", "")
    for base in GIRBO_BASES:
        for attempt in range(3):
            try:
                r = S.get(base + path, params=params, timeout=40, allow_redirects=True)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "")
                if "json" not in ctype and r.text.lstrip()[:1] not in "[{":
                    last = ("html_not_json", f"{base}{path}: {ctype or 'не JSON'} (финальный URL {r.url})")
                    break                      # этот адрес отдаёт HTML — пробуем следующий домен
                time.sleep(0.6)
                return r.json()
            except requests.RequestException as e:
                last = ("error", f"{base}{path}: {e}")
                time.sleep(3 * (attempt + 1))
    raise GirboError(*last)


def find_org(inn):
    for path in ("/advanced-search/organizations/search", "/nbo/organizations/search"):
        try:
            js = get(path, query=inn, page=0)
        except GirboError as e:
            err = e
            continue
        items = (js or {}).get("content", js if isinstance(js, list) else [])
        for it in items:
            if str(it.get("inn")) == str(inn):
                return it.get("id")
        return None
    raise err


def walk(obj, acc, dates):
    if isinstance(obj, dict):
        for k, v in obj.items():
            m = re.fullmatch(r"(current|previous|beforePrevious)(\d{4})", k)
            if m and v not in (None, ""):
                try:
                    acc.setdefault((m.group(1), m.group(2)), float(v))
                except (TypeError, ValueError):
                    pass
            elif k in DATE_KEYS and v:
                dates.setdefault(k, v)
            else:
                walk(v, acc, dates)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, acc, dates)


def org_reports(org_id, refresh=False):
    f = CACHE / f"{org_id}.json"
    if f.exists() and not refresh:
        return json.loads(f.read_text("utf-8"))
    lst = get(f"/nbo/organizations/{org_id}/bfo/")
    if lst is None:
        lst = []
    out = []
    for item in lst:
        acc, dates = {}, {}
        walk(item, acc, dates)
        if not any(k[0] == "current" for k in acc) and item.get("id"):
            walk(get(f"/nbo/bfo/{item['id']}/details") or {}, acc, dates)   # сбой → исключение, кеша нет
        out.append({"period": item.get("period") or item.get("year"), "bfo_id": item.get("id"),
                    "dates": dates, "values": {f"{a}|{c}": v for (a, c), v in acc.items()}})
    f.write_text(json.dumps(out, ensure_ascii=False), "utf-8")   # только после полного успеха
    return out


def _publish(rep, year):
    for k in DATE_KEYS:
        v = rep.get("dates", {}).get(k)
        d = pd.to_datetime(str(v)[:10], errors="coerce")
        if pd.notna(d) and d.year >= year:
            return d, f"ГИР БО: {k}"
    return pd.Timestamp(f"{year}-12-31") + pd.Timedelta(days=ANNUAL_REPORT_LAG_DAYS), \
        f"оценка: 31.12 + {ANNUAL_REPORT_LAG_DAYS} дн."


def to_rows(company, inn, reports):
    own, comp = {}, {}
    for rep in reports:
        try:
            y = int(str(rep["period"])[:4])
        except (TypeError, ValueError):
            continue
        vals = rep["values"]
        pub, pub_src = _publish(rep, y + 1)
        for tag, year, store in (("current", y, own), ("previous", y - 1, comp)):
            rec = {f"c{c}": vals.get(f"{tag}|{c}") for c in CODES}
            if all(v is None for v in rec.values()):
                continue
            src = "ГИР БО, РСБУ год" if tag == "current" else f"ГИР БО, сравнительные данные отчёта за {y}"
            store[year] = {"company": company, "inn": inn, "year": year, "period_end": f"{year}-12-31",
                           "period_months": 12, "standard": "РСБУ", "perimeter": "юрлицо",
                           "publish_date": pub.date(), "publish_date_source": pub_src,
                           "bfo_id": rep.get("bfo_id"), **rec, "source": src}
    rows = dict(comp)
    rows.update(own)               # собственный отчёт за год всегда приоритетнее сравнительных данных
    return list(rows.values())


def main(refresh=False):
    em = pd.read_csv(RAW / "emitters.csv", dtype={"inn": str})
    em = em[em.status.isin(["ok", "ok_name_match"])]
    out, logs = [], []
    for i, r in enumerate(em.itertuples(), 1):
        inn = str(r.inn).split(".")[0]
        print(f"[{i}/{len(em)}] {r.company} (ИНН {inn})")
        try:
            org = find_org(inn)
            if not org:
                status, rows = "org_not_found", []
            else:
                rows = to_rows(r.company, inn, org_reports(org, refresh))
                status = "ok" if rows else "no_reports"
            msg = ", ".join(str(x["year"]) for x in sorted(rows, key=lambda x: x["year"]))
        except GirboError as e:
            status, rows, msg = e.status, [], str(e)
        print(f"   {status} {msg[:120]}")
        logs.append({"company": r.company, "inn": inn, "status": status, "message": msg,
                     "ts": pd.Timestamp.now().isoformat(timespec="seconds")})
        out += rows
        pd.DataFrame(out).to_csv(RAW / "financials.csv", index=False)
        pd.DataFrame(logs).to_csv(RAW / "girbo_log.csv", index=False)
    lg = pd.DataFrame(logs)
    print("\nИтог по статусам:", lg.status.value_counts().to_dict() if len(lg) else {})
    if len(lg) and lg.status.isin(["error", "html_not_json"]).any():
        print("Есть сбои/HTML-ответы — перезапустите позже; уже полученное не перекачивается.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    main(ap.parse_args().refresh)
