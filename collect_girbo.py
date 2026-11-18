"""Шаг 2. Бухгалтерская отчётность (РСБУ) из ГИР БО ФНС — bo.nalog.ru.

ГИР БО бесплатно отдаёт годовую отчётность всех российских юрлиц с 2019 года
(включая мелких эмитентов ВДО, по которым больше нигде данных нет).
Значения — в тысячах рублей, коды строк — стандартные коды форм 0710001/0710002.

Вход:  data/raw/emitters.csv (ИНН из шага 1)
Выход: data/raw/financials.csv — по строке на (компания, год)

Запуск: python collect_girbo.py

Если сайт ФНС отвечает 403/таймаутами (у них бывает защита от ботов) — подождите
и перезапустите (уже скачанное не перекачивается), либо внесите данные вручную
в data/financials_manual.csv (там же можно добавить квартальную/МСФО отчётность).
"""
import json
import re
import time

import pandas as pd
import requests

from config import RAW

BASE = "https://bo.nalog.ru"
CACHE = RAW / "girbo_cache"
CACHE.mkdir(parents=True, exist_ok=True)
S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 (pd-model student project)",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://bo.nalog.ru/",
})

# строки, которые нужны для коэффициентов и модели Альтмана
CODES = ["1100", "1150", "1200", "1210", "1230", "1240", "1250", "1300", "1370",
         "1400", "1410", "1500", "1510", "1520", "1600",
         "2110", "2120", "2200", "2300", "2320", "2330", "2400"]


def get(path, **params):
    for attempt in range(4):
        try:
            r = S.get(BASE + path, params=params, timeout=40)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            time.sleep(0.6)
            return r.json()
        except Exception as e:  # noqa: BLE001
            print(f"   ! {path}: {e}")
            time.sleep(3 * (attempt + 1))
    return None


def find_org(inn):
    for path in ("/advanced-search/organizations/search", "/nbo/organizations/search"):
        js = get(path, query=inn, page=0)
        if not js:
            continue
        items = js.get("content", js if isinstance(js, list) else [])
        for it in items:
            if str(it.get("inn")) == str(inn):
                return it.get("id")
        if items:
            return items[0].get("id")
    return None


def walk(obj, acc):
    """Собирает из произвольного JSON все поля вида current1600 / previous1600."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            m = re.fullmatch(r"(current|previous|beforePrevious)(\d{4})", k)
            if m and v not in (None, ""):
                try:
                    acc.setdefault((m.group(1), m.group(2)), float(v))
                except (TypeError, ValueError):
                    pass
            else:
                walk(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, acc)


def org_reports(org_id):
    f = CACHE / f"{org_id}.json"
    if f.exists():
        return json.loads(f.read_text("utf-8"))
    lst = get(f"/nbo/organizations/{org_id}/bfo/") or []
    out = []
    for item in lst:
        period = item.get("period") or item.get("year")
        acc = {}
        walk(item, acc)
        if not any(k[0] == "current" for k in acc) and item.get("id"):
            walk(get(f"/nbo/bfo/{item['id']}/details") or {}, acc)
        out.append({"period": period, "values": {f"{a}|{c}": v for (a, c), v in acc.items()}})
    f.write_text(json.dumps(out, ensure_ascii=False), "utf-8")
    return out


def to_rows(company, inn, reports):
    rows = {}
    for rep in reports:
        try:
            y = int(str(rep["period"])[:4])
        except (TypeError, ValueError):
            continue
        vals = rep["values"]
        for shift, tag in ((0, "current"), (1, "previous")):
            year = y - shift
            rec = {c: vals.get(f"{tag}|{c}") for c in CODES}
            if all(v is None for v in rec.values()):
                continue
            # отчёт за сам год приоритетнее, чем «предыдущий год» из следующего отчёта
            if year in rows and shift == 1:
                continue
            rows[year] = {"company": company, "inn": inn, "year": year, **{f"c{c}": v for c, v in rec.items()},
                          "source": "ГИР БО (РСБУ, год)"}
    return list(rows.values())


def main():
    em = pd.read_csv(RAW / "emitters.csv", dtype={"inn": str})
    em = em[em.status == "ok"]
    out, missing = [], []
    for i, r in enumerate(em.itertuples(), 1):
        inn = str(r.inn).split(".")[0]
        print(f"[{i}/{len(em)}] {r.company} (ИНН {inn})")
        org = find_org(inn)
        if not org:
            print("   ✗ не найдена в ГИР БО")
            missing.append(r.company)
            continue
        rows = to_rows(r.company, inn, org_reports(org))
        print(f"   ✓ лет отчётности: {sorted(x['year'] for x in rows)}")
        out += rows
        pd.DataFrame(out).to_csv(RAW / "financials.csv", index=False)
    print(f"\nГотово: {len(out)} компания-лет. Без отчётности: {', '.join(missing) or 'нет'}")


if __name__ == "__main__":
    main()
