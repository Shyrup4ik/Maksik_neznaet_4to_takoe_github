"""Автотесты: форматы ответов источников, ИНН, границы окна метки, разбиение, пропуски, MARS, интерфейс.

Запуск: python -m pytest -q tests
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def sandbox(tmp_path_factory):
    d = tmp_path_factory.mktemp("pd")
    os.environ["PD_DATA_DIR"] = str(d / "data")
    os.environ["PD_MODELS_DIR"] = str(d / "models")
    os.environ["PD_AS_OF"] = "2026-11-20"
    return d


# ------------------------------------------------------------------ MOEX ISS
def test_iss_block_uppercase_columns(sandbox):
    import collect_moex as cm
    js = {"securities": {"columns": ["SECID", "TYPE", "IS_TRADED"],
                         "data": [["RU000A10CZA1", "exchange_bond", 0], ["SMLT", "common_share", 1]]}}
    df = cm.block(js, "securities")
    assert list(df.columns) == ["secid", "type", "is_traded"]


def test_emitter_list_accepts_uppercase(monkeypatch, sandbox):
    import collect_moex as cm
    monkeypatch.setattr(cm, "iss", lambda path, **p: {"securities": {
        "columns": ["SECID", "TYPE"], "data": [["A1", "exchange_bond"], ["S", "common_share"], ["O", "ofz_bond"]]}})
    assert cm.emitter_list(9848) == ["A1"]


def _search_stub(rows):
    cols = ["secid", "shortname", "isin", "emitent_id", "emitent_title", "emitent_inn", "group", "type", "is_traded"]
    return lambda path, **p: ({"securities": {"columns": cols, "data": rows if p.get("start", 0) == 0 else []}})


def test_resolve_never_takes_other_inn(monkeypatch, sandbox):
    import collect_moex as cm
    rows = [["X1", "ЛСР БО1", "RU1", 1, "ООО ЛСР-Инвест", "7801111111", "stock_bonds", "exchange_bond", 1]]
    monkeypatch.setattr(cm, "iss", _search_stub(rows))
    best, status, cands = cm.resolve("ЛСР", "7838360491")
    assert best is None and status == "not_found_inn" and "7801111111" in cands


def test_resolve_by_inn(monkeypatch, sandbox):
    import collect_moex as cm
    rows = [["X1", "ЛСР БО1", "RU1", 1, "ООО ЛСР-Инвест", "7801111111", "stock_bonds", "exchange_bond", 1],
            ["X2", "ЛСР БО2", "RU2", 2, "ПАО Группа ЛСР", "7838360491", "stock_bonds", "exchange_bond", 1]]
    monkeypatch.setattr(cm, "iss", _search_stub(rows))
    best, status, _ = cm.resolve("Группа ЛСР", "7838360491")
    assert status == "ok" and best.emitent_inn == "7838360491" and int(best.emitent_id) == 2


def test_history_error_is_not_written(monkeypatch, sandbox):
    import collect_moex as cm

    def boom(*a, **k):
        raise cm.IssError("timeout")
    monkeypatch.setattr(cm, "iss", boom)
    with pytest.raises(cm.IssError):
        cm.history("ZZZTEST")
    assert not (cm.HIST / "ZZZTEST.csv").exists()


# ------------------------------------------------------------------ ГИР БО
class _Resp:
    def __init__(self, text, ctype, status=200, url="https://bo.nalog.gov.ru/"):
        self.text, self.headers, self.status_code, self.url = text, {"Content-Type": ctype}, status, url

    def raise_for_status(self):
        pass

    def json(self):
        return json.loads(self.text)


def test_girbo_html_is_diagnosed(monkeypatch, sandbox):
    import collect_girbo as cg
    monkeypatch.setattr(cg.S, "get", lambda *a, **k: _Resp("<html>главная</html>", "text/html"))
    monkeypatch.setattr(cg.time, "sleep", lambda s: None)
    with pytest.raises(cg.GirboError) as e:
        cg.get("/nbo/organizations/search", query="1")
    assert e.value.status == "html_not_json"


def test_girbo_failure_not_cached(monkeypatch, sandbox):
    import collect_girbo as cg
    calls = {"n": 0}

    def fake_get(path, **p):
        calls["n"] += 1
        if path.endswith("/bfo/"):
            return [{"id": 5, "period": "2024"}]
        raise cg.GirboError("error", "timeout")
    monkeypatch.setattr(cg, "get", fake_get)
    with pytest.raises(cg.GirboError):
        cg.org_reports(999001)
    assert not (cg.CACHE / "999001.json").exists()


def test_girbo_comparatives_get_later_publish_date(sandbox):
    import collect_girbo as cg
    reps = [{"period": "2024", "dates": {}, "values": {"current|1600": 100.0, "previous|1600": 90.0}}]
    rows = {r["year"]: r for r in cg.to_rows("X", "7700000001", reps)}
    assert rows[2024]["c1600"] == 100.0 and "сравнительные" in rows[2023]["source"]
    assert pd.Timestamp(rows[2023]["publish_date"]) == pd.Timestamp(rows[2024]["publish_date"])
    reps.append({"period": "2023", "dates": {}, "values": {"current|1600": 91.0}})
    rows = {r["year"]: r for r in cg.to_rows("X", "7700000001", reps)}
    assert rows[2023]["c1600"] == 91.0 and rows[2023]["source"] == "ГИР БО, РСБУ год"


# ------------------------------------------------------------------ признаки
def test_missing_lines_are_not_zero():
    from features import ratios
    r = ratios({"c1600": 1000, "c1200": 400, "c1500": 300, "c1300": 300, "c2300": 50, "c2400": 40})
    assert np.isnan(r["debt_to_assets"]) and np.isnan(r["interest_coverage"]) and np.isnan(r["net_debt_to_ebit"])
    r = ratios({"c1600": 1000, "c1200": 400, "c1500": 300, "c1300": 300, "c2300": 50, "c2330": 0,
                "c1410": 0, "c1510": 0})
    assert r["interest_coverage"] == 50.0 and r["debt_to_assets"] == 0.0


def test_altman_zones():
    from features import altman_zone
    assert altman_zone(4.2) == "бедствия" and altman_zone(4.4) == "серая" and altman_zone(6) == "безопасная"


# ------------------------------------------------------------------ метки и отчётность
def test_label_window_boundaries(sandbox):
    import build_dataset as bd
    t = pd.Timestamp("2024-06-30")
    assert bd.label(t, [pd.Timestamp("2024-12-31")], "2024-01-01", pd.Timestamp("2026-01-01"))[:2] == ("known", 1)
    assert bd.label(t, [pd.Timestamp("2025-01-01")], "2024-01-01", pd.Timestamp("2026-01-01"))[:2] == ("known", 0)
    assert bd.label(t, [t], "2024-01-01", pd.Timestamp("2026-01-01"))[:2] == ("known", 0)   # событие в день среза — прошлое
    assert bd.label(t, [], "2024-01-01", pd.Timestamp("2024-12-30"))[0] == "unknown_open_window"
    assert bd.label(pd.Timestamp("2023-12-31"), [], "2024-01-01", pd.Timestamp("2026-01-01"))[0] == "unknown_no_coverage"


def test_default_interval_and_cure(sandbox):
    import build_dataset as bd
    reg = pd.DataFrame({"inn": ["1"], "event_type": ["D"], "event_date": [pd.Timestamp("2024-04-26")],
                        "cure_date": [pd.Timestamp("2024-05-31")]})
    iv = bd.default_intervals(reg)["1"]
    assert bd.in_default(iv, pd.Timestamp("2024-04-30"))
    assert not bd.in_default(iv, pd.Timestamp("2024-06-30"))


def test_target_events_one_per_episode(sandbox):
    import build_dataset as bd
    reg = pd.DataFrame({"episode_id": ["a#1", "a#1"], "inn": ["1", "1"], "event_type": ["TD", "D"],
                        "event_date": pd.to_datetime(["2024-04-11", "2024-04-26"])})
    b = bd.target_events(reg, "broad")
    s = bd.target_events(reg, "strict")
    assert len(b) == 1 and b.date.iloc[0] == pd.Timestamp("2024-04-11")
    assert len(s) == 1 and s.date.iloc[0] == pd.Timestamp("2024-04-26")


def test_reports_without_publish_date_column(sandbox):
    import build_dataset as bd
    df = pd.DataFrame({"inn": ["7700000001"], "period_end": ["2025-06-30"], "c1600": [100]})
    out = bd.normalize_reports(df, "ручной ввод")
    assert out.period_months.iloc[0] == 6
    assert out.publish_date.iloc[0] == pd.Timestamp("2025-06-30") + pd.Timedelta(days=60)
    with pytest.raises(ValueError):
        bd.normalize_reports(pd.DataFrame({"inn": ["1"], "c1600": [1]}), "ручной ввод")


def _rep(pe, m, rev, pub, c1600=100.0):
    return {"inn": "7700000001", "period_end": pe, "period_months": m, "standard": "МСФО", "perimeter": "группа",
            "publish_date": pub, "c1600": c1600, "c2110": rev}


def test_ltm_and_publication_cutoff(sandbox):
    import build_dataset as bd
    df = bd.normalize_reports(pd.DataFrame([
        _rep("2024-06-30", 6, 50, "2024-08-30"), _rep("2024-12-31", 12, 120, "2025-04-30"),
        _rep("2025-06-30", 6, 70, "2025-08-29"),
        _rep("2025-06-30", 6, 70, "2025-10-15", c1600=999.0)]), "ручной ввод")   # исправленная версия
    cur, prev = bd.pick_report(df, pd.Timestamp("2025-09-30"))
    assert cur["fin_flow_basis"] == "LTM" and cur["c2110"] == 70 + 120 - 50
    assert cur["c1600"] == 100.0                                    # исправление ещё не опубликовано
    assert prev is not None and prev["period_end"] == pd.Timestamp("2024-06-30")
    cur2, _ = bd.pick_report(df, pd.Timestamp("2025-12-31"))
    assert cur2["c1600"] == 999.0
    cur0, _ = bd.pick_report(df, pd.Timestamp("2024-08-01"))
    assert cur0 is None


def test_horizon_month_end():
    from config import horizon_end
    assert horizon_end("2024-12-31") == pd.Timestamp("2025-06-30")
    assert horizon_end("2025-06-30") == pd.Timestamp("2025-12-31")


# ------------------------------------------------------------------ разбиение
def test_splits(sandbox):
    import train
    d = pd.DataFrame({
        "inn": ["1", "1", "1", "9731004688", "9731004688", "2"],
        "date": pd.to_datetime(["2024-12-31", "2025-03-31", "2025-09-30", "2024-06-30", "2025-12-31", "2026-06-30"]),
        "label_status": ["known", "known", "known", "known", "known", "unknown_open_window"]})
    from config import horizon_end
    d["window_end"] = d.date.map(horizon_end)
    d["split"] = train.assign_splits(d, "2025-06-30", ["9731004688"])
    assert list(d.split) == ["train", "gap", "test", "holdout_backtest", "test", "score"]
    assert all(train.check_splits(d, "2025-06-30", ["9731004688"]).values())


# ------------------------------------------------------------------ MARS
def test_mars_basis_full_rank():
    from mars import MARSClassifier
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, (1500, 1))
    z = -2 + 1.5 * np.maximum(0, x[:, 0] - 1) - 1.2 * np.maximum(0, x[:, 0] + 1)
    y = (rng.random(1500) < 1 / (1 + np.exp(-z))).astype(float)
    m = MARSClassifier(max_terms=15, n_knots=10)
    _, B = m._forward(x, y)
    assert np.linalg.matrix_rank(B) == B.shape[1]


# ------------------------------------------------------------------ конвейер и интерфейс
def _run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, env=os.environ.copy())
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


@pytest.mark.parametrize("flag", ["", "--no-market"])
def test_pipeline_and_app(sandbox, flag):
    _run("tests/make_fake_raw.py", *([flag] if flag else []))
    _run("build_dataset.py")
    out = _run("train.py")
    assert "в обучении нет эмитентов holdout (Самолёт): да" in out
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert any("PD 6м" in x for x in labels)
    sel = [s for s in at.selectbox if s.label.startswith("Обязательные")][0]
    for comp in ("Самолет", "Балтийский лизинг", "ЛСР", "ДжетЛенд", "Лидер-Инвест"):
        assert comp in sel.options
    sel.select("Лидер-Инвест").run()
    assert not at.exception, [e.value for e in at.exception]
    base = [s for s in at.selectbox if s.label.startswith("Подставить")][0]
    base.select(next(o for o in base.options if o.startswith("ЛСР"))).run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    base = [s for s in at.selectbox if s.label.startswith("Подставить")][0]
    base.select("—").run()
    at.button[0].click().run()                      # пустая форма: всё «нет данных»
    assert not at.exception, [e.value for e in at.exception]


def test_collect_moex_end_to_end(monkeypatch, sandbox):
    """Имитация ISS: текущий выпуск из списка эмитента, погашенный БО-П20 — только в историческом
    листинге, чужая бумага с похожим названием — отбрасывается по ИНН, сбой истории — в журнале."""
    import collect_moex as cm
    from config import DATA, RAW
    pd.DataFrame([{"company": "Самолет", "query": "ГК Самолет", "sector": "Девелопмент", "segment": "HY",
                   "mandatory": 1, "inn": "9731004688"}]).to_csv(DATA / "universe.csv", index=False)
    pd.DataFrame(columns=["episode_id", "company", "inn", "query", "sector", "event_type", "event_date"]) \
        .to_csv(DATA / "defaults_registry.csv", index=False)
    pd.DataFrame(columns=["inn", "secid"]).to_csv(DATA / "bonds_extra.csv", index=False)
    for f in ("iss_bond_listing.csv",):
        (RAW / f).unlink(missing_ok=True)
    S = ["secid", "shortname", "isin", "emitent_id", "emitent_title", "emitent_inn", "group", "type", "is_traded"]
    cards = {"RU000A10CZA1": ["RU000A10CZA1", "СамолетP20", "RU000A10CZA1", 9848, 'ПАО "ГК "Самолет"', "9731004688",
                              "stock_bonds", "exchange_bond", 0],
             "RU000A10AAA1": ["RU000A10AAA1", "СамолетP25", "RU000A10AAA1", 9848, 'ПАО "ГК "Самолет"', "9731004688",
                              "stock_bonds", "exchange_bond", 1],
             "RU000A10ZZZ9": ["RU000A10ZZZ9", "СамолетТур1", "RU000A10ZZZ9", 1, "ООО Самолет-Тур", "7700000009",
                              "stock_bonds", "exchange_bond", 0]}

    def fake(path, **p):
        if path == "/securities.json":
            if p.get("start", 0) > 0:
                return {"securities": {"columns": S, "data": []}}
            q = p["q"]
            rows = [cards[q]] if q in cards else [cards["RU000A10AAA1"], cards["RU000A10ZZZ9"]]
            return {"securities": {"columns": S, "data": rows}}
        if path.startswith("/emitters/"):
            return {"securities": {"columns": ["SECID", "TYPE"], "data": [["RU000A10AAA1", "exchange_bond"]]}}
        if path.endswith("listing.json"):
            if p.get("start", 0) > 0:
                return {"securities": {"columns": ["SECID"], "data": []}}
            return {"securities": {"columns": ["SECID", "SHORTNAME", "NAME", "BOARDID", "history_from", "history_till"],
                                   "data": [["RU000A10CZA1", "СамолетP20", 'ПАО "ГК "Самолет" БО-П20', "TQCB",
                                             "2024-09-01", "2026-09-26"],
                                            ["RU000A10ZZZ9", "СамолетТур1", "Самолет-Тур 01", "TQCB",
                                             "2020-01-01", "2021-01-01"]]}}
        if path.startswith("/securities/"):
            s = path.split("/")[2].replace(".json", "")
            return {"description": {"columns": ["name", "title", "value"],
                                    "data": [["SECID", "", s], ["ISIN", "", s], ["SHORTNAME", "", s],
                                             ["ISSUEDATE", "", "2024-09-01"], ["MATDATE", "", "2026-09-29"]]}}
        if "/history/" in path:
            if "AAA1" in path:
                raise cm.IssError("timeout")
            if p.get("start", 0) > 0:
                return {"history": {"columns": ["TRADEDATE"], "data": []}}
            return {"history": {"columns": ["TRADEDATE", "LEGALCLOSEPRICE", "YIELDCLOSE", "VALUE"],
                                "data": [["2026-09-25", 51.0, 250.0, 1e6]]},
                    "history.cursor": {"columns": ["INDEX", "TOTAL", "PAGESIZE"], "data": [[0, 1, 100]]}}
        raise AssertionError(path)

    monkeypatch.setattr(cm, "iss", fake)
    cm.LOG.clear()
    cm.main()
    b = pd.read_csv(RAW / "bonds.csv")
    lg = pd.read_csv(RAW / "collect_log.csv")
    assert set(b.secid) == {"RU000A10CZA1"} and b.source.iloc[0] == "listing"
    assert ((lg.secid == "RU000A10ZZZ9") & (lg.status == "inn_mismatch")).any()
    assert ((lg.secid == "RU000A10AAA1") & (lg.status == "error")).any()


# ------------------------------------------------------------------ исправления по проверке от 25.11.2026
def test_broad_td_before_slice_d_inside_window_is_excluded(sandbox):
    """TD до среза, D внутри окна: срез не должен получать y=0 (раньше broad ставил 0)."""
    import build_dataset as bd
    reg = pd.DataFrame({"episode_id": ["k#1", "k#1"], "inn": ["1", "1"], "event_type": ["TD", "D"],
                        "event_date": pd.to_datetime(["2024-06-14", "2024-07-01"]), "cure_date": [pd.NaT, pd.NaT]})
    t = pd.Timestamp("2024-06-30")
    broad = bd.default_intervals(reg, "broad")["1"]
    assert bd.in_default(broad, t)                                   # срез внутри неплатежа — исключается
    assert not bd.in_default(broad, pd.Timestamp("2024-06-13"))      # до TD срез обычный
    strict = bd.default_intervals(reg, "strict")["1"]
    assert not bd.in_default(strict, t)                              # strict: до D срез обычный, y=1
    assert bd.label(t, list(bd.target_events(reg, "strict").date), "2024-01-01", pd.Timestamp("2026-01-01"))[:2] == ("known", 1)


def test_broad_cured_episode_resumes(sandbox):
    import build_dataset as bd
    reg = pd.DataFrame({"episode_id": ["k#1", "k#1"], "inn": ["1", "1"], "event_type": ["TD", "D"],
                        "event_date": pd.to_datetime(["2024-04-11", "2024-04-26"]),
                        "cure_date": pd.to_datetime(["2024-05-31", "2024-05-31"])})
    iv = bd.default_intervals(reg, "broad")["1"]
    assert bd.in_default(iv, pd.Timestamp("2024-04-15")) and not bd.in_default(iv, pd.Timestamp("2024-06-30"))


def test_pipeline_without_any_financials_fails_clearly(sandbox):
    """Нет отчётности совсем: сборка не падает с KeyError, обучение останавливается понятным сообщением."""
    _run("tests/make_fake_raw.py", "--no-fin")
    _run("build_dataset.py")
    r = subprocess.run([sys.executable, "train.py"], cwd=ROOT, capture_output=True, text=True, env=os.environ.copy())
    assert r.returncode != 0 and "нет строк с отчётностью" in (r.stdout + r.stderr)
