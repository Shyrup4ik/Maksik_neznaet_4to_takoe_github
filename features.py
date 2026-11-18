"""Расчёт финансовых коэффициентов и Z-счёта Альтмана из строк РСБУ.

Используется и при сборке базы, и в интерфейсе (ручной ввод отчётности),
чтобы признаки считались одинаково.
Коды строк: 1200 оборотные активы, 1210 запасы, 1240 фин.вложения, 1250 денежные средства,
1300 капитал, 1370 нераспределённая прибыль, 1400/1500 долгоср./краткоср. обязательства,
1410/1510 займы, 1600 баланс, 2110 выручка, 2200 прибыль от продаж, 2300 прибыль до налога,
2330 проценты к уплате, 2400 чистая прибыль.
"""
import numpy as np

FIN_FEATS = ["current_ratio", "quick_ratio", "cash_ratio", "equity_ratio", "debt_to_assets",
             "st_debt_share", "roa", "ebit_margin", "interest_coverage", "net_debt_to_ebit",
             "log_assets", "neg_equity", "X1_wc_ta", "X2_re_ta", "X3_ebit_ta", "X4_eq_tl"]
DYN_FEATS = ["d_equity_ratio", "d_current_ratio", "d_debt_to_assets", "d_roa",
             "rev_growth", "asset_growth", "d_interest_coverage"]
MKT_FEATS = ["mkt_price_min", "mkt_price_med", "mkt_yield_max", "mkt_yield_med",
             "mkt_spread", "mkt_price_chg_3m", "mkt_vol_3m", "mkt_log_turnover"]
RAT_FEATS = ["rating_num", "rating_chg_6m"]
HIST_FEATS = ["had_event_12m"]
ALTMAN_X = ["X1_wc_ta", "X2_re_ta", "X3_ebit_ta", "X4_eq_tl"]

LABELS_RU = {
    "current_ratio": "Текущая ликвидность", "quick_ratio": "Быстрая ликвидность",
    "cash_ratio": "Абсолютная ликвидность", "equity_ratio": "Доля собств. капитала (автономия)",
    "debt_to_assets": "Долг / Активы", "st_debt_share": "Доля краткосрочного долга",
    "roa": "ROA", "ebit_margin": "Рентабельность EBIT", "interest_coverage": "Покрытие процентов EBIT",
    "net_debt_to_ebit": "Чистый долг / EBIT", "log_assets": "Размер (ln активов)",
    "neg_equity": "Отрицательный капитал", "X1_wc_ta": "X1: Оборотный капитал / Активы",
    "X2_re_ta": "X2: Нераспр. прибыль / Активы", "X3_ebit_ta": "X3: EBIT / Активы",
    "X4_eq_tl": "X4: Капитал / Обязательства",
    "d_equity_ratio": "Δ доли капитала за год", "d_current_ratio": "Δ текущей ликвидности за год",
    "d_debt_to_assets": "Δ Долг/Активы за год", "d_roa": "Δ ROA за год", "rev_growth": "Рост выручки",
    "asset_growth": "Рост активов", "d_interest_coverage": "Δ покрытия процентов",
    "mkt_price_min": "Мин. цена облигаций, % номинала", "mkt_price_med": "Медианная цена облигаций, %",
    "mkt_yield_max": "Макс. доходность, %", "mkt_yield_med": "Медианная доходность, %",
    "mkt_spread": "Спред к медиане рынка, п.п.", "mkt_price_chg_3m": "Изменение цены за 3 мес.",
    "mkt_vol_3m": "Волатильность цены 3 мес.", "mkt_log_turnover": "Оборот торгов (ln)",
    "rating_num": "Кредитный рейтинг (1=AAA … 22=D)", "rating_chg_6m": "Изменение рейтинга за 6 мес.",
    "had_event_12m": "Был тех./дефолт за 12 мес.", "z_altman": "Z''-счёт Альтмана",
}


def _g(d, code):
    v = d.get(f"c{code}") if d is not None else None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    return v


def _div(a, b):
    return a / b if (b is not None and np.isfinite(b) and b != 0 and np.isfinite(a)) else np.nan


def ratios(cur):
    """cur — dict со строками c1600, c1200 … Возвращает dict коэффициентов."""
    TA, CA, CL = _g(cur, 1600), _g(cur, 1200), _g(cur, 1500)
    inv, cash, fin = _g(cur, 1210), _g(cur, 1250), _g(cur, 1240)
    EQ, RE = _g(cur, 1300), _g(cur, 1370)
    LT, ST = _g(cur, 1400), _g(cur, 1500)
    LTD, STD = np.nan_to_num(_g(cur, 1410)), np.nan_to_num(_g(cur, 1510))
    REV, EBT, INT, NI = _g(cur, 2110), _g(cur, 2300), _g(cur, 2330), _g(cur, 2400)
    INT = abs(INT) if np.isfinite(INT) else np.nan
    TL = np.nansum([LT, ST]) if (np.isfinite(LT) or np.isfinite(ST)) else (TA - EQ)
    EBIT = EBT + (INT if np.isfinite(INT) else 0) if np.isfinite(EBT) else np.nan
    debt = LTD + STD
    liquid = np.nansum([cash, fin])
    r = {
        "current_ratio": _div(CA, CL),
        "quick_ratio": _div(CA - np.nan_to_num(inv), CL),
        "cash_ratio": _div(liquid, CL),
        "equity_ratio": _div(EQ, TA),
        "debt_to_assets": _div(debt, TA),
        "st_debt_share": _div(STD, debt) if debt > 0 else 0.0,
        "roa": _div(NI, TA),
        "ebit_margin": _div(EBIT, REV),
        "interest_coverage": (np.clip(_div(EBIT, INT), -10, 50) if (np.isfinite(INT) and INT > 0)
                              else (50.0 if np.isfinite(EBIT) and EBIT > 0 else np.nan)),
        "net_debt_to_ebit": (np.clip(_div(debt - liquid, EBIT), -20, 50) if (np.isfinite(EBIT) and EBIT > 0)
                             else (50.0 if debt - liquid > 0 else 0.0) if np.isfinite(EBIT) else np.nan),
        "log_assets": np.log(TA * 1000) if np.isfinite(TA) and TA > 0 else np.nan,
        "neg_equity": float(EQ < 0) if np.isfinite(EQ) else np.nan,
        "X1_wc_ta": _div(CA - CL, TA),
        "X2_re_ta": _div(RE, TA),
        "X3_ebit_ta": _div(EBIT, TA),
        "X4_eq_tl": _div(EQ, TL),
    }
    r["z_altman"] = altman_z(r)
    return r


def altman_z(r):
    """Z''-score Альтмана для развивающихся рынков (Altman, Hartzell, Peck, 1995).
    Z'' = 3.25 + 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4.
    Зоны: Z'' > 5.85 — безопасная, 4.15…5.85 — серая, < 4.15 — зона бедствия
    (эквивалент 2.6 / 1.1 для версии без константы 3.25)."""
    x = [r.get(k) for k in ALTMAN_X]
    if any(v is None or not np.isfinite(v) for v in x):
        return np.nan
    x = [float(np.clip(v, -5, 5)) for v in x]
    return 3.25 + 6.56 * x[0] + 3.26 * x[1] + 6.72 * x[2] + 1.05 * x[3]


def altman_zone(z):
    if z is None or not np.isfinite(z):
        return "нет данных"
    return "безопасная" if z > 5.85 else ("серая" if z >= 4.15 else "бедствия")


def dynamics(cur_r, prev_r, cur, prev):
    if prev_r is None:
        return {k: np.nan for k in DYN_FEATS}
    d = lambda k: cur_r.get(k, np.nan) - prev_r.get(k, np.nan)  # noqa: E731
    rev, rev0 = _g(cur, 2110), _g(prev, 2110)
    ta, ta0 = _g(cur, 1600), _g(prev, 1600)
    return {
        "d_equity_ratio": d("equity_ratio"),
        "d_current_ratio": d("current_ratio"),
        "d_debt_to_assets": d("debt_to_assets"),
        "d_roa": d("roa"),
        "rev_growth": np.clip(rev / rev0 - 1, -1, 5) if (rev0 and np.isfinite(rev0) and rev0 > 0) else np.nan,
        "asset_growth": np.clip(ta / ta0 - 1, -1, 5) if (ta0 and np.isfinite(ta0) and ta0 > 0) else np.nan,
        "d_interest_coverage": d("interest_coverage"),
    }


# Шкала рейтингов (национальная шкала АКРА/Эксперт РА/НКР/НРА)
_SCALE = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
          "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"]


def rating_to_num(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).upper().replace("RU", "").replace("(", "").replace(")", "").replace(".RU", "")
    s = s.replace("|", "").replace(".", "").strip()
    s = s.split()[0] if s else s
    for i, k in sorted(enumerate(_SCALE), key=lambda t: -len(t[1])):
        if s.startswith(k):
            return float(i + 1)
    if s in ("RD", "SD", "SEL"):
        return 22.0
    return np.nan
