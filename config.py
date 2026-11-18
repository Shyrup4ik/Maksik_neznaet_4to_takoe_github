"""Общие настройки проекта: даты, горизонт, пути, определение дефолта."""
from pathlib import Path
import pandas as pd

import os

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("PD_DATA_DIR", ROOT / "data"))
RAW = DATA / "raw"
MODELS = Path(os.environ.get("PD_MODELS_DIR", ROOT / "models"))
for p in (RAW, MODELS):
    p.mkdir(parents=True, exist_ok=True)

# --- Период наблюдений (квартальные срезы) ---
START_DATE = "2019-03-31"
TODAY = pd.Timestamp.today().normalize()

# --- Целевая переменная ---
HORIZON_DAYS = 182          # дефолт в течение ~6 месяцев после даты среза
# "broad"  — любой технический или реальный дефолт (неисполнение в срок купона/номинала/оферты)
# "strict" — только реальный дефолт (техдефолт, не исправленный в льготный период)
TARGET_MODE = "broad"

# --- Out-of-time разбиение ---
TRAIN_END = "2024-12-31"            # срезы до этой даты включительно -> обучение
TEST_START = "2025-01-01"           # срезы с этой даты -> тест (если метка уже известна)
FORCE_TEST_COMPANIES = ["Самолет"]  # эти компании целиком уходят в тест, в обучение не попадают

# --- Отсутствие заглядывания в будущее ---
# Годовая РСБУ за год Y считается доступной с 1 апреля года Y+1 (срок сдачи в ФНС — 31 марта).
ANNUAL_REPORT_LAG_DAYS = 91
# Ручная квартальная отчётность (data/financials_manual.csv) — по колонке publish_date,
# если её нет — period_end + 60 дней.
MANUAL_REPORT_LAG_DAYS = 60

# Рыночные данные: окно для расчёта динамики цены/доходности
MARKET_WINDOW_DAYS = 91
MAX_STALE_PRICE_DAYS = 30   # последняя сделка не старше N дней — иначе рыночных признаков нет

# --- Инвестиционное правило ---
LGD = 0.70                  # потери при дефолте (ВДО в РФ: возвраты обычно 20–40% номинала)
RISK_FREE_ANNUAL = 0.165    # безрисковая ставка, годовых (поменяйте на актуальную ключевую ставку/ОФЗ)
MAX_WEIGHT = 0.10           # максимальная доля капитала в одного эмитента
KELLY_FRACTION = 0.5        # доля от «полного Келли»

ISS = "https://iss.moex.com/iss"
