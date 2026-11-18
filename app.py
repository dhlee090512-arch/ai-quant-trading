from datetime import datetime, timedelta
import gc
import json
import os
import traceback
from zoneinfo import ZoneInfo
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

KST = ZoneInfo("Asia/Seoul")


def now_kst_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
  return f"{datetime.now(KST).strftime(fmt)} KST"


# 페이지 기본 설정
st.set_page_config(
    page_title="AI 퀀트 트레이딩 대시보드", page_icon="📈", layout="wide"
)

st.title("🚀 AI 기반 퀀트 트레이딩 시스템")
st.write(
    "LightGBM 3중 앙상블 모델 기반 실전 분석 대시보드 (테마 롤링 가산점 &"
    " 목표 익절 선달성 확률 동기화)"
)

# 야후 파이낸스 차단 우회를 위한 세션 설정
session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
        " like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
})

# 로그 기준 상장폐지/인수합병/티커 변경 등으로 실패하는 티커 제외 목록
FAILED_OR_DELISTED_TICKERS: set[str] = {
    "DFS",
    "ACCD",
    "EXAS",
    "FOLD",
    "MASI",
    "MMC",
    "ANSS",
    "GPS",
    "JNPR",
    "BK",
    "FI",
    "HOLX",
    "SGEN",
    "CTLT",
    "JWN",
}

# 기존 model_config에 없는 종목만 테마별 35개씩(총 140개) 신규 추가
RAW_NEW_TICKERS_BY_THEME: dict[str, list[str]] = {
    "IT": [
        "ASML",
        "TSM",
        "ARM",
        "MRVL",
        "INTC",
        "GFS",
        "UMC",
        "ASX",
        "STM",
        "SAP",
        "SHOP",
        "SPOT",
        "UBER",
        "DASH",
        "ABNB",
        "RBLX",
        "ZM",
        "TWLO",
        "ESTC",
        "GTLB",
        "VRT",
        "DELL",
        "CIEN",
        "LITE",
        "COHR",
        "MKSI",
        "ENTG",
        "ONTO",
        "ACLS",
        "FORM",
        "CRUS",
        "SLAB",
        "SYNA",
        "ALGM",
        "DIOD",
    ],
    "Finance": [
        "BX",
        "KKR",
        "APO",
        "ARES",
        "CG",
        "OWL",
        "LPLA",
        "IBKR",
        "TW",
        "MKTX",
        "TRU",
        "EFX",
        "AFRM",
        "UPST",
        "XYZ",
        "MELI",
        "NU",
        "PAGS",
        "STNE",
        "XP",
        "VIRT",
        "SEIC",
        "EVR",
        "MC",
        "LAZ",
        "PIPR",
        "SF",
        "JEF",
        "STEP",
        "HLNE",
        "TPG",
        "BN",
        "AMG",
        "VOYA",
        "EQH",
    ],
    "Healthcare": [
        "NVO",
        "AZN",
        "GSK",
        "SNY",
        "TAK",
        "BNTX",
        "NVAX",
        "ALNY",
        "BMRN",
        "SRPT",
        "IONS",
        "EXEL",
        "ARGX",
        "ACAD",
        "FOLD",
        "CRSP",
        "EDIT",
        "NTLA",
        "BEAM",
        "PACB",
        "TWST",
        "TXG",
        "DOCS",
        "TDOC",
        "HIMS",
        "ACCD",
        "AMN",
        "THC",
        "INMD",
        "TNDM",
        "IRTC",
        "LIVN",
        "MASI",
        "MEDP",
        "ICLR",
    ],
    "Defensive": [
        "UL",
        "DEO",
        "BUD",
        "BTI",
        "SAM",
        "CCEP",
        "KOF",
        "FMX",
        "BJ",
        "ACI",
        "SFM",
        "CHWY",
        "W",
        "ETSY",
        "EBAY",
        "JD",
        "BABA",
        "PDD",
        "AEO",
        "ANF",
        "GPS",
        "M",
        "KSS",
        "JWN",
        "BBY",
        "ULTA",
        "FIVE",
        "OLLI",
        "USFD",
        "PFGC",
        "CHEF",
        "SMPL",
        "POST",
        "NOMD",
        "ARMK",
    ],
}


def build_clean_new_tickers_by_theme() -> dict[str, list[str]]:
  cleaned: dict[str, list[str]] = {}
  seen_new: set[str] = set()
  for theme_name, tickers in RAW_NEW_TICKERS_BY_THEME.items():
    theme_list: list[str] = []
    for tk in tickers:
      norm_tk = str(tk).strip().upper()
      if (
          norm_tk
          and norm_tk != "IT"
          and norm_tk not in FAILED_OR_DELISTED_TICKERS
          and norm_tk not in seen_new
      ):
        seen_new.add(norm_tk)
        theme_list.append(norm_tk)
    cleaned[theme_name] = theme_list
  return cleaned


NEW_TICKERS_BY_THEME = build_clean_new_tickers_by_theme()
NEW_THEME_MAP: dict[str, str] = {
    tk: theme_name
    for theme_name, tickers in NEW_TICKERS_BY_THEME.items()
    for tk in tickers
}
NEW_UNIVERSE_TICKERS: list[str] = list(NEW_THEME_MAP.keys())


def merge_config_with_new_tickers(config: dict, config_path: str) -> dict:
  updated_cfg = dict(config) if isinstance(config, dict) else {}

  merged_theme_map: dict[str, str] = {}
  raw_theme_map = updated_cfg.get("theme_map", {})
  if isinstance(raw_theme_map, dict):
    for k, v in raw_theme_map.items():
      norm_k = str(k).strip().upper()
      if (
          norm_k
          and norm_k != "IT"
          and norm_k not in FAILED_OR_DELISTED_TICKERS
          and v
      ):
        merged_theme_map[norm_k] = str(v)

  for tk, theme_name in NEW_THEME_MAP.items():
    if tk not in merged_theme_map:
      merged_theme_map[tk] = theme_name

  updated_cfg["theme_map"] = merged_theme_map

  norm_universe: list[str] = []
  seen_univ: set[str] = set()
  raw_universe = updated_cfg.get("universe_tickers", [])
  if isinstance(raw_universe, list):
    for tk in raw_universe:
      norm_tk = str(tk).strip().upper()
      if (
          norm_tk
          and norm_tk != "IT"
          and norm_tk not in FAILED_OR_DELISTED_TICKERS
          and norm_tk not in seen_univ
      ):
        seen_univ.add(norm_tk)
        norm_universe.append(norm_tk)

  for tk in NEW_UNIVERSE_TICKERS:
    if tk not in seen_univ:
      seen_univ.add(tk)
      norm_universe.append(tk)

  updated_cfg["universe_tickers"] = norm_universe

  try:
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
      json.dump(updated_cfg, f, ensure_ascii=False, indent=2)
  except Exception:
    pass

  return updated_cfg


# 1. 깃허브에 저장된 모델, 피처 목록 및 설정 메타데이터 로드 (캐싱 적용)
@st.cache_resource
def load_models_and_metadata():
  try:
    ranker = lgb.Booster(model_file="models/ranker_model.txt")
    classifier = lgb.Booster(model_file="models/classifier_model.txt")
    tp_reg = lgb.Booster(model_file="models/tp_regressor_model.txt")

    with open("models/feature_cols.json", "r", encoding="utf-8") as f:
      feature_cols = json.load(f)

    config_path = "models/model_config.json"
    if os.path.exists(config_path):
      with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    else:
      raise FileNotFoundError(
          "models/model_config.json 파일을 찾을 수 없습니다. 코랩에서 셀 7을"
          " 실행해 메타데이터를 업로드해주세요."
      )

    config = merge_config_with_new_tickers(config, config_path)
    return ranker, classifier, tp_reg, feature_cols, config, None
  except Exception as e:
    return None, None, None, None, None, str(e)


ranker_lgb, classifier_lgb, tp_regressor, FEATURE_COLS, MODEL_CONFIG, load_error = (
    load_models_and_metadata()
)

# 사이드바 메뉴 구성
menu = st.sidebar.selectbox(
    "메뉴 선택",
    [
        "🔥 당일 종목 추천 (스캐너)",
        "📅 과거 날짜 종목 추천 (스캐너)",
        "🔍 특정 종목 당일 분석",
        "⏳ 특정 종목 과거 시점 분석",
        "🛡️ 기존 보유 포지션 진단 및 매매 가이드",
    ],
)

# 공통 카드 CSS 스타일 정의
METRIC_CSS = """
<style>
.metric-container {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    margin-top: 15px;
    margin-bottom: 20px;
}
.metric-card {
    background-color: #1E1E1E;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 15px;
    flex: 1;
    text-align: center;
}
.metric-title {
    font-size: 14px;
    color: #AAAAAA;
    margin-bottom: 5px;
}
.metric-value {
    font-size: 20px;
    font-weight: bold;
    color: #FFFFFF;
}
.metric-sub {
    font-size: 13px;
    margin-top: 6px;
    font-weight: 600;
}
.status-pass {
    color: #00FF7F;
}
.status-fail {
    color: #FF6B6B;
}
.status-warn {
    color: #FBBF24;
}
.pick-box {
    background: linear-gradient(180deg, #1A1D24 0%, #14171D 100%);
    border: 1px solid #2E3544;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 16px;
}
.pick-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #2A303C;
    padding-bottom: 10px;
    margin-bottom: 14px;
}
.pick-rank-ticker {
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF;
}
.pick-badges {
    display: flex;
    align-items: center;
    gap: 8px;
}
.pick-score {
    background-color: rgba(56, 189, 248, 0.14);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 13px;
    font-weight: 700;
}
.pick-surv {
    background-color: rgba(0, 255, 127, 0.12);
    color: #00FF7F;
    border: 1px solid rgba(0, 255, 127, 0.35);
    border-radius: 999px;
    padding: 4px 12px;
    font-size: 13px;
    font-weight: 700;
}
.pick-atr {
    background-color: rgba(251, 191, 36, 0.14);
    color: #FBBF24;
    border: 1px solid rgba(251, 191, 36, 0.35);
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 13px;
    font-weight: 700;
}
.pick-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}
.pick-item {
    background-color: #111318;
    border: 1px solid #232834;
    border-radius: 8px;
    padding: 12px;
}
.pick-item-label {
    font-size: 12px;
    color: #9AA4B2;
    margin-bottom: 4px;
}
.pick-item-val {
    font-size: 17px;
    font-weight: 700;
    color: #F3F4F6;
}
.pick-sl {
    color: #FF6B6B;
    font-size: 13px;
    font-weight: 600;
    margin-left: 4px;
}
.pick-tp {
    color: #4ECDC4;
    font-size: 13px;
    font-weight: 600;
    margin-left: 4px;
}
</style>
"""


def render_fixed_candlestick(
    ohlc_df: pd.DataFrame,
    title: str,
    sl_price: float,
    tp_price: float,
    entry_price: float | None = None,
    buy_ts: pd.Timestamp | None = None,
    expiry_ts: pd.Timestamp | None = None,
    height: int = 440,
):
  if ohlc_df is None or ohlc_df.empty:
    st.warning("캔들차트를 그릴 데이터가 없습니다.")
    return

  df_c = ohlc_df.copy()
  df_c["Date"] = pd.to_datetime(df_c["Date"])
  df_c = df_c.sort_values("Date").reset_index(drop=True)

  for col in ["Open", "High", "Low", "Close"]:
    if col not in df_c.columns:
      df_c[col] = df_c["Close"]
    df_c[col] = pd.to_numeric(df_c[col], errors="coerce")

  df_c = df_c.dropna(subset=["Open", "High", "Low", "Close"])
  if df_c.empty:
    st.warning("유효한 OHLC 캔들 데이터가 없습니다.")
    return

  fig = go.Figure()

  fig.add_trace(
      go.Candlestick(
          x=df_c["Date"],
          open=df_c["Open"],
          high=df_c["High"],
          low=df_c["Low"],
          close=df_c["Close"],
          name="주가",
          increasing_line_color="#00FF7F",
          increasing_fillcolor="rgba(0, 255, 127, 0.35)",
          decreasing_line_color="#FF6B6B",
          decreasing_fillcolor="rgba(255, 107, 107, 0.35)",
      )
  )

  low_bound = min(sl_price, tp_price)
  high_bound = max(sl_price, tp_price)
  fig.add_hrect(
      y0=low_bound,
      y1=high_bound,
      fillcolor="rgba(78, 205, 196, 0.08)",
      line_width=0,
  )

  if entry_price is not None:
    fig.add_hrect(
        y0=sl_price,
        y1=entry_price,
        fillcolor="rgba(255, 107, 107, 0.09)",
        line_width=0,
    )
    fig.add_hrect(
        y0=entry_price,
        y1=tp_price,
        fillcolor="rgba(78, 205, 196, 0.11)",
        line_width=0,
    )

  fig.add_hline(
      y=tp_price,
      line_dash="dash",
      line_width=1.6,
      line_color="#4ECDC4",
      annotation_text=f"익절가 ${tp_price:.2f}",
      annotation_position="top left",
      annotation_font=dict(color="#4ECDC4", size=11),
  )

  fig.add_hline(
      y=sl_price,
      line_dash="dash",
      line_width=1.6,
      line_color="#FF6B6B",
      annotation_text=f"손절가 ${sl_price:.2f}",
      annotation_position="bottom left",
      annotation_font=dict(color="#FF6B6B", size=11),
  )

  if buy_ts is not None:
    b_ts = pd.Timestamp(buy_ts)
    fig.add_shape(
        type="line",
        x0=b_ts,
        x1=b_ts,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="#38BDF8", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=b_ts,
        y=1.02,
        xref="x",
        yref="paper",
        text="매수 시점",
        showarrow=False,
        font=dict(color="#38BDF8", size=11),
        bgcolor="rgba(17, 19, 24, 0.85)",
    )

  if expiry_ts is not None:
    e_ts = pd.Timestamp(expiry_ts)
    fig.add_shape(
        type="line",
        x0=e_ts,
        x1=e_ts,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="#FBBF24", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=e_ts,
        y=1.02,
        xref="x",
        yref="paper",
        text="보유 만기 시점",
        showarrow=False,
        font=dict(color="#FBBF24", size=11),
        bgcolor="rgba(17, 19, 24, 0.85)",
    )

  y_min = min(float(df_c["Low"].min()), sl_price, tp_price)
  y_max = max(float(df_c["High"].max()), sl_price, tp_price)
  pad = max((y_max - y_min) * 0.08, 1.0)

  fig.update_layout(
      title=dict(text=title, font=dict(size=15, color="#F3F4F6")),
      height=height,
      paper_bgcolor="#111318",
      plot_bgcolor="#14171D",
      font=dict(color="#D1D5DB"),
      margin=dict(l=10, r=20, t=55, b=15),
      showlegend=False,
      xaxis=dict(
          type="date",
          fixedrange=True,
          rangeslider=dict(visible=False),
          gridcolor="#232834",
          showgrid=True,
      ),
      yaxis=dict(
          range=[y_min - pad, y_max + pad],
          fixedrange=True,
          gridcolor="#232834",
          showgrid=True,
      ),
      dragmode=False,
  )

  st.plotly_chart(
      fig,
      width="stretch",
      config={"staticPlot": True, "displayModeBar": False},
  )


def normalize_ohlcv_df(raw_df: pd.DataFrame) -> pd.DataFrame:
  if raw_df is None or raw_df.empty:
    return pd.DataFrame()

  df = raw_df.copy()
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  df = df.reset_index()
  col_mapping = {}
  for c in df.columns:
    lc = str(c).lower()
    if "date" in lc or "datetime" in lc:
      col_mapping[c] = "Date"
    elif "open" in lc:
      col_mapping[c] = "Open"
    elif "high" in lc:
      col_mapping[c] = "High"
    elif "low" in lc:
      col_mapping[c] = "Low"
    elif "close" in lc and "adj" not in lc:
      col_mapping[c] = "Close"
    elif "volume" in lc:
      col_mapping[c] = "Volume"
  df = df.rename(columns=col_mapping)

  if "Date" not in df.columns:
    df["Date"] = df.iloc[:, 0]

  df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
  for col in ["Open", "High", "Low", "Close", "Volume"]:
    if col in df.columns:
      df[col] = pd.to_numeric(df[col], errors="coerce")

  df = df.dropna(subset=["Close"]).sort_values("Date").reset_index(drop=True)
  return df


def build_market_tables(df_fresh: pd.DataFrame):
  if df_fresh is None or df_fresh.empty or "Ticker" not in df_fresh.columns:
    return pd.DataFrame(
        columns=["Date", "VIX", "VIX_Change_1d", "VIX_Change_5d", "VIX_Pos_20d"]
    ), pd.DataFrame(columns=["Date", "SPY_Ret_5d"])

  market_mask = df_fresh["Ticker"].str.upper().isin(["^VIX", "VIX", "SPY"])
  market_data = df_fresh[market_mask][["Date", "Ticker", "Close"]].copy()
  market_data["Ticker"] = (
      market_data["Ticker"].str.upper().str.replace("^", "", regex=False)
  )

  if not market_data.empty and "Date" in market_data.columns:
    market_data = (
        market_data.pivot(index="Date", columns="Ticker", values="Close")
        .reset_index()
    )
    market_data.columns.name = None
    market_data = market_data.sort_values("Date").ffill()

    if "VIX" in market_data.columns:
      market_data["VIX_Change_1d"] = market_data["VIX"].pct_change(1)
      market_data["VIX_Change_5d"] = market_data["VIX"].pct_change(5)
      roll_min = market_data["VIX"].rolling(20).min()
      roll_max = market_data["VIX"].rolling(20).max()
      market_data["VIX_Pos_20d"] = (market_data["VIX"] - roll_min) / (
          roll_max - roll_min + 1e-8
      )
    else:
      market_data["VIX"] = 18.5
      market_data["VIX_Change_1d"] = 0.0
      market_data["VIX_Change_5d"] = 0.0
      market_data["VIX_Pos_20d"] = 0.5

    if "SPY" in market_data.columns:
      spy_df = market_data[["Date", "SPY"]].copy()
      spy_df["SPY_Ret_5d"] = spy_df["SPY"].pct_change(5)
    else:
      spy_df = pd.DataFrame({"Date": market_data["Date"], "SPY_Ret_5d": 0.0})
  else:
    market_data = pd.DataFrame({
        "Date": df_fresh["Date"].unique(),
        "VIX": 18.5,
        "VIX_Change_1d": 0.0,
        "VIX_Change_5d": 0.0,
        "VIX_Pos_20d": 0.5,
    })
    spy_df = pd.DataFrame(
        {"Date": df_fresh["Date"].unique(), "SPY_Ret_5d": 0.0}
    )

  return market_data, spy_df


def compute_stock_factors(group: pd.DataFrame) -> pd.DataFrame:
  g = group.copy().sort_values("Date").reset_index(drop=True)
  if len(g) < 100:
    return pd.DataFrame()

  o, h, l, c, v = g["Open"], g["High"], g["Low"], g["Close"], g["Volume"]
  prev_c = c.shift(1)
  candle_range = (h - l) + 1e-8

  g["Body_Ratio"] = (c - o) / candle_range
  g["Upper_Shadow"] = (h - np.maximum(o, c)) / candle_range
  g["Lower_Shadow"] = (np.minimum(o, c) - l) / candle_range
  g["Close_Pos"] = (c - l) / candle_range
  g["Gap_Pct"] = (o - prev_c) / (prev_c + 1e-8)

  for win in [5, 20, 60]:
    r_min = l.rolling(win).min()
    r_max = h.rolling(win).max()
    g[f"Pos_Box_{win}d"] = (c - r_min) / (r_max - r_min + 1e-8)
    g[f"Ret_{win}d"] = c.pct_change(win)

  g["Vol_Ratio_5d"] = v / (v.rolling(5).mean() + 1e-8)
  g["Vol_Ratio_20d"] = v / (v.rolling(20).mean() + 1e-8)
  g["Vol_Trend_5_20"] = v.rolling(5).mean() / (v.rolling(20).mean() + 1e-8)

  tr = pd.concat(
      [h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1
  ).max(axis=1)
  atr14 = tr.rolling(14).mean()
  g["ATR_Pct"] = atr14 / (c + 1e-8)
  g["Range_Expansion"] = candle_range / (atr14 + 1e-8)

  ma5 = c.rolling(5).mean()
  ma20 = c.rolling(20).mean()
  ma60 = c.rolling(60).mean()
  g["Dist_MA5"] = (c - ma5) / (ma5 + 1e-8)
  g["Dist_MA20"] = (c - ma20) / (ma20 + 1e-8)
  g["Dist_MA60"] = (c - ma60) / (ma60 + 1e-8)

  typical_price = (h + l + c) / 3.0
  vwap = (v * typical_price).rolling(20).sum() / (v.rolling(20).sum() + 1e-8)
  g["VWAP_Ratio"] = (c - vwap) / (vwap + 1e-8)

  delta = c.diff()
  gain = (delta.where(delta > 0, 0)).rolling(14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
  rs = gain / (loss + 1e-8)
  g["RSI_14"] = 100 - (100 / (1 + rs))

  return g


def attach_theme_features(df_in: pd.DataFrame, config: dict) -> pd.DataFrame:
  df_out = df_in.copy()
  cfg_theme_map = config.get("theme_map", {})
  merged_theme_map = {
      **{str(k).strip().upper(): str(v) for k, v in cfg_theme_map.items()},
      **NEW_THEME_MAP,
  }

  if "Theme" not in df_out.columns:
    df_out["Theme"] = (
        df_out["Ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(merged_theme_map)
        .fillna("Defensive")
    )
  else:
    df_out["Theme"] = df_out["Theme"].fillna(
        df_out["Ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .map(merged_theme_map)
        .fillna("Defensive")
    )

  if "Ret_5d" in df_out.columns:
    df_out["Theme_Ret_5d"] = (
        df_out.groupby(["Date", "Theme"])["Ret_5d"].transform("mean")
    )
  else:
    df_out["Theme_Ret_5d"] = 0.0

  if "Ret_20d" in df_out.columns:
    df_out["Theme_Ret_20d"] = (
        df_out.groupby(["Date", "Theme"])["Ret_20d"].transform("mean")
    )
  else:
    df_out["Theme_Ret_20d"] = 0.0

  if "Dist_MA20" in df_out.columns:
    df_out["Theme_Dist_MA20"] = (
        df_out.groupby(["Date", "Theme"])["Dist_MA20"].transform("mean")
    )
  else:
    df_out["Theme_Dist_MA20"] = 0.0

  df_out["Theme_Ret_5d_Zscore"] = df_out.groupby("Date")["Theme_Ret_5d"].transform(
      lambda x: (x - x.mean()) / (x.std() + 1e-8)
  )
  df_out["Rel_Strength_Theme_5d"] = df_out["Ret_5d"] - df_out["Theme_Ret_5d"]
  return df_out


def build_feature_matrix(
    df_sub: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
  mat = pd.DataFrame(index=df_sub.index)
  for col in feature_cols:
    if col in df_sub.columns:
      mat[col] = pd.to_numeric(df_sub[col], errors="coerce").fillna(0.0)
    else:
      mat[col] = 0.0
  return mat


def score_candidates(
    df_sub: pd.DataFrame, config: dict, feature_cols: list[str]
) -> pd.DataFrame:
  scored = df_sub.copy()
  baseline_rate = float(config.get("baseline_tp_rate", 0.35))
  boost_weight = float(config.get("boost_weight", 8.0))
  theme_roll_rates = config.get("theme_roll_tp_rates", {})

  scored["Theme_Roll_TP_Rate"] = (
      scored["Theme"].map(theme_roll_rates).fillna(baseline_rate).astype(float)
  )
  scored["Theme_Boost"] = (
      scored["Theme_Roll_TP_Rate"] - baseline_rate
  ) * boost_weight

  x_mat = build_feature_matrix(scored, feature_cols)
  scored["Pred_Score"] = np.asarray(ranker_lgb.predict(x_mat), dtype=float)
  scored["Adjusted_Pred_Score"] = scored["Pred_Score"] + scored["Theme_Boost"]

  cls_raw = np.asarray(classifier_lgb.predict(x_mat), dtype=float)
  if cls_raw.ndim == 2:
    scored["TP_Hit_Prob"] = np.clip(cls_raw[:, -1], 0.0, 1.0)
  else:
    scored["TP_Hit_Prob"] = np.clip(cls_raw, 0.0, 1.0)

  scored["Predicted_TP"] = np.asarray(tp_regressor.predict(x_mat), dtype=float)
  return scored


def compute_trade_levels(
    entry_price: float, atr_pct: float, predicted_tp: float, config: dict
):
  max_sl_limit = float(config.get("max_sl_limit", -8.0))
  min_tp_rr = float(config.get("min_tp_rr", 1.5))

  atr_pct_val = float(atr_pct) * 100.0
  sl_pct = max(-(atr_pct_val * 1.5), max_sl_limit)
  min_required_tp = abs(sl_pct) * min_tp_rr
  tp_pct = max(float(predicted_tp), min_required_tp)

  sl_price = entry_price * (1.0 + sl_pct / 100.0)
  tp_price = entry_price * (1.0 + tp_pct / 100.0)
  return sl_price, sl_pct, tp_price, tp_pct


def render_scanner_view(
    df_clean: pd.DataFrame,
    target_date_ts: pd.Timestamp,
    top_n_percent: float,
    min_tp_hit_prob: float,
    max_atr_pct: float,
    config: dict,
    feature_cols: list[str],
    date_label_prefix: str,
    last_fetched_kst: str | None = None,
    requested_date_str: str | None = None,
):
  run_time_kst = now_kst_str()
  fetched_str = last_fetched_kst or run_time_kst
  df_eligible = df_clean[df_clean["Date"] <= target_date_ts].copy()

  if df_eligible.empty:
    st.warning(
        f"⚠️ 선택한 기준일({requested_date_str or target_date_ts.strftime('%Y-%m-%d')}) "
        "이하의 유효한 시세 데이터가 없습니다."
    )
    return

  effective_date = df_eligible["Date"].max()
  effective_date_str = pd.to_datetime(effective_date).strftime("%Y-%m-%d")
  scan_timestamp = (
      f"{effective_date_str} | 마지막 데이터 다운로드: {fetched_str}"
  )

  if requested_date_str and requested_date_str != effective_date_str:
    st.caption(
        f"ℹ️ 요청하신 `{requested_date_str}`은 휴장일/미집계일이므로, "
        f"직전 유효 거래일인 **`{effective_date_str}`** 기준으로 조회했습니다."
    )

  latest_subset = df_eligible[df_eligible["Date"] == effective_date].copy()
  if latest_subset.empty:
    st.warning("⚠️ 해당 기준일에 조건을 충족하는 유효한 데이터가 없습니다.")
    return

  scored_latest = score_candidates(latest_subset, config, feature_cols)
  group_sorted = scored_latest.sort_values(
      by="Adjusted_Pred_Score", ascending=False
  ).reset_index(drop=True)

  top_k = max(1, int(len(group_sorted) * (top_n_percent / 100.0)))
  top_cutoff_score = float(
      group_sorted.iloc[top_k - 1]["Adjusted_Pred_Score"]
  )
  avg_universe_score = float(group_sorted["Adjusted_Pred_Score"].mean())

  top10_rows = []
  for r_idx, t_row in group_sorted.head(10).iterrows():
    rank_pos = r_idx + 1
    t_entry = float(t_row["Close"])
    t_atr_pct = float(t_row["ATR_Pct"]) * 100.0
    t_sl_p, t_sl_pct, t_tp_p, t_tp_pct = compute_trade_levels(
        t_entry,
        float(t_row["ATR_Pct"]),
        float(t_row["Predicted_TP"]),
        config,
    )
    t_prob_val = float(t_row["TP_Hit_Prob"])
    in_top_k = rank_pos <= top_k
    prob_ok = t_prob_val >= min_tp_hit_prob
    atr_ok = t_atr_pct < max_atr_pct

    if in_top_k and prob_ok and atr_ok:
      status_badge = "선별 충족 ✅"
    elif in_top_k and not atr_ok:
      status_badge = f"ATR {max_atr_pct:.1f}% 초과 ❌"
    elif in_top_k and not prob_ok:
      status_badge = "상위컷 충족·확률 미달 ⚠️"
    else:
      status_badge = "상위컷 외 ❌"

    top10_rows.append({
        "순위": rank_pos,
        "Ticker": t_row["Ticker"],
        "Theme": str(t_row.get("Theme", "-")),
        "조정 점수": round(float(t_row["Adjusted_Pred_Score"]), 4),
        "기본 점수": round(float(t_row["Pred_Score"]), 4),
        "테마가산점": f"{float(t_row['Theme_Boost']):+.3f}",
        "ATR(%)": f"{t_atr_pct:.2f}%",
        "진입가($)": round(t_entry, 2),
        "손절가($)": round(t_sl_p, 2),
        "손절폭(%)": f"{t_sl_pct:.2f}%",
        "익절가($)": round(t_tp_p, 2),
        "익절폭(%)": f"+{t_tp_pct:.2f}%",
        "익절선달성률": f"{t_prob_val * 100.0:.1f}%",
        "선별 상태": status_badge,
    })
  df_top10_all = pd.DataFrame(top10_rows)

  top_candidates = group_sorted.head(top_k).copy()
  top_candidates["ATR_Pct_Val"] = (
      pd.to_numeric(top_candidates["ATR_Pct"], errors="coerce") * 100.0
  )
  filtered_latest = top_candidates[
      (top_candidates["TP_Hit_Prob"] >= min_tp_hit_prob)
      & (top_candidates["ATR_Pct_Val"] < max_atr_pct)
  ].reset_index(drop=True)

  df_display = pd.DataFrame()
  avg_filtered_score = None

  if len(filtered_latest) > 0:
    avg_filtered_score = float(
        filtered_latest["Adjusted_Pred_Score"].mean()
    )
    display_results = []

    for idx, row in filtered_latest.iterrows():
      rank_num = idx + 1
      entry_price = float(row["Close"])
      atr_pct_val = float(row["ATR_Pct"]) * 100.0
      sl_price, sl_pct, tp_price, tp_pct = compute_trade_levels(
          entry_price,
          float(row["ATR_Pct"]),
          float(row["Predicted_TP"]),
          config,
      )
      tp_prob_pct = float(row["TP_Hit_Prob"]) * 100.0
      adj_score_val = float(row["Adjusted_Pred_Score"])
      boost_val = float(row["Theme_Boost"])
      theme_val = str(row.get("Theme", "-"))

      display_results.append({
          "순위": rank_num,
          "Ticker": row["Ticker"],
          "Theme": theme_val,
          "조정 점수": round(adj_score_val, 4),
          "테마가산점": f"{boost_val:+.3f}",
          "ATR(%)": f"{atr_pct_val:.2f}%",
          "진입가($)": round(entry_price, 2),
          "손절가($)": round(sl_price, 2),
          "손절폭(%)": f"{sl_pct:.2f}%",
          "익절가($)": round(tp_price, 2),
          "익절폭(%)": f"+{tp_pct:.2f}%",
          "익절선달성률": f"{tp_prob_pct:.1f}%",
      })

    df_display = pd.DataFrame(display_results)
    st.success(
        f"✨ 기준(상위 {top_n_percent:.0f}% 랭킹, 익절선달성률"
        f" {min_tp_hit_prob*100:.0f}% 이상, ATR < {max_atr_pct:.1f}%)"
        f" 충족 종목 {len(df_display)}개 선별 완료!"
    )
  else:
    st.warning(
        f"⚠️ 기준(상위 {top_n_percent:.0f}% 랭킹, 익절선달성률"
        f" {min_tp_hit_prob*100:.0f}% 이상, ATR < {max_atr_pct:.1f}%)"
        "을 충족하는 종목이 없습니다. (관망)"
    )

  st.markdown(f"📅 **{date_label_prefix}:** `{scan_timestamp}`")

  if df_display is not None and not df_display.empty:
    st.markdown(METRIC_CSS, unsafe_allow_html=True)

    cards_per_row = 2
    num_picks = len(df_display)
    for start_idx in range(0, num_picks, cards_per_row):
      cols = st.columns(cards_per_row)
      for offset in range(cards_per_row):
        row_idx = start_idx + offset
        if row_idx >= num_picks:
          continue
        r = df_display.iloc[row_idx]
        with cols[offset]:
          box_html = (
              '<div class="pick-box">'
              '<div class="pick-header">'
              f'<div class="pick-rank-ticker">#{r["순위"]} {r["Ticker"]} <span style="font-size:13px; color:#9AA4B2;">[{r["Theme"]}]</span></div>'
              '<div class="pick-badges">'
              f'<div class="pick-atr">ATR {r["ATR(%)"]}</div>'
              f'<div class="pick-score">조정점수 {r["조정 점수"]:.4f} ({r["테마가산점"]})</div>'
              f'<div class="pick-surv">익절선달성률 {r["익절선달성률"]}</div>'
              "</div>"
              "</div>"
              '<div class="pick-grid">'
              '<div class="pick-item">'
              '<div class="pick-item-label">진입가($)</div>'
              f'<div class="pick-item-val">${r["진입가($)"]:.2f}</div>'
              "</div>"
              '<div class="pick-item">'
              '<div class="pick-item-label">손절가 / 손절폭</div>'
              f'<div class="pick-item-val">${r["손절가($)"]:.2f}<span class="pick-sl">({r["손절폭(%)"].strip()})</span></div>'
              "</div>"
              '<div class="pick-item">'
              '<div class="pick-item-label">익절가 / 익절폭</div>'
              f'<div class="pick-item-val">${r["익절가($)"]:.2f}<span class="pick-tp">({r["익절폭(%)"].strip()})</span></div>'
              "</div>"
              "</div>"
              "</div>"
          )
          st.markdown(box_html, unsafe_allow_html=True)

    st.markdown("#### 📋 전체 추천 종목 요약 표")
    st.dataframe(df_display, width="stretch", hide_index=True)

    cutoff_str = (
        f"{top_cutoff_score:.4f}" if top_cutoff_score is not None else "-"
    )
    avg_rec_str = (
        f"{avg_filtered_score:.4f}"
        if avg_filtered_score is not None
        else "-"
    )
    avg_univ_str = (
        f"{avg_universe_score:.4f}"
        if avg_universe_score is not None
        else "-"
    )

    summary_cards_html = (
        '<div class="metric-container">'
        '<div class="metric-card">'
        f'<div class="metric-title">상위 {top_n_percent:.0f}% 조정 점수 (커트라인)</div>'
        f'<div class="metric-value" style="color:#38BDF8;">{cutoff_str}</div>'
        f'<div class="metric-sub" style="color:#AAAAAA;">기준일 전체 대상 상위 {top_n_percent:.0f}% 경계</div>'
        "</div>"
        '<div class="metric-card">'
        '<div class="metric-title">추천 종목 평균 조정 점수</div>'
        f'<div class="metric-value" style="color:#00FF7F;">{avg_rec_str}</div>'
        '<div class="metric-sub" style="color:#AAAAAA;">최종 선별 추천 종목 평균</div>'
        "</div>"
        '<div class="metric-card">'
        '<div class="metric-title">전체 대상 평균 조정 점수</div>'
        f'<div class="metric-value">{avg_univ_str}</div>'
        '<div class="metric-sub" style="color:#AAAAAA;">기준일 전체 대상 종목 평균</div>'
        "</div>"
        "</div>"
    )
    st.markdown(summary_cards_html, unsafe_allow_html=True)

  if df_top10_all is not None and not df_top10_all.empty:
    st.markdown("---")
    st.markdown("#### 🏆 기준일 조정 점수 상위 10개 종목 (진입 선별 기준 무관)")
    st.caption(
        f"기준일: {effective_date_str} | 마지막 데이터 다운로드: {fetched_str} | "
        f"Adjusted_Pred_Score 내림차순 상위 10개 (상위 {top_n_percent:.0f}% 컷 · "
        f"익절선달성률 ≥ {min_tp_hit_prob*100:.0f}% · ATR < {max_atr_pct:.1f}% 참고 표시)"
    )
    st.dataframe(df_top10_all, width="stretch", hide_index=True)


# 모델 로드 실패 시 에러 안내 문구 출력
if load_error is not None:
  st.error(
      "🚨 모델 파일 및 메타데이터 로딩 실패! GitHub 레포지토리의 `models/`"
      f" 폴더 상태를 확인해주세요.\n\n**에러 내용:** {load_error}"
  )
else:
  HOLDING_PERIOD = int(MODEL_CONFIG.get("holding_period", 7))
  TOP_N_PERCENT = float(MODEL_CONFIG.get("top_n_percent", 20.0))
  MIN_TP_HIT_PROB = float(MODEL_CONFIG.get("min_tp_hit_prob", 0.30))
  MAX_ATR_PCT = float(MODEL_CONFIG.get("max_atr_pct", 6.0))
  BOOST_WEIGHT = float(MODEL_CONFIG.get("boost_weight", 8.0))
  LOOKBACK_DAYS = int(MODEL_CONFIG.get("lookback_days", 40))
  UNIVERSE_TICKERS = [
      str(t).strip().upper()
      for t in MODEL_CONFIG.get("universe_tickers", NEW_UNIVERSE_TICKERS)
      if str(t).strip().upper()
      and str(t).strip().upper() != "IT"
      and str(t).strip().upper() not in FAILED_OR_DELISTED_TICKERS
  ]

  base_numeric_cols = [
      c
      for c in FEATURE_COLS
      if c
      not in {
          "Theme_Ret_5d",
          "Theme_Ret_20d",
          "Theme_Dist_MA20",
          "Theme_Ret_5d_Zscore",
          "Rel_Strength_Theme_5d",
      }
  ]

  DATA_START_DATE = "2022-01-01"

  @st.cache_data(ttl=3600, show_spinner=False)
  def fetch_and_process_universe_with_meta(
      universe_tuple: tuple[str, ...], start_date: str
  ):
    fetched_kst = now_kst_str()
    active_list = [
        t for t in universe_tuple if t not in FAILED_OR_DELISTED_TICKERS
    ]
    download_list = list(dict.fromkeys(active_list + ["^VIX", "SPY"]))
    df_raw_yf = yf.download(
        download_list,
        start=start_date,
        progress=False,
        auto_adjust=True,
        session=session,
    )

    if df_raw_yf is None or df_raw_yf.empty:
      return pd.DataFrame(), fetched_kst

    if isinstance(df_raw_yf.columns, pd.MultiIndex):
      df_fresh = df_raw_yf.stack(future_stack=True).reset_index()
      if "level_1" in df_fresh.columns:
        df_fresh = df_fresh.rename(columns={"level_1": "Ticker"})
    else:
      df_fresh = df_raw_yf.reset_index()

    df_fresh.columns = [
        str(col[0]).lower() if isinstance(col, tuple) else str(col).lower()
        for col in df_fresh.columns
    ]
    rename_map = {}
    for c in df_fresh.columns:
      if "date" in c:
        rename_map[c] = "Date"
      elif "ticker" in c or "level_1" in c:
        rename_map[c] = "Ticker"
      elif "open" in c:
        rename_map[c] = "Open"
      elif "high" in c:
        rename_map[c] = "High"
      elif "low" in c:
        rename_map[c] = "Low"
      elif "close" in c and "adj" not in c:
        rename_map[c] = "Close"
      elif "volume" in c:
        rename_map[c] = "Volume"
    df_fresh = df_fresh.rename(columns=rename_map)

    if "Date" in df_fresh.columns:
      df_fresh["Date"] = pd.to_datetime(
          df_fresh["Date"], errors="coerce"
      ).dt.tz_localize(None)
    if "Ticker" in df_fresh.columns:
      df_fresh["Ticker"] = df_fresh["Ticker"].astype(str).str.upper()
    df_fresh = df_fresh.dropna(subset=["Date", "Close"])

    market_data, spy_df = build_market_tables(df_fresh)

    df_stocks = df_fresh[
        ~df_fresh["Ticker"].isin(["VIX", "^VIX", "SPY"])
    ].copy()
    del df_raw_yf, df_fresh
    gc.collect()

    processed_dfs = []
    for _, group in df_stocks.groupby("Ticker"):
      g = compute_stock_factors(group)
      if not g.empty:
        processed_dfs.append(g)

    del df_stocks
    gc.collect()

    if not processed_dfs:
      return pd.DataFrame(), fetched_kst

    df_all = pd.concat(processed_dfs, ignore_index=True)
    del processed_dfs
    gc.collect()

    df_all = pd.merge(df_all, market_data, on="Date", how="inner")
    df_all = pd.merge(
        df_all, spy_df[["Date", "SPY_Ret_5d"]], on="Date", how="inner"
    )
    df_all["Rel_Strength_5d"] = df_all["Ret_5d"] - df_all["SPY_Ret_5d"]
    df_all["Ret_5d_Zscore"] = df_all.groupby("Date")["Ret_5d"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    df_all = attach_theme_features(df_all, MODEL_CONFIG)

    req_cols = [c for c in base_numeric_cols if c in df_all.columns]
    for col in req_cols:
      df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    df_clean = df_all.dropna(subset=req_cols).reset_index(drop=True)
    del df_all
    gc.collect()
    return df_clean, fetched_kst

  def fetch_and_process_universe_data():
    df_clean, _ = fetch_and_process_universe_with_meta(
        tuple(UNIVERSE_TICKERS), DATA_START_DATE
    )
    return df_clean

  @st.cache_data(ttl=3600, show_spinner=False)
  def fetch_single_ticker_with_market(ticker_upper: str):
    df_clean = fetch_and_process_universe_data()
    if not df_clean.empty:
      matched = df_clean[df_clean["Ticker"] == ticker_upper].copy()
      if not matched.empty:
        raw_single = yf.download(
            ticker_upper,
            start=DATA_START_DATE,
            progress=False,
            auto_adjust=True,
            session=session,
        )
        raw_norm = normalize_ohlcv_df(raw_single)
        return matched, raw_norm, True

    raw_single = yf.download(
        ticker_upper,
        start=DATA_START_DATE,
        progress=False,
        auto_adjust=True,
        session=session,
    )
    raw_norm = normalize_ohlcv_df(raw_single)
    if raw_norm.empty or len(raw_norm) < 100:
      return pd.DataFrame(), raw_norm, False

    raw_norm["Ticker"] = ticker_upper

    market_raw = yf.download(
        ["^VIX", "SPY"],
        start=DATA_START_DATE,
        progress=False,
        auto_adjust=True,
        session=session,
    )
    if isinstance(market_raw.columns, pd.MultiIndex):
      mkt_stacked = market_raw.stack(future_stack=True).reset_index()
      if "level_1" in mkt_stacked.columns:
        mkt_stacked = mkt_stacked.rename(columns={"level_1": "Ticker"})
      mkt_stacked.columns = [
          str(col[0]).lower() if isinstance(col, tuple) else str(col).lower()
          for col in mkt_stacked.columns
      ]
      mkt_rename = {}
      for c in mkt_stacked.columns:
        if "date" in c:
          mkt_rename[c] = "Date"
        elif "ticker" in c or "level_1" in c:
          mkt_rename[c] = "Ticker"
        elif "close" in c and "adj" not in c:
          mkt_rename[c] = "Close"
      mkt_stacked = mkt_stacked.rename(columns=mkt_rename)
      mkt_stacked["Date"] = pd.to_datetime(mkt_stacked["Date"]).dt.tz_localize(
          None
      )
      mkt_stacked["Ticker"] = mkt_stacked["Ticker"].astype(str).str.upper()
      market_data, spy_df = build_market_tables(mkt_stacked)
    else:
      market_data = pd.DataFrame({
          "Date": raw_norm["Date"],
          "VIX": 18.5,
          "VIX_Change_1d": 0.0,
          "VIX_Change_5d": 0.0,
          "VIX_Pos_20d": 0.5,
      })
      spy_df = pd.DataFrame({"Date": raw_norm["Date"], "SPY_Ret_5d": 0.0})

    g = compute_stock_factors(raw_norm)
    if g.empty:
      return pd.DataFrame(), raw_norm, False

    g = pd.merge(g, market_data, on="Date", how="inner")
    g = pd.merge(g, spy_df[["Date", "SPY_Ret_5d"]], on="Date", how="inner")
    g["Rel_Strength_5d"] = g["Ret_5d"] - g["SPY_Ret_5d"]
    g["Ret_5d_Zscore"] = 0.0
    g = attach_theme_features(g, MODEL_CONFIG)

    req_cols = [c for c in base_numeric_cols if c in g.columns]
    for col in req_cols:
      g[col] = pd.to_numeric(g[col], errors="coerce")

    g = g.dropna(subset=req_cols).reset_index(drop=True)
    return g, raw_norm, False

  # -------------------------------------------------------------
  # 1. 당일 종목 추천 (스캐너)
  # -------------------------------------------------------------
  if menu == "🔥 당일 종목 추천 (스캐너)":
    st.header(
        f"🔥 당일 기준 스윙 추천 스캐너 ({HOLDING_PERIOD}거래일 보유 / "
        f"Top {TOP_N_PERCENT:.0f}% · 익절선달성률 ≥ {MIN_TP_HIT_PROB*100:.0f}% · "
        f"ATR < {MAX_ATR_PCT:.1f}%)"
    )
    st.caption(
        f"테마 롤링 가산점 반영(Boost_W={BOOST_WEIGHT}, Lookback={LOOKBACK_DAYS}일) "
        f"| 유니버스: {len(UNIVERSE_TICKERS)}개 종목 "
        f"(theme_map: {len(MODEL_CONFIG.get('theme_map', {}))}개)"
    )

    with st.spinner(
        "유니버스 시세·테마 팩터 연산 및 3중 앙상블 추론 중..."
    ):
      try:
        df_clean, last_fetched_kst = fetch_and_process_universe_with_meta(
            tuple(UNIVERSE_TICKERS), DATA_START_DATE
        )
        st.caption(f"🕒 데이터 마지막 다운로드 시각: **{last_fetched_kst}**")
        if not df_clean.empty:
          latest_date = pd.Timestamp(df_clean["Date"].max())
          render_scanner_view(
              df_clean=df_clean,
              target_date_ts=latest_date,
              top_n_percent=TOP_N_PERCENT,
              min_tp_hit_prob=MIN_TP_HIT_PROB,
              max_atr_pct=MAX_ATR_PCT,
              config=MODEL_CONFIG,
              feature_cols=FEATURE_COLS,
              date_label_prefix="타겟 기준일 (최신 마감일)",
              last_fetched_kst=last_fetched_kst,
          )
        else:
          st.error("데이터 전처리 결과가 비어 있습니다.")
      except Exception as e:
        st.error(f"스캔 중 오류 발생: {e}")
        st.code(traceback.format_exc(), language="python")

  # -------------------------------------------------------------
  # 1-2. 과거 날짜 종목 추천 (스캐너)
  # -------------------------------------------------------------
  elif menu == "📅 과거 날짜 종목 추천 (스캐너)":
    st.header(
        f"📅 과거 날짜 기준 스윙 추천 스캐너 ({HOLDING_PERIOD}거래일 보유 / "
        f"Top {TOP_N_PERCENT:.0f}% · 익절선달성률 ≥ {MIN_TP_HIT_PROB*100:.0f}% · "
        f"ATR < {MAX_ATR_PCT:.1f}%)"
    )
    st.caption(
        f"테마 롤링 가산점 반영(Boost_W={BOOST_WEIGHT}, Lookback={LOOKBACK_DAYS}일) "
        f"| 유니버스: {len(UNIVERSE_TICKERS)}개 종목 "
        f"(theme_map: {len(MODEL_CONFIG.get('theme_map', {}))}개)"
    )

    default_past_scan_date = (datetime.now(KST) - timedelta(days=1)).date()
    selected_scan_date = st.date_input(
        "조회할 과거 기준일 선택",
        value=default_past_scan_date,
        max_value=datetime.now(KST).date(),
        key="past_scanner_date",
    )

    with st.spinner(
        f"{selected_scan_date.strftime('%Y-%m-%d')} 기준 유니버스 팩터 연산 및"
        " 추론 중..."
    ):
      try:
        df_clean, last_fetched_kst = fetch_and_process_universe_with_meta(
            tuple(UNIVERSE_TICKERS), DATA_START_DATE
        )
        st.caption(f"🕒 데이터 마지막 다운로드 시각: **{last_fetched_kst}**")
        if not df_clean.empty:
          target_ts = pd.Timestamp(selected_scan_date)
          render_scanner_view(
              df_clean=df_clean,
              target_date_ts=target_ts,
              top_n_percent=TOP_N_PERCENT,
              min_tp_hit_prob=MIN_TP_HIT_PROB,
              max_atr_pct=MAX_ATR_PCT,
              config=MODEL_CONFIG,
              feature_cols=FEATURE_COLS,
              date_label_prefix="선택 과거 기준일",
              last_fetched_kst=last_fetched_kst,
              requested_date_str=selected_scan_date.strftime("%Y-%m-%d"),
          )
        else:
          st.error("데이터 전처리 결과가 비어 있습니다.")
      except Exception as e:
        st.error(f"과거 날짜 스캔 중 오류 발생: {e}")
        st.code(traceback.format_exc(), language="python")

  # -------------------------------------------------------------
  # 2. 특정 종목 당일 분석
  # -------------------------------------------------------------
  elif menu == "🔍 특정 종목 당일 분석":
    st.header(f"🔍 특정 종목 당일 딥다이브 분석 ({HOLDING_PERIOD}거래일 기준)")
    target_ticker_raw = st.text_input(
        "분석할 티커 입력 (대소문자 무관, 유니버스 외 종목도 지원)", "NVDA"
    )
    target_ticker = target_ticker_raw.strip().upper()

    if st.button("분석 시작"):
      if not target_ticker:
        st.warning("티커를 입력해주세요.")
      else:
        with st.spinner(f"{target_ticker} 분석 중..."):
          try:
            t_df, raw_norm_df, in_universe = fetch_single_ticker_with_market(
                target_ticker
            )

            if t_df.empty or len(t_df) < 30:
              st.error(
                  f"'{target_ticker}'에 대한 충분한 데이터가 없습니다. 상장한 지"
                  " 얼마 되지 않아 누적 거래일 수가 적거나(최소 100거래일"
                  " 필요) 티커 오타일 수 있습니다."
              )
            else:
              source_label = (
                  "유니버스 동기화 데이터"
                  if in_universe
                  else "yfinance 개별 다운로드 + VIX/SPY 결합 데이터"
              )
              latest_scored = score_candidates(
                  t_df.iloc[[-1]], MODEL_CONFIG, FEATURE_COLS
              )
              latest = latest_scored.iloc[0]

              adj_score = float(latest["Adjusted_Pred_Score"])
              boost_val = float(latest["Theme_Boost"])
              theme_val = str(latest.get("Theme", "-"))
              tp_prob_pct = float(latest["TP_Hit_Prob"]) * 100.0
              atr_pct_val = float(latest["ATR_Pct"]) * 100.0
              pred_tp = float(latest["Predicted_TP"])
              close_p = float(latest["Close"])

              sl_price, sl_pct, tp_price, tp_pct = compute_trade_levels(
                  close_p, float(latest["ATR_Pct"]), pred_tp, MODEL_CONFIG
              )

              latest_bar_ts = pd.Timestamp(latest["Date"])
              latest_bar_date = latest_bar_ts.strftime("%Y-%m-%d")
              proj_expiry_ts = latest_bar_ts + pd.Timedelta(
                  days=int(HOLDING_PERIOD * 1.45)
              )

              prob_passed = float(latest["TP_Hit_Prob"]) >= MIN_TP_HIT_PROB
              atr_passed = atr_pct_val < MAX_ATR_PCT
              rank_status_text = "비교군 없음"
              rank_status_class = "status-warn"
              in_top_pct = False

              df_univ = fetch_and_process_universe_data()
              if not df_univ.empty:
                day_sub = df_univ[df_univ["Date"] == latest_bar_ts].copy()
                if not day_sub.empty:
                  day_scored = score_candidates(
                      day_sub, MODEL_CONFIG, FEATURE_COLS
                  ).sort_values(
                      by="Adjusted_Pred_Score", ascending=False
                  ).reset_index(drop=True)
                  top_k = max(1, int(len(day_scored) * (TOP_N_PERCENT / 100.0)))
                  top_tickers = set(
                      day_scored.head(top_k)["Ticker"].str.upper()
                  )

                  matched_pos = day_scored.index[
                      day_scored["Ticker"].str.upper() == target_ticker
                  ].tolist()
                  if matched_pos:
                    cur_rank = matched_pos[0] + 1
                    total_cnt = len(day_scored)
                    if target_ticker in top_tickers:
                      in_top_pct = True
                      rank_status_text = (
                          f"상위 {TOP_N_PERCENT:.0f}% 충족 ✅"
                          f" ({cur_rank}/{total_cnt}위)"
                      )
                      rank_status_class = "status-pass"
                    else:
                      rank_status_text = (
                          f"상위 {TOP_N_PERCENT:.0f}% 미달 ❌"
                          f" ({cur_rank}/{total_cnt}위)"
                      )
                      rank_status_class = "status-fail"
                  else:
                    cutoff_score = float(
                        day_scored.iloc[top_k - 1]["Adjusted_Pred_Score"]
                    )
                    if adj_score >= cutoff_score:
                      in_top_pct = True
                      rank_status_text = (
                          f"상위 {TOP_N_PERCENT:.0f}% 선 충족 ✅ (유니버스 외)"
                      )
                      rank_status_class = "status-pass"
                    else:
                      rank_status_text = (
                          f"상위 {TOP_N_PERCENT:.0f}% 미달 ❌ (유니버스 외)"
                      )
                      rank_status_class = "status-fail"

              prob_status_text = (
                  f"기준 충족 ✅ (≥ {MIN_TP_HIT_PROB*100:.0f}%)"
                  if prob_passed
                  else f"기준 미달 ❌ (< {MIN_TP_HIT_PROB*100:.0f}%)"
              )
              prob_status_class = "status-pass" if prob_passed else "status-fail"

              atr_status_text = (
                  f"기준 충족 ✅ (< {MAX_ATR_PCT:.1f}%)"
                  if atr_passed
                  else f"변동성 과다 ❌ (≥ {MAX_ATR_PCT:.1f}%)"
              )
              atr_status_class = "status-pass" if atr_passed else "status-fail"

              rank_headline_text = (
                  f"상위 {TOP_N_PERCENT:.0f}% 충족 ✅"
                  if in_top_pct
                  else f"상위 {TOP_N_PERCENT:.0f}% 미달 ❌"
              )

              buy_recommended = prob_passed and in_top_pct and atr_passed

              st.subheader(f"📊 {target_ticker} [{theme_val}] 분석 리포트")
              st.caption(
                  f"데이터 기준일: {latest_bar_date} | 분석 시각:"
                  f" {now_kst_str()} | 소스: {source_label}"
              )

              if buy_recommended:
                st.success(
                    f"✅ **매수 권장:** [{target_ticker}] 익절선달성률"
                    f" ({tp_prob_pct:.1f}% ≥ {MIN_TP_HIT_PROB*100:.0f}%),"
                    f" 랭킹 상위 {TOP_N_PERCENT:.0f}% 및 ATR"
                    f" ({atr_pct_val:.2f}% < {MAX_ATR_PCT:.1f}%) 기준을"
                    " 충족합니다."
                )
              else:
                failed_reasons = []
                if not prob_passed:
                  failed_reasons.append(
                      f"익절선달성률 {tp_prob_pct:.1f}% (<"
                      f" {MIN_TP_HIT_PROB*100:.0f}%)"
                  )
                if not in_top_pct:
                  failed_reasons.append(
                      f"상위 {TOP_N_PERCENT:.0f}% 랭킹 미달"
                  )
                if not atr_passed:
                  failed_reasons.append(
                      f"ATR {atr_pct_val:.2f}% (≥ {MAX_ATR_PCT:.1f}%)"
                  )
                reason_str = ", ".join(failed_reasons)
                st.warning(
                    f"⛔ **매수 비권장 (관망 권장):** [{target_ticker}] 진입"
                    f" 기준 미달 ({reason_str})"
                )

              st.markdown(
                  f"**현재 종가 (진입가):** <span"
                  f' style="font-size: 28px; font-weight: bold; color:'
                  f' #00FF7F;">${close_p:.2f}</span>',
                  unsafe_allow_html=True,
              )

              st.markdown(METRIC_CSS, unsafe_allow_html=True)
              rec_label = "매수 권장 ✅" if buy_recommended else "관망 권장 ⛔"
              rec_class = "status-pass" if buy_recommended else "status-fail"

              filter_cards_html = (
                  '<div class="metric-container">'
                  '<div class="metric-card">'
                  '<div class="metric-title">목표 익절선달성률</div>'
                  f'<div class="metric-value">{tp_prob_pct:.1f}%</div>'
                  f'<div class="metric-sub {prob_status_class}">{prob_status_text}</div>'
                  "</div>"
                  '<div class="metric-card">'
                  '<div class="metric-title">ATR (14일 변동성)</div>'
                  f'<div class="metric-value">{atr_pct_val:.2f}%</div>'
                  f'<div class="metric-sub {atr_status_class}">{atr_status_text}</div>'
                  "</div>"
                  '<div class="metric-card">'
                  '<div class="metric-title">조정 랭킹 점수</div>'
                  f'<div class="metric-value">{adj_score:.4f}</div>'
                  f'<div class="metric-sub" style="color:#38BDF8;">테마가산점 {boost_val:+.3f}</div>'
                  "</div>"
                  '<div class="metric-card">'
                  f'<div class="metric-title">상위 {TOP_N_PERCENT:.0f}% 랭킹 여부</div>'
                  f'<div class="metric-value {rank_status_class}" style="font-size:17px;">{rank_headline_text}</div>'
                  f'<div class="metric-sub" style="color:#AAAAAA;">{rank_status_text} · 테마 {theme_val}</div>'
                  "</div>"
                  '<div class="metric-card">'
                  '<div class="metric-title">매수 권장 여부</div>'
                  f'<div class="metric-value {rec_class}">{rec_label}</div>'
                  f'<div class="metric-sub" style="color:#AAAAAA;">Top {TOP_N_PERCENT:.0f}% · 확률≥{MIN_TP_HIT_PROB*100:.0f}% · ATR&lt;{MAX_ATR_PCT:.1f}%</div>'
                  "</div>"
                  "</div>"
              )
              st.markdown(filter_cards_html, unsafe_allow_html=True)

              metric_html = (
                  '<div class="metric-container">'
                  '<div class="metric-card">'
                  '<div class="metric-title">권장 손절가</div>'
                  f'<div class="metric-value">${sl_price:.2f} <span'
                  f' style="font-size:14px; color:#FF6B6B;">({sl_pct:.2f}%)</span></div>'
                  "</div>"
                  '<div class="metric-card">'
                  '<div class="metric-title">예상 익절가</div>'
                  f'<div class="metric-value">${tp_price:.2f} <span'
                  f' style="font-size:14px; color:#4ECDC4;">(+{tp_pct:.2f}%)</span></div>'
                  "</div>"
                  '<div class="metric-card">'
                  '<div class="metric-title">예상 목표익절폭</div>'
                  f'<div class="metric-value" style="color: #4ECDC4;">+{pred_tp:.2f}%</div>'
                  "</div>"
                  "</div>"
              )
              st.markdown(metric_html, unsafe_allow_html=True)

              chart_src = (
                  raw_norm_df
                  if raw_norm_df is not None and not raw_norm_df.empty
                  else t_df
              )
              chart_window = chart_src.tail(35).copy()
              render_fixed_candlestick(
                  ohlc_df=chart_window,
                  title=(
                      f"{target_ticker} 캔들차트 (기준/진입 시점 및 손절·익절"
                      " 구간)"
                  ),
                  sl_price=sl_price,
                  tp_price=tp_price,
                  entry_price=close_p,
                  buy_ts=latest_bar_ts,
                  expiry_ts=proj_expiry_ts,
              )
          except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")
            st.code(traceback.format_exc(), language="python")

  # -------------------------------------------------------------
  # 3. 특정 종목 과거 시점 분석
  # -------------------------------------------------------------
  elif menu == "⏳ 특정 종목 과거 시점 분석":
    st.header(
        f"⏳ 특정 종목 과거 특정 날짜 기준 시그널 검증 ({HOLDING_PERIOD}거래일"
        " 사후 검증)"
    )
    target_ticker_raw = st.text_input(
        "분석할 티커 입력 (대소문자 무관, 유니버스 외 종목도 지원)",
        "NVDA",
        key="past_ticker",
    )
    target_ticker = target_ticker_raw.strip().upper()
    default_past_date = (datetime.now(KST) - timedelta(days=30)).date()
    target_date = st.date_input("조회할 과거 날짜 선택", default_past_date)

    if st.button("과거 시점 시그널 검증 실행"):
      if not target_ticker:
        st.warning("티커를 입력해주세요.")
      else:
        with st.spinner(f"{target_ticker} 과거 데이터 조회 및 사후 검증 중..."):
          try:
            df_t, full_t_df, in_universe = fetch_single_ticker_with_market(
                target_ticker
            )

            if df_t.empty:
              st.error(
                  f"'{target_ticker}'에 대한 충분한 데이터를 가져오지"
                  " 못했습니다. 상장한 지 얼마 되지 않아 누적 거래일 수가"
                  " 적거나(최소 100거래일 필요) 티커 오타일 수 있습니다."
              )
            else:
              source_label = (
                  "유니버스 동기화 데이터"
                  if in_universe
                  else "yfinance 개별 다운로드 + VIX/SPY 결합 데이터"
              )
              df_t = df_t.copy()
              df_t["DateOnly"] = pd.to_datetime(df_t["Date"]).dt.date
              match_row = df_t[df_t["DateOnly"] == target_date]

              if not match_row.empty:
                scored_row = score_candidates(
                    match_row.iloc[[-1]], MODEL_CONFIG, FEATURE_COLS
                )
                row = scored_row.iloc[0]

                adj_score = float(row["Adjusted_Pred_Score"])
                boost_val = float(row["Theme_Boost"])
                theme_val = str(row.get("Theme", "-"))
                tp_prob_pct = float(row["TP_Hit_Prob"]) * 100.0
                atr_pct_val = float(row["ATR_Pct"]) * 100.0
                pred_tp = float(row["Predicted_TP"])
                close_p = float(row["Close"])

                sl_price, sl_pct, tp_price, tp_pct = compute_trade_levels(
                    close_p, float(row["ATR_Pct"]), pred_tp, MODEL_CONFIG
                )

                prob_passed = float(row["TP_Hit_Prob"]) >= MIN_TP_HIT_PROB
                atr_passed = atr_pct_val < MAX_ATR_PCT

                if full_t_df is None or full_t_df.empty:
                  full_t_df = df_t.copy()
                else:
                  full_t_df = full_t_df.copy()

                full_t_df["Date"] = pd.to_datetime(full_t_df["Date"])
                full_t_df["DateOnly"] = full_t_df["Date"].dt.date

                future_sub = full_t_df[
                    full_t_df["DateOnly"] >= target_date
                ].head(HOLDING_PERIOD + 1)

                buy_ts = pd.Timestamp(target_date)
                expiry_ts = None

                if len(future_sub) > 1:
                  valid_period = future_sub.iloc[1:]
                  expiry_ts = pd.Timestamp(valid_period["Date"].iloc[-1])

                  def get_scalar(val):
                    if isinstance(val, pd.Series):
                      return float(val.iloc[0])
                    return float(val)

                  high_n = get_scalar(valid_period["High"].max())
                  low_n = get_scalar(valid_period["Low"].min())
                  close_n = get_scalar(valid_period["Close"].iloc[-1])
                  ret_n = (close_n - close_p) / close_p * 100

                  tp_reached = high_n >= tp_price
                  sl_touched = low_n <= sl_price

                  tp_str = (
                      f"성공 🎯 ({high_n:.2f} 도달)" if tp_reached else "미달 ❌"
                  )
                  sl_str = (
                      "위험 ⚠️ (터치 발생)"
                      if sl_touched
                      else "안전 ✅ (터치 없음)"
                  )
                else:
                  high_n, low_n, close_n, ret_n = (
                      close_p,
                      close_p,
                      close_p,
                      0.0,
                  )
                  tp_str, sl_str = "데이터 부족", "데이터 부족"

                st.subheader(
                    f"📊 {target_ticker} [{theme_val}] - {target_date} 기준"
                    f" {HOLDING_PERIOD}일 검증 리포트"
                )
                st.caption(
                    f"검증 실행 시각: {now_kst_str()} | 소스: {source_label}"
                )

                if prob_passed and atr_passed:
                  st.success(
                      f"✨ **[{target_ticker}] 익절선달성률 기준"
                      f" ({tp_prob_pct:.1f}% ≥ {MIN_TP_HIT_PROB*100:.0f}%)"
                      f" 및 ATR ({atr_pct_val:.2f}% < {MAX_ATR_PCT:.1f}%)"
                      " 충족!**"
                  )
                else:
                  st.warning(
                      f"⚠️ **[{target_ticker}] 시그널 기준 미확인/미달"
                      f" (익절선달성률 {tp_prob_pct:.1f}%, ATR"
                      f" {atr_pct_val:.2f}%)**"
                  )

                st.markdown(METRIC_CSS, unsafe_allow_html=True)
                atr_sub_color = "#00FF7F" if atr_passed else "#FF6B6B"
                atr_sub_text = (
                    f"기준 < {MAX_ATR_PCT:.1f}%"
                    if atr_passed
                    else f"기준 ≥ {MAX_ATR_PCT:.1f}% (과다)"
                )
                card_html_1 = (
                    '<div class="metric-container">'
                    '<div class="metric-card">'
                    '<div class="metric-title">진입가 (기준 종가)</div>'
                    f'<div class="metric-value">${close_p:.2f}</div>'
                    "</div>"
                    '<div class="metric-card">'
                    '<div class="metric-title">조정 랭킹 점수</div>'
                    f'<div class="metric-value">{adj_score:.4f}</div>'
                    f'<div class="metric-sub" style="color:#38BDF8;">테마가산점 {boost_val:+.3f}</div>'
                    "</div>"
                    '<div class="metric-card">'
                    '<div class="metric-title">목표 익절선달성률</div>'
                    f'<div class="metric-value">{tp_prob_pct:.1f}%</div>'
                    f'<div class="metric-sub" style="color:#AAAAAA;">기준 {MIN_TP_HIT_PROB*100:.0f}% 이상</div>'
                    "</div>"
                    '<div class="metric-card">'
                    '<div class="metric-title">ATR (14일 변동성)</div>'
                    f'<div class="metric-value">{atr_pct_val:.2f}%</div>'
                    f'<div class="metric-sub" style="color:{atr_sub_color};">{atr_sub_text}</div>'
                    "</div>"
                    '<div class="metric-card">'
                    '<div class="metric-title">예상 목표익절폭</div>'
                    f'<div class="metric-value" style="color:#4ECDC4;">+{pred_tp:.2f}%</div>'
                    "</div>"
                    "</div>"
                )
                st.markdown(card_html_1, unsafe_allow_html=True)

                ret_n_color = "#00FF7F" if ret_n >= 0 else "#FF6B6B"
                card_html_2 = (
                    '<div class="metric-container">'
                    '<div class="metric-card">'
                    '<div class="metric-title">산출 손절가</div>'
                    f'<div class="metric-value" style="color:#FF6B6B;">${sl_price:.2f}'
                    f' <span style="font-size:14px;">({sl_pct:.2f}%)</span></div>'
                    "</div>"
                    '<div class="metric-card">'
                    '<div class="metric-title">산출 익절가</div>'
                    f'<div class="metric-value" style="color:#4ECDC4;">${tp_price:.2f}'
                    f' <span style="font-size:14px;">(+{tp_pct:.2f}%)</span></div>'
                    "</div>"
                    '<div class="metric-card">'
                    f'<div class="metric-title">{HOLDING_PERIOD}일 최고가 / 익절달성</div>'
                    f'<div class="metric-value" style="font-size:17px;">${high_n:.2f}'
                    f' <br><span style="font-size:13px; color:#4ECDC4;">{tp_str}</span></div>'
                    "</div>"
                    '<div class="metric-card">'
                    f'<div class="metric-title">{HOLDING_PERIOD}일 만기 수익률</div>'
                    f'<div class="metric-value" style="color:{ret_n_color};">{ret_n:+.2f}%</div>'
                    "</div>"
                    "</div>"
                )
                st.markdown(card_html_2, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                  st.info(
                      f"🛡️ **손절 터치 여부:** {sl_str} ({HOLDING_PERIOD}일"
                      f" 최저가: ${low_n:.2f})"
                  )
                with col2:
                  st.success(
                      f"🎯 **목표 익절 도달:** {tp_str} (목표가:"
                      f" ${tp_price:.2f})"
                  )

                st.subheader(
                    f"📈 매수 시점 ~ {HOLDING_PERIOD}거래일 보유 만기 캔들차트"
                )
                pre_sub = full_t_df[
                    full_t_df["DateOnly"] < target_date
                ].tail(10)
                post_buffer = full_t_df[
                    full_t_df["DateOnly"]
                    > (
                        expiry_ts.date()
                        if expiry_ts is not None
                        else target_date
                    )
                ].head(5)
                chart_df = pd.concat(
                    [pre_sub, future_sub, post_buffer], ignore_index=True
                ).drop_duplicates(subset=["Date"])

                render_fixed_candlestick(
                    ohlc_df=chart_df,
                    title=(
                        f"{target_ticker} ({target_date} 기준 {HOLDING_PERIOD}일"
                        " 시그널 검증 캔들차트)"
                    ),
                    sl_price=sl_price,
                    tp_price=tp_price,
                    entry_price=close_p,
                    buy_ts=buy_ts,
                    expiry_ts=expiry_ts,
                )

              else:
                st.warning(
                    "선택한 날짜에 해당하는 거래 데이터가 없습니다 (휴장일"
                    " 등)."
                )
          except Exception as e:
            st.error(f"과거 데이터 조회 중 오류 발생: {e}")
            st.code(traceback.format_exc(), language="python")

  # -------------------------------------------------------------
  # 4. 기존 보유 포지션 진단 및 매매 가이드
  # -------------------------------------------------------------
  elif menu == "🛡️ 기존 보유 포지션 진단 및 매매 가이드":
    st.header(
        f"🛡️ 기존 보유 포지션 진단 및 매매 가이드 ({HOLDING_PERIOD}거래일 스윙"
        " 기준)"
    )
    col1, col2, col3 = st.columns(3)
    with col1:
      pos_ticker_raw = st.text_input(
          "보유 종목 티커 (대소문자 무관)", "NVDA", key="pos_ticker"
      )
      pos_ticker = pos_ticker_raw.strip().upper()
    with col2:
      default_pos_date = (datetime.now(KST) - timedelta(days=15)).date()
      pos_date = st.date_input(
          "매수 날짜", default_pos_date, key="pos_date"
      )
    with col3:
      pos_price = st.number_input(
          "매수가격 ($)", value=140.0, step=1.0, key="pos_price"
      )

    if st.button("포지션 진단 실행"):
      if not pos_ticker:
        st.warning("티커를 입력해주세요.")
      else:
        with st.spinner(f"{pos_ticker} 실시간 분석 중..."):
          try:
            t_df, raw_norm_df, in_universe = fetch_single_ticker_with_market(
                pos_ticker
            )

            if t_df.empty or len(t_df) < 30:
              st.error(
                  f"'{pos_ticker}' 종목 데이터를 충분히 가져오지 못했습니다."
                  " 상장한 지 얼마 되지 않아 누적 거래일 수가 적어(최소"
                  " 100거래일 필요) 지표 산출 기준을 충족하지 못했거나 티커를"
                  " 찾지 못했을 수 있습니다."
              )
            else:
              source_label = (
                  "유니버스 동기화 데이터"
                  if in_universe
                  else "yfinance 개별 다운로드 + VIX/SPY 결합 데이터"
              )
              latest_scored = score_candidates(
                  t_df.iloc[[-1]], MODEL_CONFIG, FEATURE_COLS
              )
              latest_row = latest_scored.iloc[0]

              curr_price = float(latest_row["Close"])
              holding_return = (curr_price - pos_price) / pos_price * 100.0
              tp_prob_pct = float(latest_row["TP_Hit_Prob"]) * 100.0
              atr_pct_val = float(latest_row["ATR_Pct"]) * 100.0
              pred_tp = float(latest_row["Predicted_TP"])
              theme_val = str(latest_row.get("Theme", "-"))

              sl_price, sl_pct, tp_price, tp_pct = compute_trade_levels(
                  curr_price, float(latest_row["ATR_Pct"]), pred_tp, MODEL_CONFIG
              )
              passed_prob = float(latest_row["TP_Hit_Prob"]) >= MIN_TP_HIT_PROB

              if holding_return <= sl_pct:
                action = "STOP LOSS (손절가 이탈 - 즉시 손절 실행)"
              elif holding_return >= tp_pct:
                action = "TAKE PROFIT (목표 익절가 달성 - 수익 실현 권장)"
              elif float(latest_row["TP_Hit_Prob"]) < 0.20:
                action = "SELL / EXIT (익절선달성률 저하 - 비중 축소 또는 매도 권장)"
              elif passed_prob:
                action = "HOLD & RIDE (목표 익절선달성률 양호 - 보유 유지)"
              else:
                action = "WATCH / CAUTION (중립 관망 - 추세 및 손절라인 모니터링)"

              st.subheader(
                  f"🛡️ [{pos_ticker}] ({theme_val}) 기존 보유 포지션 진단 리포트"
              )
              st.caption(
                  f"진단 시각: {now_kst_str()} | 매수일: {pos_date} | 소스:"
                  f" {source_label}"
              )
              st.info(f"🎯 **[최종 의사결정 제안]: {action}**")

              holding_ret_color = (
                  "#00FF7F" if holding_return >= 0 else "#FF6B6B"
              )
              st.markdown(METRIC_CSS, unsafe_allow_html=True)
              card_html_pos = (
                  '<div class="metric-container">'
                  '<div class="metric-card">'
                  '<div class="metric-title">매수가 / 현재가</div>'
                  f'<div class="metric-value">${pos_price:.2f} → ${curr_price:.2f}</div>'
                  "</div>"
                  '<div class="metric-card">'
                  '<div class="metric-title">미실현 손익</div>'
                  f'<div class="metric-value" style="color:{holding_ret_color};">{holding_return:+.2f}%</div>'
                  "</div>"
                  '<div class="metric-card">'
                  '<div class="metric-title">목표 익절선달성률</div>'
                  f'<div class="metric-value">{tp_prob_pct:.1f}%</div>'
                  f'<div class="metric-sub" style="color:#AAAAAA;">기준 {MIN_TP_HIT_PROB*100:.0f}% 이상</div>'
                  "</div>"
                  '<div class="metric-card">'
                  '<div class="metric-title">ATR (14일 변동성)</div>'
                  f'<div class="metric-value">{atr_pct_val:.2f}%</div>'
                  f'<div class="metric-sub" style="color:#AAAAAA;">권장 상한 &lt; {MAX_ATR_PCT:.1f}%</div>'
                  "</div>"
                  '<div class="metric-card">'
                  '<div class="metric-title">갱신된 추천 손절/익절</div>'
                  f'<div class="metric-value" style="font-size:15px;">SL: ${sl_price:.2f}<br>TP: ${tp_price:.2f}</div>'
                  "</div>"
                  "</div>"
              )
              st.markdown(card_html_pos, unsafe_allow_html=True)

              chart_src = (
                  raw_norm_df
                  if raw_norm_df is not None and not raw_norm_df.empty
                  else t_df
              )
              pos_chart_df = chart_src.tail(40).copy()
              buy_dt_ts = pd.Timestamp(pos_date)
              exp_dt_ts = buy_dt_ts + pd.Timedelta(
                  days=int(HOLDING_PERIOD * 1.45)
              )
              render_fixed_candlestick(
                  ohlc_df=pos_chart_df,
                  title=(
                      f"{pos_ticker} 보유 포지션 캔들차트 (매수일 및 갱신"
                      " 손절·익절 라인)"
                  ),
                  sl_price=sl_price,
                  tp_price=tp_price,
                  entry_price=pos_price,
                  buy_ts=buy_dt_ts,
                  expiry_ts=exp_dt_ts,
              )
          except Exception as e:
            st.error(f"포지션 진단 중 오류 발생: {e}")
            st.code(traceback.format_exc(), language="python")
