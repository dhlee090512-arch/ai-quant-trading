from datetime import datetime, timedelta
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
    "LightGBM 3중 앙상블 모델 기반 실전 분석 대시보드 (코랩 셀 1·2 전처리 파이프라인"
    " 100% 동기화)"
)

# 야후 파이낸스 차단 우회를 위한 세션 설정
session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
        " like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
})


# 1. 깃허브에 저장된 모델, 피처 목록 및 설정 메타데이터 로드 (캐싱 적용)
@st.cache_resource
def load_models_and_metadata():
  try:
    ranker = lgb.Booster(model_file="models/ranker_model.txt")
    classifier = lgb.Booster(model_file="models/classifier_model.txt")
    tp_reg = lgb.Booster(model_file="models/tp_regressor_model.txt")

    with open("models/feature_cols.json", "r", encoding="utf-8") as f:
      feature_cols = json.load(f)

    if os.path.exists("models/model_config.json"):
      with open("models/model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    else:
      raise FileNotFoundError(
          "models/model_config.json 파일을 찾을 수 없습니다. 코랩에서 셀 7을"
          " 실행해 메타데이터를 업로드해주세요."
      )

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
    margin-bottom: 25px;
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
.pick-surv {
    background-color: rgba(0, 255, 127, 0.12);
    color: #00FF7F;
    border: 1px solid rgba(0, 255, 127, 0.35);
    border-radius: 999px;
    padding: 4px 12px;
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

  # 손절가~익절가 범위 하이라이트 영역
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

  # 익절가 / 손절가 가로선
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

  # 매수 시점 세로선
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

  # 보유 만기 시점 세로선
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

  # y축 범위 확보 (손절가/익절가/캔들 고저가 모두 포함)
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
      use_container_width=True,
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


# 모델 로드 실패 시 에러 안내 문구 출력
if load_error is not None:
  st.error(
      "🚨 모델 파일 및 메타데이터 로딩 실패! GitHub 레포지토리의 `models/`"
      f" 폴더 상태를 확인해주세요.\n\n**에러 내용:** {load_error}"
  )
else:
  HOLDING_PERIOD = MODEL_CONFIG["holding_period"]
  UNIVERSE_TICKERS = [t.upper() for t in MODEL_CONFIG["universe_tickers"]]


  # -------------------------------------------------------------
  # 코랩 셀 1 & 셀 2 전체 데이터프레임 전처리 및 피처 연산 파이프라인
  # -------------------------------------------------------------
  @st.cache_data(ttl=3600)
  def fetch_and_process_universe_data():
    download_list = list(set(UNIVERSE_TICKERS + ["^VIX", "SPY"]))
    df_raw_yf = yf.download(
        download_list, start="2018-01-01", progress=False, session=session
    )

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
      df_fresh["Date"] = pd.to_datetime(df_fresh["Date"]).dt.tz_localize(None)
    if "Ticker" in df_fresh.columns:
      df_fresh["Ticker"] = df_fresh["Ticker"].astype(str).str.upper()
    df_fresh = df_fresh.dropna(subset=["Close"])

    market_data, spy_df = build_market_tables(df_fresh)

    df_stocks = df_fresh[
        ~df_fresh["Ticker"].isin(["VIX", "^VIX", "SPY"])
    ].copy()
    processed_dfs = []

    for _, group in df_stocks.groupby("Ticker"):
      g = compute_stock_factors(group)
      if not g.empty:
        processed_dfs.append(g)

    if not processed_dfs:
      return pd.DataFrame()

    df_all = pd.concat(processed_dfs, ignore_index=True)
    df_all = pd.merge(df_all, market_data, on="Date", how="inner")
    df_all = pd.merge(
        df_all, spy_df[["Date", "SPY_Ret_5d"]], on="Date", how="inner"
    )
    df_all["Rel_Strength_5d"] = df_all["Ret_5d"] - df_all["SPY_Ret_5d"]

    df_all["Ret_5d_Zscore"] = df_all.groupby("Date")["Ret_5d"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    for col in FEATURE_COLS:
      if col in df_all.columns:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    df_clean = df_all.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df_clean


  @st.cache_data(ttl=3600)
  def fetch_single_ticker_with_market(ticker_upper: str):
    df_clean = fetch_and_process_universe_data()
    if not df_clean.empty:
      matched = df_clean[df_clean["Ticker"] == ticker_upper].copy()
      if not matched.empty:
        raw_single = yf.download(
            ticker_upper,
            start="2018-01-01",
            progress=False,
            auto_adjust=True,
            session=session,
        )
        raw_norm = normalize_ohlcv_df(raw_single)
        return matched, raw_norm, True

    raw_single = yf.download(
        ticker_upper,
        start="2018-01-01",
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
        start="2018-01-01",
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

    for col in FEATURE_COLS:
      if col in g.columns:
        g[col] = pd.to_numeric(g[col], errors="coerce")

    g = g.dropna(subset=FEATURE_COLS).reset_index(drop=True)
