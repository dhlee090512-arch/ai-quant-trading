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
    return g, raw_norm, False


  # -------------------------------------------------------------
  # 1. 당일 종목 추천 (스캐너)
  # -------------------------------------------------------------
  if menu == "🔥 당일 종목 추천 (스캐너)":
    st.header(
        f"🔥 당일 기준 정예 챔피언 스캐너 ({HOLDING_PERIOD}거래일 스윙 기준)"
    )
    st.write(
        "코랩 셀 1·2의 전처리 파이프라인과 100% 동일한 데이터 및 피처 기준으로"
        " 스캔합니다."
    )

    df_display = None
    scan_timestamp = None
    run_time_kst = now_kst_str()

    with st.spinner(
        "코랩 전처리 파이프라인 가동 중 (전체 벌크 다운로드 및 팩터 연산)..."
    ):
      try:
        df_clean = fetch_and_process_universe_data()

        if not df_clean.empty:
          latest_date = df_clean["Date"].max()
          scan_timestamp = f"{pd.to_datetime(latest_date).strftime('%Y-%m-%d')} (조회 시각: {run_time_kst})"

          MAX_DAILY_ATR_PCT = 0.045
          latest_subset = df_clean[
              (df_clean["Date"] == latest_date)
              & (df_clean["ATR_Pct"] <= MAX_DAILY_ATR_PCT)
          ].copy()

          if (
              len(latest_subset) > 0
              and latest_subset[FEATURE_COLS].notna().all().all()
          ):
            x_matrix = latest_subset[FEATURE_COLS].apply(
                pd.to_numeric, errors="coerce"
            )

            latest_subset["Pred_Score"] = ranker_lgb.predict(x_matrix)
            latest_subset["Survive_Prob"] = classifier_lgb.predict(x_matrix)
            latest_subset["Predicted_TP"] = tp_regressor.predict(x_matrix)

            group_sorted = latest_subset.sort_values(
                by="Pred_Score", ascending=False
            )
            top_k = max(1, int(len(group_sorted) * 0.10))
            top_candidates = group_sorted.head(top_k)

            MIN_SURVIVE_PROB_COLAB = 0.58
            filtered_latest = top_candidates[
                top_candidates["Survive_Prob"] >= MIN_SURVIVE_PROB_COLAB
            ].reset_index(drop=True)

            if len(filtered_latest) > 0:
              MAX_SL_LIMIT = -8.0
              display_results = []

              for idx, row in filtered_latest.iterrows():
                rank_num = idx + 1
                entry_price = float(row["Close"])
                atr_val = float(row["ATR_Pct"]) * 100

                sl_pct = max(-(atr_val * 1.5), MAX_SL_LIMIT)
                model_pred_tp = float(row["Predicted_TP"])
                min_required_tp = abs(sl_pct) * 1.5
                tp_pct = max(model_pred_tp, min_required_tp)

                sl_price = entry_price * (1 + sl_pct / 100.0)
                tp_price = entry_price * (1 + tp_pct / 100.0)
                surv_prob_val = float(row["Survive_Prob"] * 100)

                display_results.append({
                    "순위": rank_num,
                    "Ticker": row["Ticker"],
                    "진입가($)": round(entry_price, 2),
                    "손절가($)": round(sl_price, 2),
                    "조정 손절폭(%)": f"{sl_pct:.2f} %",
                    "동적익절가($)": round(tp_price, 2),
                    "동적 익절폭(%)": f"+{tp_pct:.2f} %",
                    "생존확률": f"{surv_prob_val:.1f} %",
                })

              df_display = pd.DataFrame(display_results)
              st.success(
                  "✨ 코랩 전처리 로직과 완벽히 일치하는 정예 챔피언 선별"
                  " 완료!"
              )
            else:
              st.warning(
                  "⚠️ 조건(상위 10% 랭킹, 생존율 58% 이상)을 통과한 종목이"
                  " 없습니다."
              )
              df_display = pd.DataFrame()
          else:
            st.warning(
                "⚠️ 최신 마감일에 조건을 충족하는 유효한 데이터가 없습니다."
            )
            df_display = pd.DataFrame()
        else:
          st.error("데이터 전처리 결과가 비어 있습니다.")
          df_display = pd.DataFrame()
      except Exception as e:
        st.error(f"스캔 중 오류 발생: {e}")
        st.code(traceback.format_exc(), language="python")
        df_display = pd.DataFrame()

    if df_display is not None and not df_display.empty:
      if scan_timestamp:
        st.markdown(f"📅 **타겟 기준일 (최신 마감일):** `{scan_timestamp}`")

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
                f'<div class="pick-rank-ticker">#{r["순위"]} {r["Ticker"]}</div>'
                f'<div class="pick-surv">생존확률 {r["생존확률"]}</div>'
                "</div>"
                '<div class="pick-grid">'
                '<div class="pick-item">'
                '<div class="pick-item-label">진입가($)</div>'
                f'<div class="pick-item-val">${r["진입가($)"]:.2f}</div>'
                "</div>"
                '<div class="pick-item">'
                '<div class="pick-item-label">손절가 / 조정 손절폭</div>'
                f'<div class="pick-item-val">${r["손절가($)"]:.2f}<span class="pick-sl">({r["조정 손절폭(%)"].strip()})</span></div>'
                "</div>"
                '<div class="pick-item">'
                '<div class="pick-item-label">동적익절가 / 동적 익절폭</div>'
                f'<div class="pick-item-val">${r["동적익절가($)"]:.2f}<span class="pick-tp">({r["동적 익절폭(%)"].strip()})</span></div>'
                "</div>"
                "</div>"
                "</div>"
            )
            st.markdown(box_html, unsafe_allow_html=True)

      st.markdown("#### 📋 전체 추천 종목 요약 표")
      st.dataframe(df_display, use_container_width=True, hide_index=True)

  # -------------------------------------------------------------
  # 2. 특정 종목 당일 분석
  # -------------------------------------------------------------
  elif menu == "🔍 특정 종목 당일 분석":
    st.header(f"🔍 특정 종목 당일 딥다이브 분석 ({HOLDING_PERIOD}거래일 기준)")
    target_ticker_raw = st.text_input(
        "분석할 티커 입력 (대소문자 무관, 유니버스 외 종목도 지원)", "TSM"
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
                  f"'{target_ticker}'에 대한 충분한 데이터가 없습니다. 티커"
                  " 오타를 확인해주세요."
              )
            else:
              source_label = (
                  "유니버스 동기화 데이터"
                  if in_universe
                  else "yfinance 개별 다운로드 + VIX/SPY 결합 데이터"
              )
              latest = t_df.iloc[[-1]]
              x_val = latest[FEATURE_COLS].apply(
                  pd.to_numeric, errors="coerce"
              )

              score = float(ranker_lgb.predict(x_val)[0])
              surv_prob = float(classifier_lgb.predict(x_val)[0] * 100)
              pred_tp = float(tp_regressor.predict(x_val)[0])
              close_p = float(latest["Close"].values[0])
              atr_val = float(latest["ATR_Pct"].values[0]) * 100

              sl_pct = max(-(atr_val * 1.5), -8.0)
              min_required_tp = abs(sl_pct) * 1.5
              tp_pct = max(pred_tp, min_required_tp)

              sl_price = close_p * (1 + sl_pct / 100.0)
              tp_price = close_p * (1 + tp_pct / 100.0)

              latest_bar_ts = pd.Timestamp(latest["Date"].values[0])
              latest_bar_date = latest_bar_ts.strftime("%Y-%m-%d")
              proj_expiry_ts = latest_bar_ts + pd.Timedelta(
                  days=int(HOLDING_PERIOD * 1.45)
              )

              # 기준 충족 여부 계산
              surv_passed = surv_prob >= 58.0
              atr_passed = atr_val <= 4.5

              # 기준일 유니버스(ATR <= 4.5%) 내 랭킹 점수 상위 10% 여부 확인
              rank_status_text = "비교군 없음"
              rank_status_class = "status-warn"
              rank_passed = False

              df_univ = fetch_and_process_universe_data()
              if not df_univ.empty:
                day_sub = df_univ[
                    (df_univ["Date"] == latest_bar_ts)
                    & (df_univ["ATR_Pct"] <= 0.045)
                ].copy()
                if not day_sub.empty:
                  x_day = day_sub[FEATURE_COLS].apply(
                      pd.to_numeric, errors="coerce"
                  )
                  day_sub["Pred_Score"] = ranker_lgb.predict(x_day)
                  day_sorted = day_sub.sort_values(
                      by="Pred_Score", ascending=False
                  ).reset_index(drop=True)
                  top_k = max(1, int(len(day_sorted) * 0.10))
                  top_tickers = set(day_sorted.head(top_k)["Ticker"].str.upper())

                  matched_pos = day_sorted.index[
                      day_sorted["Ticker"].str.upper() == target_ticker
                  ].tolist()
                  if matched_pos:
                    cur_rank = matched_pos[0] + 1
                    total_cnt = len(day_sorted)
                    if target_ticker in top_tickers:
                      rank_passed = True
                      rank_status_text = (
                          f"상위 10% 충족 ✅ ({cur_rank}/{total_cnt}위)"
                      )
                      rank_status_class = "status-pass"
                    else:
                      rank_status_text = (
                          f"상위 10% 미달 ❌ ({cur_rank}/{total_cnt}위)"
                      )
                      rank_status_class = "status-fail"
                  else:
                    cutoff_score = float(
                        day_sorted.iloc[top_k - 1]["Pred_Score"]
                    )
                    if score >= cutoff_score:
                      rank_passed = True
                      rank_status_text = "상위 10% 선 충족 ✅ (유니버스 외)"
                      rank_status_class = "status-pass"
                    else:
                      rank_status_text = "상위 10% 미달 ❌ (유니버스 외)"
                      rank_status_class = "status-fail"

              surv_status_text = (
                  "기준 충족 ✅ (≥ 58.0%)"
                  if surv_passed
                  else "기준 미달 ❌ (< 58.0%)"
              )
              surv_status_class = "status-pass" if surv_passed else "status-fail"

              atr_status_text = (
                  "기준 충족 ✅ (≤ 4.50%)"
                  if atr_passed
                  else "기준 미달 ❌ (> 4.50%)"
              )
              atr_status_class = "status-pass" if atr_passed else "status-fail"

              buy_recommended = surv_passed and atr_passed

              st.subheader(f"📊 {target_ticker} 분석 리포트")
              st.caption(
                  f"데이터 기준일: {latest_bar_date} | 분석 시각:"
                  f" {now_kst_str()} | 소스: {source_label}"
              )

              if buy_recommended:
                st.success(
                    f"✅ **매수 권장:** [{target_ticker}] 생존 확률"
                    f" ({surv_prob:.1f}% ≥ 58.0%) 및 일일 ATR 변동성"
                    f" ({atr_val:.2f}% ≤ 4.50%) 기준을 충족합니다."
                )
              else:
                failed_reasons = []
                if not surv_passed:
                  failed_reasons.append(f"생존 확률 {surv_prob:.1f}% (< 58.0%)")
                if not atr_passed:
                  failed_reasons.append(f"일일 ATR {atr_val:.2f}% (> 4.50%)")
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

              # 1행: 핵심 필터 지표 (생존 확률 / 일일 ATR 변동성 / 종목 점수 / 매수 권장 여부)
              rec_label = "매수 권장 ✅" if buy_recommended else "관망 권장 ⛔"
              rec_class = "status-pass" if buy_recommended else "status-fail"

              filter_cards_html = (
                  '<div class="metric-container">'
                  '<div class="metric-card">'
                  '<div class="metric-title">손절 생존 확률</div>'
                  f'<div class="metric-value">{surv_prob:.1f}%</div>'
                  f'<div class="metric-sub {surv_status_class}">{surv_status_text}</div>'
                  "</div>"
                  '<div class="metric-card">'
                  '<div class="metric-title">일일 ATR 변동성</div>'
                  f'<div class="metric-value">{atr_val:.2f}%</div>'
                  f'<div class="metric-sub {atr_status_class}">{atr_status_text}</div>'
                  "</div>"
                  '<div class="metric-card">'
                  '<div class="metric-title">종목 점수 (랭킹)</div>'
                  f'<div class="metric-value">{score:.4f}</div>'
                  f'<div class="metric-sub {rank_status_class}">{rank_status_text}</div>'
                  "</div>"
                  '<div class="metric-card">'
                  '<div class="metric-title">매수 권장 여부</div>'
                  f'<div class="metric-value {rec_class}">{rec_label}</div>'
                  f'<div class="metric-sub" style="color:#AAAAAA;">생존≥58% &amp; ATR≤4.5%</div>'
                  "</div>"
                  "</div>"
              )
              st.markdown(filter_cards_html, unsafe_allow_html=True)

              # 2행: 손절가 / 익절가 / 목표익절폭
              metric_html = (
                  '<div class="metric-container">'
                  '<div class="metric-card">'
                  '<div class="metric-title">권장 손절가</div>'
                  f'<div class="metric-value">${sl_price:.2f} <span'
                  f' style="font-size:14px; color:#FF6B6B;">({sl_pct:.2f}%)</span></div>'
                  "</div>"
                  '<div class="metric-card">'
                  '<div class="metric-title">예상 동적익절가</div>'
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
        "TSM",
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
                  " 못했습니다."
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
                row = match_row.iloc[0]
                x_val = (
                    row[FEATURE_COLS]
                    .to_frame()
                    .T.apply(pd.to_numeric, errors="coerce")
                )

                score = float(ranker_lgb.predict(x_val)[0])
                surv_prob = float(classifier_lgb.predict(x_val)[0] * 100)
                pred_tp = float(tp_regressor.predict(x_val)[0])
                close_p = float(row["Close"])

                atr_val = float(row["ATR_Pct"]) * 100
                atr_limit_passed = atr_val <= 4.5

                sl_pct = max(-(atr_val * 1.5), -8.0)
                min_required_tp = abs(sl_pct) * 1.5
                tp_pct = max(pred_tp, min_required_tp)

                sl_price = close_p * (1 + sl_pct / 100.0)
                tp_price = close_p * (1 + tp_pct / 100.0)

                surv_passed = surv_prob >= 58.0
                is_champion = surv_passed and atr_limit_passed

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
                      f"위험 ⚠️ (터치 발생)"
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
                    f"📊 {target_ticker} - {target_date} 기준"
                    f" {HOLDING_PERIOD}일 검증 리포트"
                )
                st.caption(
                    f"검증 실행 시각: {now_kst_str()} | 소스: {source_label}"
                )

                if is_champion:
                  st.success(
                      f"✨ **[{target_ticker}] 매수 진입 적격 챔피언 시그널"
                      " 발생!** (손절 생존 확률 >= 58% 및 변동성 기준 충족)"
                  )
                else:
                  st.warning(f"⚠️ **[{target_ticker}] 조건 미달 (관망 권장)**")

                st.markdown(METRIC_CSS, unsafe_allow_html=True)
                card_html_1 = (
                    '<div class="metric-container">'
                    '<div class="metric-card">'
                    '<div class="metric-title">진입가 (기준 종가)</div>'
                    f'<div class="metric-value">${close_p:.2f}</div>'
                    "</div>"
                    '<div class="metric-card">'
                    '<div class="metric-title">모델 랭킹 점수</div>'
                    f'<div class="metric-value">{score:.4f}</div>'
                    "</div>"
                    '<div class="metric-card">'
                    '<div class="metric-title">손절 생존 확률</div>'
                    f'<div class="metric-value">{surv_prob:.1f}%</div>'
                    "</div>"
                    '<div class="metric-card">'
                    '<div class="metric-title">일일 ATR 변동성</div>'
                    f'<div class="metric-value">{atr_val:.2f}%</div>'
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
          "보유 종목 티커 (대소문자 무관)", "AAPL", key="pos_ticker"
      )
      pos_ticker = pos_ticker_raw.strip().upper()
    with col2:
      default_pos_date = (datetime.now(KST) - timedelta(days=15)).date()
      pos_date = st.date_input(
          "매수 날짜", default_pos_date, key="pos_date"
      )
    with col3:
      pos_price = st.number_input(
          "매수가격 ($)", value=300.0, step=1.0, key="pos_price"
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
              )
            else:
              source_label = (
                  "유니버스 동기화 데이터"
                  if in_universe
                  else "yfinance 개별 다운로드 + VIX/SPY 결합 데이터"
              )
              latest_row = t_df.iloc[[-1]]
              curr_price = float(latest_row["Close"].values[0])
              holding_return = (curr_price - pos_price) / pos_price * 100

              x_val = latest_row[FEATURE_COLS].apply(
                  pd.to_numeric, errors="coerce"
              )
              score = float(ranker_lgb.predict(x_val)[0])
              surv_prob = float(classifier_lgb.predict(x_val)[0] * 100)
              pred_tp = float(tp_regressor.predict(x_val)[0])
              atr_val = float(latest_row["ATR_Pct"].values[0]) * 100

              is_valid_atr = atr_val <= 4.5
              sl_pct = max(-(atr_val * 1.5), -8.0)
              min_required_tp = abs(sl_pct) * 1.5
              tp_pct = max(pred_tp, min_required_tp)

              sl_price = curr_price * (1 + sl_pct / 100.0)
              tp_price = curr_price * (1 + tp_pct / 100.0)
              passed_survive = surv_prob >= 58.0

              if not is_valid_atr:
                action = (
                    "SELL / REDUCE (변동성 위험 - 비중 축소 또는 매도 추천)"
                )
              elif surv_prob < 50.0:
                action = "SELL / EXIT (하방 리스크 고조 - 즉시 매도 권장)"
              elif holding_return <= sl_pct:
                action = "STOP LOSS (손절가 이탈 - 즉시 손절 실행)"
              elif holding_return >= tp_pct:
                action = "TAKE PROFIT (목표 익절가 달성 - 수익 실현 권장)"
              elif passed_survive:
                action = "HOLD & RIDE (강력 홀딩 - 모멘텀 유지)"
              else:
                action = "WATCH / CAUTION (중립 관망)"

              st.subheader(f"🛡️ [{pos_ticker}] 기존 보유 포지션 진단 리포트")
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
                  '<div class="metric-title">손절 생존 확률</div>'
                  f'<div class="metric-value">{surv_prob:.1f}%</div>'
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
