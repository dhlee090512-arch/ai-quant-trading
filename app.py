from datetime import datetime, timedelta
import datetime as dt
import json
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# 페이지 기본 설정
st.set_page_config(
    page_title="AI 퀀트 트레이딩 대시보드", page_icon="📈", layout="wide"
)

st.title("🚀 AI 기반 퀀트 트레이딩 시스템")
st.write(
    "LightGBM 3중 앙상블 모델(Ranking + Survival + TP Regressor) 기반 실전"
    " 분석 대시보드"
)

# 야후 파이낸스 차단 우회를 위한 세션 설정
session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML,"
        " like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
})


# 1. 깃허브에 저장된 모델 및 피처 목록 로드 (캐싱 적용)
@st.cache_resource
def load_models_and_features():
  try:
    ranker = lgb.Booster(model_file="models/ranker_model.txt")
    classifier = lgb.Booster(model_file="models/classifier_model.txt")
    tp_reg = lgb.Booster(model_file="models/tp_regressor_model.txt")

    with open("models/feature_cols.json", "r") as f:
      feature_cols = json.load(f)

    return ranker, classifier, tp_reg, feature_cols, None
  except Exception as e:
    return None, None, None, None, str(e)


ranker_lgb, classifier_lgb, tp_regressor, FEATURE_COLS, load_error = (
    load_models_and_features()
)

# 사이드바 메뉴 구성 (모델 로드 여부와 상관없이 항상 노출)
menu = st.sidebar.selectbox(
    "메뉴 선택",
    [
        "🔥 당일 종목 추천 (스캐너)",
        "🔍 특정 종목 당일 분석",
        "⏳ 특정 종목 과거 시점 분석",
    ],
)

# 모델 로드 실패 시 에러 안내 문구 출력
if load_error is not None:
  st.error(
      "🚨 모델 파일 로딩 실패! GitHub 레포지토리의 `models/` 폴더 안에 모델"
      f" 파일들이 정상적으로 업로드되었는지 확인해주세요.\n\n**에러 내용:**"
      f" {load_error}"
  )
else:
  # 분석 유니버스 티커 리스트
  UNIVERSE_TICKERS = [
      "AAPL",
      "MSFT",
      "GOOGL",
      "AMZN",
      "NVDA",
      "META",
      "TSLA",
      "BRK-B",
      "LLY",
      "AVGO",
      "JPM",
      "XOM",
      "UNH",
      "V",
      "PG",
      "JNJ",
      "MA",
      "HD",
      "COST",
      "MRK",
      "ABBV",
      "NFLX",
      "AMD",
      "BAC",
      "CRM",
      "CVX",
      "WMT",
      "COIN",
      "PLTR",
      "TSM",
      "INTC",
      "QCOM",
      "IBM",
      "AMAT",
      "LRCX",
      "MU",
      "ARM",
      "MRVL",
      "SMCI",
      "SOXL",
      "TQQQ",
      "SQQQ",
      "BA",
      "DIS",
      "NKE",
      "SBUX",
      "PYPL",
      "SHOP",
      "UBER",
      "ABNB",
      "SNOW",
      "NET",
      "CRWD",
      "DDOG",
      "ZS",
      "PANW",
      "NOW",
      "MDB",
      "TEAM",
      "ADBE",
      "INTU",
      "ORCL",
      "CSCO",
      "ACN",
      "TXN",
      "NEE",
      "UNP",
      "HON",
      "UPS",
      "CAT",
      "GE",
      "RTX",
      "LOW",
      "SPGI",
      "BLK",
      "MARA",
  ]


  # 피처 엔지니어링 함수
  def compute_features(df):
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
      if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Close", "High", "Low", "Open", "Volume"])

    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_60d"] = df["Close"].pct_change(60)

    df["Ret_5d"] = df["Close"].pct_change(5)
    df["Ret_20d"] = df["Close"].pct_change(20)
    df["Ret_60d"] = df["Close"].pct_change(60)
    df["Rel_Strength_5d"] = df["Ret_5d"] - df["Ret_5d"].rolling(20).mean()

    df["MA_5"] = df["Close"] / df["Close"].rolling(5).mean() - 1
    df["MA_20"] = df["Close"] / df["Close"].rolling(20).mean() - 1
    df["MA_60"] = df["Close"] / df["Close"].rolling(60).mean() - 1
    df["Dist_MA5"] = df["Close"] - df["Close"].rolling(5).mean()
    df["Dist_MA20"] = df["Close"] - df["Close"].rolling(20).mean()
    df["Dist_MA60"] = df["Close"] - df["Close"].rolling(60).mean()

    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()
    df["ATR_Pct"] = df["ATR"] / df["Close"]

    df["Vol_Ratio_5d"] = df["Volume"] / df["Volume"].rolling(5).mean()
    df["Vol_Ratio_20d"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Vol_Trend_5_20"] = df["Volume"].rolling(5).mean() / df[
        "Volume"
    ].rolling(20).mean()

    h_60 = df["High"].rolling(60).max()
    l_60 = df["Low"].rolling(60).min()
    df["Pos_Box_60d"] = (df["Close"] - l_60) / (h_60 - l_60 + 1e-8)
    df["Pos_Box_20d"] = (df["Close"] - df["Low"].rolling(20).min()) / (
        df["High"].rolling(20).max() - df["Low"].rolling(20).min() + 1e-8
    )
    df["Pos_Box_5d"] = (df["Close"] - df["Low"].rolling(5).min()) / (
        df["High"].rolling(5).max() - df["Low"].rolling(5).min() + 1e-8
    )

    df["Body_Ratio"] = (df["Close"] - df["Open"]) / (df["Open"] + 1e-8)
    df["Upper_Shadow"] = (df["High"] - np.maximum(df["Close"], df["Open"])) / (
        df["Close"] + 1e-8
    )
    df["Lower_Shadow"] = (np.minimum(df["Close"], df["Open"]) - df["Low"]) / (
        df["Close"] + 1e-8
    )
    df["Close_Pos"] = (df["Close"] - df["Low"]) / (
        df["High"] - df["Low"] + 1e-8
    )
    df["Gap_Pct"] = (
        (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100
    )
    df["Range_Expansion"] = (df["High"] - df["Low"]) / (
        df["High"].rolling(5).mean() - df["Low"].rolling(5).mean() + 1e-8
    )
    df["Ret_5d_Zscore"] = (
        df["Return_5d"] - df["Return_5d"].rolling(20).mean()
    ) / (df["Return_5d"].rolling(20).std() + 1e-8)
    df["VWAP_Ratio"] = df["Close"] / (
        (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3)
        .rolling(20)
        .sum()
        / df["Volume"].rolling(20).sum()
        + 1e-8
    )

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    df["VIX"] = 18.5
    df["VIX_Change_1d"] = 0.0
    df["VIX_Change_5d"] = 0.0
    df["VIX_Pos_20d"] = 0.5

    return df


  # -------------------------------------------------------------
  # 1. 당일 종목 추천 (스캐너)
  # -------------------------------------------------------------
  if menu == "🔥 당일 종목 추천 (스캐너)":
    st.header("🔥 당일 기준 정예 챔피언 스캐너")
    st.write(
        "저장된 최신 스캔 데이터를 즉시 불러옵니다. 강제로 새로 스캔하려면 아래"
        " 버튼을 누르세요."
    )

    cache_file = "models/latest_scan_results.csv"
    run_scan = st.button("🔄 실시간 전체 유니버스 재스캔 실행")

    df_display = None
    scan_timestamp = None

    if not run_scan and os.path.exists(cache_file):
      try:
        df_display = pd.read_csv(cache_file)
        mtime = os.path.getmtime(cache_file)
        scan_timestamp = dt.datetime.fromtimestamp(mtime).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        st.info("📂 캐시된 최신 스캔 데이터를 즉시 불러왔습니다.")
      except Exception:
        run_scan = True

    if run_scan or df_display is None:
      with st.spinner(
          f"총 {len(UNIVERSE_TICKERS)}개 종목 실시간 스캔 및 추론 중..."
      ):
        progress_bar = st.progress(0)
        status_text = st.empty()

        results = []
        error_logs = []
        total_tickers = len(UNIVERSE_TICKERS)

        for idx, ticker in enumerate(UNIVERSE_TICKERS):
          status_text.text(
              f"({idx+1}/{total_tickers}) {ticker} 데이터 수집 및 분석 중..."
          )
          progress_bar.progress((idx + 1) / total_tickers)

          try:
            df_t = yf.download(
                ticker,
                period="6mo",
                progress=False,
                auto_adjust=True,
                session=session,
            )
            if isinstance(df_t.columns, pd.MultiIndex):
              df_t.columns = df_t.columns.get_level_values(0)

            if len(df_t) < 30:
              continue

            df_t = compute_features(df_t)
            latest = df_t.iloc[[-1]]

            if latest[FEATURE_COLS].isna().sum().sum() == 0:
              x_val = latest[FEATURE_COLS]
              score = ranker_lgb.predict(x_val)[0]
              surv_prob = classifier_lgb.predict(x_val)[0] * 100
              pred_tp = tp_regressor.predict(x_val)[0]

              close_p = float(latest["Close"].values[0])
              atr_p = float(latest["ATR_Pct"].values[0])

              sl_pct = max(-(atr_p * 1.5 * 100), -8.0)
              tp_pct = max(pred_tp, abs(sl_pct) * 1.5)

              sl_price = close_p * (1 + sl_pct / 100)
              tp_price = close_p * (1 + tp_pct / 100)

              results.append({
                  "Ticker": ticker,
                  "진입가($)": round(close_p, 2),
                  "손절가($)": round(sl_price, 2),
                  "조정 손절폭(%)": round(sl_pct, 2),
                  "동적익절가($)": round(tp_price, 2),
                  "동적 익절폭(%)": round(tp_pct, 2),
                  "생존확률": round(surv_prob, 1),
                  "Pred_Score": score,
              })
          except Exception as e:
            error_logs.append(f"[{ticker}] 에러 내용: {str(e)}")
            continue

        progress_bar.empty()
        status_text.empty()

        if results:
          res_df = pd.DataFrame(results)
          filtered = res_df[res_df["생존확률"] >= 62.0].sort_values(
              by="Pred_Score", ascending=False
          )

          if len(filtered) > 0:
            df_display = filtered.drop(columns=["Pred_Score"]).reset_index(
                drop=True
            )
            os.makedirs("models", exist_ok=True)
            df_display.to_csv(cache_file, index=False)

            scan_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.success("✨ 실시간 재스캔 완료 및 최신 데이터가 저장되었습니다!")
          else:
            st.warning("조건을 충족하는 종목이 없습니다.")
            df_display = pd.DataFrame()
        else:
          st.error("데이터를 불러오지 못했습니다.")
          if error_logs:
            st.code("\n".join(error_logs), language="text")
          df_display = pd.DataFrame()

    if df_display is not None and not df_display.empty:
      if scan_timestamp:
        st.markdown(f"📅 **데이터 기준 시점 (최종 스캔):** `{scan_timestamp}`")

      display_df = df_display.copy()
      display_df.index = display_df.index + 1
      display_df.index.name = "순위"
      display_df = display_df.reset
