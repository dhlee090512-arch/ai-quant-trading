import json
from datetime import datetime, timedelta
import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# 페이지 기본 설정
st.set_page_config(
    page_title="7일 스윙 퀀트 대시보드", page_icon="📈", layout="wide"
)

st.title("🚀 AI 기반 7일 스윙 퀀트 트레이딩 시스템")
st.write(
    "LightGBM 3중 앙상블 모델(Ranking + Survival + TP Regressor) 기반 실전"
    " 분석 대시보드"
)


# 1. 깃허브에 저장된 모델 및 피처 목록 로드 (캐싱 적용)
@st.cache_resource
def load_models_and_features():
  try:
    ranker = lgb.Booster(model_file="models/ranker_model.txt")
    classifier = lgb.Booster(model_file="models/classifier_model.txt")
    tp_reg = lgb.Booster(model_file="models/tp_regressor_model.txt")

    with open("models/feature_cols.json", "r") as f:
      feature_cols = json.load(f)

    return ranker, classifier, tp_reg, feature_cols
  except Exception as e:
    st.error(
        f"모델 파일을 불러오는 중 오류가 발생했습니다. (models/ 폴더 확인 필요): {e}"
    )
    return None, None, None, None


ranker_lgb, classifier_lgb, tp_regressor, FEATURE_COLS = (
    load_models_and_features()
)

if RANKER_LOADED := (ranker_lgb is not None):
  # 사이드바 메뉴 구성
  menu = st.sidebar.selectbox(
      "메뉴 선택",
      [
          "🔥 당일 종목 추천 (스캐너)",
          "🔍 특정 종목 당일 분석",
          "⏳ 특정 종목 과거 시점 분석",
      ],
  )

  # 분석 유니버스 정의
  UNIVERSE_TICKERS = [
      "AAPL",
      "MSFT",
      "GOOGL",
      "AMZN",
      "NVDA",
      "TSLA",
      "META",
      "NFLX",
      "AMD",
      "INTC",
      "QCOM",
      "AVGO",
      "TXN",
      "AMAT",
      "MU",
      "LRCX",
  ]


  # 코랩과 동일한 피처 엔지니어링 함수
  def compute_features(df):
    df["Return_1d"] = df["Close"].pct_change(1)
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_60d"] = df["Close"].pct_change(60)
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
    st.header("🔥 당일 기준 정예 챔피언 스캐너 (7일 스윙)")
    st.write(
        "유니버스 전체 종목의 최신 데이터를 수집하여 7일 스윙 시그널을 도출합니다."
    )

    if st.button("실시간 스캔 실행"):
      with st.spinner("데이터 수집 및 모델 추론 중..."):
        results = []
        for ticker in UNIVERSE_TICKERS:
          try:
            df_t = yf.download(
                ticker, period="6mo", progress=False, auto_adjust=True
            )
            if isinstance(df_t.columns, pd.MultiIndex):
              df_t.columns = df_t.columns.get_level_values(0)
            df_t = compute_features(df_t)
            latest = df_t.iloc[[-1]].copy()

            if latest[FEATURE_COLS].isna().sum().sum() == 0:
              x_val = latest[FEATURE_COLS]
              score = ranker_lgb.predict(x_val)[0]
              surv_prob = classifier_lgb.predict(x_val)[0] * 100
              pred_tp = tp_regressor.predict(x_val)[0]

              results.append({
                  "Ticker": ticker,
                  "Close": latest["Close"].values[0],
                  "ATR_Pct": latest["ATR_Pct"].values[0] * 100,
                  "Pred_Score": score,
                  "Survive_Prob": surv_prob,
                  "Predicted_TP": pred_tp,
              })
          except Exception as e:
            continue

        if results:
          res_df = pd.DataFrame(results)
          # 생존 확률 62% 이상 필터링
          filtered = res_df[res_df["Survive_Prob"] >= 62.0].sort_values(
              by="Pred_Score", ascending=False
          )

          if len(filtered) > 0:
            st.success(
                f"총 {len(filtered)}개의 정예 챔피언 종목이 포착되었습니다!"
            )
            st.dataframe(filtered.reset_index(drop=True))
          else:
            st.warning(
                "오늘 기준 조건(생존확률 62% 이상)을 충족하는 종목이 없습니다."
                " (관망 추천)"
            )
        else:
          st.error("데이터를 불러오지 못했습니다.")

  # -------------------------------------------------------------
  # 2. 특정 종목 당일 분석
  # -------------------------------------------------------------
  elif menu == "🔍 특정 종목 당일 분석":
    st.header("🔍 특정 종목 당일 딥다이브 분석")
    target_ticker = st.text_input(
        "분석할 티커 입력 (예: AAPL, GOOGL)", "GOOGL"
    ).upper()

    if st.button("분석 시작"):
      with st.spinner(f"{target_ticker} 분석 중..."):
        try:
          df_t = yf.download(
              target_ticker, period="6mo", progress=False, auto_adjust=True
          )
          if isinstance(df_t.columns, pd.MultiIndex):
            df_t.columns = df_t.columns.get_level_values(0)
          df_t = compute_features(df_t)
          latest = df_t.iloc[[-1]].copy()

          x_val = latest[FEATURE_COLS]
          score = ranker_lgb.predict(x_val)[0]
          surv_prob = classifier_lgb.predict(x_val)[0] * 100
          pred_tp = tp_regressor.predict(x_val)[0]
          close_p = latest["Close"].values[0]
          atr_p = latest["ATR_Pct"].values[0]

          sl_pct = max(-(atr_p * 1.5 * 100), -8.0)
          tp_pct = max(pred_tp, abs(sl_pct) * 1.5)

          st.metric(label="현재 종가", value=f"${close_p:.2f}")
          col1, col2, col3 = st.columns(3)
          col1.metric("모델 생존 확률", f"{surv_prob:.1f}%")
          col2.metric("권장 손절폭", f"{sl_pct:.2f}%")
          col3.metric("예상 목표익절폭", f"+{tp_pct:.2f}%")

          st.line_chart(df_t["Close"].tail(30))
        except Exception as e:
          st.error(f"분석 중 오류 발생: {e}")

  # -------------------------------------------------------------
  # 3. 특정 종목 과거 시점 분석
  # -------------------------------------------------------------
  elif menu == "⏳ 특정 종목 과거 시점 분석":
    st.header("⏳ 특정 종목 과거 특정 날짜 기준 시그널 검증")
    target_ticker = st.text_input(
        "분석할 티커 입력 (예: TSLA)", "TSLA", key="past_ticker"
    ).upper()
    target_date = st.date_input(
        "조회할 과거 날짜 선택", datetime.now() - timedelta(days=30)
    )

    if st.button("과거 시점 시그널 조회"):
      try:
        df_t = yf.download(
            target_ticker, period="2y", progress=False, auto_adjust=True
        )
        if isinstance(df_t.columns, pd.MultiIndex):
          df_t.columns = df_t.columns.get_level_values(0)
        df_t = compute_features(df_t)
        df_t.index = pd.to_datetime(df_t.index).date()

        if target_date in df_t.index:
          row = df_t.loc[[target_date]]
          x_val = row[FEATURE_COLS]

          surv_prob = classifier_lgb.predict(x_val)[0] * 100
          pred_tp = tp_regressor.predict(x_val)[0]
          close_p = row["Close"].values[0]

          st.success(f"{target_date} 기준 {target_ticker} 분석 결과")
          st.write(f"- 당일 종가: ${close_p:.2f}")
          st.write(f"- 모델 생존 확률: {surv_prob:.1f}%")
          st.write(f"- 예측 익절폭: +{pred_tp:.2f}%")

          sub_future = df_t.loc[target_date:].head(8)
          st.line_chart(sub_future["Close"])
        else:
          st.warning(
              "선택한 날짜에 해당하는 거래 데이터가 없습니다 (휴장일 등)."
          )
      except Exception as e:
        st.error(f"과거 데이터 조회 중 오류 발생: {e}")
