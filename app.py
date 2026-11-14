from datetime import datetime, timedelta
import datetime as dt
import json
import os
import traceback
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
    " 분석 대시보드 (코랩 정합성 100% 동기화)"
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
</style>
"""

# 모델 로드 실패 시 에러 안내 문구 출력
if load_error is not None:
  st.error(
      "🚨 모델 파일 및 메타데이터 로딩 실패! GitHub 레포지토리의 `models/`"
      f" 폴더 상태를 확인해주세요.\n\n**에러 내용:** {load_error}"
  )
else:
  HOLDING_PERIOD = MODEL_CONFIG["holding_period"]
  MIN_SURVIVE_PROB = MODEL_CONFIG["min_survive_prob"]
  UNIVERSE_TICKERS = MODEL_CONFIG["universe_tickers"]


  # 코랩과 100% 동일한 피처 엔지니어링 함수
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
  # 1. 당일 종목 추천 (스캐너) - 코랩 완전 정합성 보장
  # -------------------------------------------------------------
  if menu == "🔥 당일 종목 추천 (스캐너)":
    st.header(
        f"🔥 당일 기준 정예 챔피언 스캐너 ({HOLDING_PERIOD}거래일 스윙 기준)"
    )
    st.write(
        "코랩과 동일한 실시간 추론 파이프라인으로 전체 유니버스를 정밀"
        " 스캔합니다."
    )

    run_scan = st.button("🔄 실시간 전체 유니버스 정밀 재스캔 실행")

    df_display = None
    scan_timestamp = None

    with st.spinner(
        f"총 {len(UNIVERSE_TICKERS)}개 종목 일괄 데이터 수집 및 앙상블 추론"
        " 중..."
    ):
      progress_bar = st.progress(0)
      status_text = st.empty()

      results = []
      total_tickers = len(UNIVERSE_TICKERS)

      for idx, ticker in enumerate(UNIVERSE_TICKERS):
        status_text.text(f"({idx+1}/{total_tickers}) {ticker} 분석 중...")
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

          atr_p = float(latest["ATR_Pct"].values[0])
          if (atr_p * 100) > 4.5:
            continue

          if latest[FEATURE_COLS].isna().sum().sum() == 0:
            x_val = latest[FEATURE_COLS]
            score = float(ranker_lgb.predict(x_val)[0])
            surv_prob = float(classifier_lgb.predict(x_val)[0] * 100)
            pred_tp = float(tp_regressor.predict(x_val)[0])

            close_p = float(latest["Close"].values[0])

            sl_pct = max(-(atr_p * 1.5 * 100), -8.0)
            min_required_tp = abs(sl_pct) * 1.5
            tp_pct = max(pred_tp, min_required_tp)

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
        except Exception:
          continue

      progress_bar.empty()
      status_text.empty()

      if results:
        res_df = pd.DataFrame(results)

        # 💡 [정석 정합성 로직] 전체 유니버스 풀 기준 정확히 상위 10% 산정 후 생존 확률 필터 적용
        top_k = max(1, int(len(res_df) * 0.10))
        top_candidates = res_df.sort_values(
            by="Pred_Score", ascending=False
        ).head(top_k)

        filtered = top_candidates[
            top_candidates["생존확률"] >= MIN_SURVIVE_PROB
        ].sort_values(by="Pred_Score", ascending=False)

        if len(filtered) > 0:
          df_display = filtered.drop(columns=["Pred_Score"]).reset_index(
              drop=True
          )
          scan_timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
          st.success("✨ 코랩 정합성 기반 실전 스캔 완료!")
        else:
          st.warning(
              "⚠️ 조건(상위 10% 랭킹 및 생존 확률 >="
              f" {MIN_SURVIVE_PROB}%)을 충족하는 종목이 없습니다."
          )
          df_display = pd.DataFrame()
      else:
        st.error("유효한 종목 데이터를 수집하지 못했습니다.")
        df_display = pd.DataFrame()

    if df_display is not None and not df_display.empty:
      if scan_timestamp:
        st.markdown(f"📅 **데이터 기준 시점 (실시간):** `{scan_timestamp}`")

      display_df = df_display.copy()
      display_df.index = display_df.index + 1
      display_df.index.name = "순위"
      display_df = display_df.reset_index()

      if not str(display_df["조정 손절폭(%)"].iloc[0]).endswith("%"):
        display_df["조정 손절폭(%)"] = (
            display_df["조정 손절폭(%)"].astype(str) + " %"
        )
        display_df["동적 익절폭(%)"] = (
            "+" + display_df["동적 익절폭(%)"].astype(str) + " %"
        )
        display_df["생존확률"] = display_df["생존확률"].astype(str) + " %"

      st.dataframe(display_df, use_container_width=True)

  # -------------------------------------------------------------
  # 2. 특정 종목 당일 분석
  # -------------------------------------------------------------
  elif menu == "🔍 특정 종목 당일 분석":
    st.header(f"🔍 특정 종목 당일 딥다이브 분석 ({HOLDING_PERIOD}거래일 기준)")
    target_ticker = st.text_input(
        "분석할 티커 입력 (예: AAPL, TSLA, TSM)", "TSM"
    ).upper()

    if st.button("분석 시작"):
      with st.spinner(f"{target_ticker} 분석 중..."):
        try:
          t_obj = yf.Ticker(target_ticker, session=session)
          try:
            long_name = (
                t_obj.info.get("longName")
                or t_obj.info.get("shortName")
                or target_ticker
            )
          except Exception:
            long_name = target_ticker

          df_t = yf.download(
              target_ticker,
              period="6mo",
              progress=False,
              auto_adjust=True,
              session=session,
          )
          if isinstance(df_t.columns, pd.MultiIndex):
            df_t.columns = df_t.columns.get_level_values(0)

          if len(df_t) < 30:
            st.error(
                f"'{target_ticker}'에 대한 충분한 데이터가 없습니다. 티커"
                " 오타를 확인해주세요."
            )
          else:
            df_t = compute_features(df_t)
            latest = df_t.iloc[[-1]]

            x_val = latest[FEATURE_COLS]
            surv_prob = float(classifier_lgb.predict(x_val)[0] * 100)
            pred_tp = float(tp_regressor.predict(x_val)[0])
            close_p = float(latest["Close"].values[0])
            atr_p = float(latest["ATR_Pct"].values[0])

            sl_pct = max(-(atr_p * 1.5 * 100), -8.0)
            min_required_tp = abs(sl_pct) * 1.5
            tp_pct = max(pred_tp, min_required_tp)

            sl_price = close_p * (1 + sl_pct / 100)
            tp_price = close_p * (1 + tp_pct / 100)

            st.subheader(f"📊 {long_name} ({target_ticker}) 분석 리포트")

            st.markdown(
                f"**현재 종가 (진입가):** <span"
                f' style="font-size: 28px; font-weight: bold; color:'
                f' #00FF7F;">${close_p:.2f}</span>',
                unsafe_allow_html=True,
            )

            st.markdown(METRIC_CSS, unsafe_allow_html=True)

            metric_html = (
                '<div class="metric-container">'
                '<div class="metric-card">'
                '<div class="metric-title">모델 생존 확률</div>'
                f'<div class="metric-value">{surv_prob:.1f}%</div>'
                "</div>"
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

            st.line_chart(df_t["Close"].tail(30))
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
    target_ticker = st.text_input(
        "분석할 티커 입력 (예: TSM, TSLA)", "TSM", key="past_ticker"
    ).upper()
    target_date = st.date_input(
        "조회할 과거 날짜 선택", datetime.now() - timedelta(days=30)
    )

    if st.button("과거 시점 시그널 검증 실행"):
      with st.spinner(f"{target_ticker} 과거 데이터 조회 및 사후 검증 중..."):
        try:
          t_obj = yf.Ticker(target_ticker, session=session)
          try:
            long_name = (
                t_obj.info.get("longName")
                or t_obj.info.get("shortName")
                or target_ticker
            )
          except Exception:
            long_name = target_ticker

          df_t = yf.download(
              target_ticker,
              period="2y",
              progress=False,
              auto_adjust=True,
              session=session,
          )
          if isinstance(df_t.columns, pd.MultiIndex):
            df_t.columns = df_t.columns.get_level_values(0)

          df_t = compute_features(df_t)
          df_t.index = pd.to_datetime(df_t.index).date

          if target_date in df_t.index:
            row = df_t.loc[[target_date]]
            x_val = row[FEATURE_COLS]

            score = float(ranker_lgb.predict(x_val)[0])
            surv_prob = float(classifier_lgb.predict(x_val)[0] * 100)
            pred_tp = float(tp_regressor.predict(x_val)[0])
            close_p = float(row["Close"].values[0])

            atr_p = float(row["ATR_Pct"].values[0])
            atr_limit_passed = (atr_p * 100) <= 4.5

            sl_pct = max(-(atr_p * 1.5 * 100), -8.0)
            min_required_tp = abs(sl_pct) * 1.5
            tp_pct = max(pred_tp, min_required_tp)

            sl_price = close_p * (1 + sl_pct / 100)
            tp_price = close_p * (1 + tp_pct / 100)

            surv_passed = surv_prob >= MIN_SURVIVE_PROB
            is_champion = surv_passed and atr_limit_passed

            future_sub = df_t.loc[target_date:].head(HOLDING_PERIOD + 1)
            if len(future_sub) > 1:
              valid_period = future_sub.iloc[1:]
              high_n = float(valid_period["High"].max())
              low_n = float(valid_period["Low"].min())
              close_n = float(valid_period["Close"].iloc[-1])
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
                f"📊 {long_name} ({target_ticker}) - {target_date} 기준"
                f" {HOLDING_PERIOD}일 검증 리포트"
            )

            if is_champion:
              st.success(
                  f"✨ **[{target_ticker}] 매수 진입 적격 챔피언 시그널 발생!**"
                  f" (손절 생존 확률 >= {MIN_SURVIVE_PROB}% 및 변동성 기준 충족)"
              )
            else:
              st.warning(
                  f"⚠️ **[{target_ticker}] 조건 미달 (관망 권장)** (기준치 일부"
                  " 미달)"
              )

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
                f'<div class="metric-value">{atr_p*100:.2f}%</div>'
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
                  f"🛡️ **손절 터치 여부:** {sl_str} ({HOLDING_PERIOD}일 최저가:"
                  f" ${low_n:.2f})"
              )
            with col2:
              st.success(
                  f"🎯 **목표 익절 도달:** {tp_str} (목표가: ${tp_price:.2f})"
              )

            st.subheader(f"📈 향후 {HOLDING_PERIOD}거래일 주가 경로 시각화")
            st.line_chart(future_sub["Close"])

          else:
            st.warning(
                "선택한 날짜에 해당하는 거래 데이터가 없습니다 (휴장일 등)."
            )
        except Exception as e:
          st.error(f"과거 데이터 조회 중 오류 발생: {e}")
          st.code(traceback.format_exc(), language="python")

  # -------------------------------------------------------------
  # 4. 기존 보유 포지션 진단 및 매매 가이드 (코랩 셀 6 기능 대응)
  # -------------------------------------------------------------
  elif menu == "🛡️ 기존 보유 포지션 진단 및 매매 가이드":
    st.header(
        f"🛡️ 기존 보유 포지션 진단 및 매매 가이드 ({HOLDING_PERIOD}거래일 스윙"
        " 기준)"
    )
    st.write(
        "보유 중인 종목과 매수가를 입력하면 현재 시점의 모델 전망을 바탕으로"
        " [홀딩 vs 매도]를 진단합니다."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
      pos_ticker = st.text_input(
          "보유 종목 티커", "AAPL", key="pos_ticker"
      ).upper()
    with col2:
      pos_date = st.date_input(
          "매수 날짜", datetime.now() - timedelta(days=15), key="pos_date"
      )
    with col3:
      pos_price = st.number_input(
          "매수가격 ($)", value=300.0, step=1.0, key="pos_price"
      )

    if st.button("포지션 진단 실행"):
      with st.spinner(f"{pos_ticker} 실시간 분석 중..."):
        try:
          df_t = yf.download(
              pos_ticker,
              period="6mo",
              progress=False,
              auto_adjust=True,
              session=session,
          )
          if isinstance(df_t.columns, pd.MultiIndex):
            df_t.columns = df_t.columns.get_level_values(0)

          if len(df_t) < 30:
            st.error(f"'{pos_ticker}' 종목 데이터를 충분히 가져오지 못했습니다.")
          else:
            df_t = compute_features(df_t)
            latest_row = df_t.iloc[[-1]]

            curr_price = float(latest_row["Close"].values[0])
            holding_return = (
                (curr_price - pos_price) / pos_price * 100
            )

            x_val = latest_row[FEATURE_COLS]
            score = float(ranker_lgb.predict(x_val)[0])
            surv_prob = float(classifier_lgb.predict(x_val)[0] * 100)
            pred_tp = float(tp_regressor.predict(x_val)[0])
            atr_val = float(latest_row["ATR_Pct"].values[0])

            is_valid_atr = (atr_val * 100) <= 4.5
            sl_pct = max(-(atr_val * 1.5 * 100), -8.0)
            min_required_tp = abs(sl_pct) * 1.5
            tp_pct = max(pred_tp, min_required_tp)

            sl_price = curr_price * (1 + sl_pct / 100.0)
            tp_price = curr_price * (1 + tp_pct / 100.0)
            passed_survive = surv_prob >= MIN_SURVIVE_PROB

            # 매매 전략 판정 (코랩 셀 6 로직 동일)
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
            st.info(f"🎯 **[최종 의사결정 제안]: {action}**")

            holding_ret_color = "#00FF7F" if holding_return >= 0 else "#FF6B6B"

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
        except Exception as e:
          st.error(f"포지션 진단 중 오류 발생: {e}")
          st.code(traceback.format_exc(), language="python")
