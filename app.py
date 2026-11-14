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
  UNIVERSE_TICKERS = MODEL_CONFIG["universe_tickers"]


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
    df_fresh = df_fresh.dropna(subset=["Close"])

    # 1. 시장 데이터 (VIX, SPY) 가공 (셀 2 동일)
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

    # 2. 종목별 고도화 팩터 연산 (셀 2 동일)
    df_stocks = df_fresh[
        ~df_fresh["Ticker"].str.upper().isin(["VIX", "^VIX", "SPY"])
    ].copy()
    processed_dfs = []

    for ticker, group in df_stocks.groupby("Ticker"):
      g = group.copy().sort_values("Date").reset_index(drop=True)
      if len(g) < 100:
        continue

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
      g["Vol_Trend_5_20"] = v.rolling(5).mean() / (
          v.rolling(20).mean() + 1e-8
      )

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
      vwap = (v * typical_price).rolling(20).sum() / (
          v.rolling(20).sum() + 1e-8
      )
      g["VWAP_Ratio"] = (c - vwap) / (vwap + 1e-8)

      delta = c.diff()
      gain = (delta.where(delta > 0, 0)).rolling(14).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
      rs = gain / (loss + 1e-8)
      g["RSI_14"] = 100 - (100 / (1 + rs))

      processed_dfs.append(g)

    if not processed_dfs:
      return pd.DataFrame()

    df_all = pd.concat(processed_dfs, ignore_index=True)
    df_all = pd.merge(df_all, market_data, on="Date", how="inner")
    df_all = pd.merge(
        df_all, spy_df[["Date", "SPY_Ret_5d"]], on="Date", how="inner"
    )
    df_all["Rel_Strength_5d"] = df_all["Ret_5d"] - df_all["SPY_Ret_5d"]

    # 크로스 섹셔널 Z-Score
    df_all["Ret_5d_Zscore"] = df_all.groupby("Date")["Ret_5d"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    df_clean = df_all.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    return df_clean


  # -------------------------------------------------------------
  # 1. 당일 종목 추천 (스캐너) - 코랩 셀 1·2 완전 동기화 추론
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

    with st.spinner(
        "코랩 전처리 파이프라인 가동 중 (전체 벌크 다운로드 및 팩터 연산)..."
    ):
      try:
        df_clean = fetch_and_process_universe_data()

        if not df_clean.empty:
          latest_date = df_clean["Date"].max()
          scan_timestamp = pd.to_datetime(latest_date).strftime("%Y-%m-%d")

          MAX_DAILY_ATR_PCT = 0.045  # ATR_Pct는 소수점 형태 (4.5% = 0.045)
          latest_subset = df_clean[
              (df_clean["Date"] == latest_date)
              & (df_clean["ATR_Pct"] <= MAX_DAILY_ATR_PCT)
          ].copy()

          if (
              len(latest_subset) > 0
              and latest_subset[FEATURE_COLS].notna().all().all()
          ):
            latest_subset["Pred_Score"] = ranker_lgb.predict(
                latest_subset[FEATURE_COLS]
            )
            latest_subset["Survive_Prob"] = classifier_lgb.predict(
                latest_subset[FEATURE_COLS]
            )
            latest_subset["Predicted_TP"] = tp_regressor.predict(
                latest_subset[FEATURE_COLS]
            )

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
                atr_val = float(row["ATR_Pct"]) * 100  # % 환산

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
        st.markdown(
            f"📅 **타겟 기준일 (최신 마감일):** `{scan_timestamp}` (코랩 셀 1·2"
            " 완벽 동기화)"
        )
      st.dataframe(df_display, use_container_width=True)

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
          df_clean = fetch_and_process_universe_data()
          t_df = df_clean[df_clean["Ticker"] == target_ticker]

          if t_df.empty or len(t_df) < 30:
            st.error(
                f"'{target_ticker}'에 대한 충분한 데이터가 없습니다. 티커"
                " 오타를 확인해주세요."
            )
          else:
            latest = t_df.iloc[[-1]]
            x_val = latest[FEATURE_COLS]

            surv_prob = float(classifier_lgb.predict(x_val)[0] * 100)
            pred_tp = float(tp_regressor.predict(x_val)[0])
            close_p = float(latest["Close"].values[0])
            atr_val = float(latest["ATR_Pct"].values[0]) * 100

            sl_pct = max(-(atr_val * 1.5), -8.0)
            min_required_tp = abs(sl_pct) * 1.5
            tp_pct = max(pred_tp, min_required_tp)

            sl_price = close_p * (1 + sl_pct / 100.0)
            tp_price = close_p * (1 + tp_pct / 100.0)

            st.subheader(f"📊 {target_ticker} 분석 리포트")
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
            st.line_chart(t_df.set_index("Date")["Close"].tail(30))
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
          df_clean = fetch_and_process_universe_data()
          df_t = df_clean[df_clean["Ticker"] == target_ticker].copy()
          df_t["DateOnly"] = pd.to_datetime(df_t["Date"]).dt.date

          match_row = df_t[df_t["DateOnly"] == target_date]

          if not match_row.empty:
            row = match_row.iloc[0]
            x_val = row[FEATURE_COLS].to_frame().T

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

            # 향후 기간 수익률 확인
            full_t_df = yf.download(
                target_ticker,
                start="2018-01-01",
                progress=False,
                session=session,
            )
            full_t_df = full_t_df.reset_index()
            full_t_df["DateOnly"] = pd.to_datetime(
                full_t_df.iloc[:, 0]
            ).dt.date
            future_sub = full_t_df[
                full_t_df["DateOnly"] >= target_date
            ].head(HOLDING_PERIOD + 1)

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
                f"📊 {target_ticker} - {target_date} 기준"
                f" {HOLDING_PERIOD}일 검증 리포트"
            )

            if is_champion:
              st.success(
                  f"✨ **[{target_ticker}] 매수 진입 적격 챔피언 시그널 발생!**"
                  " (손절 생존 확률 >= 58% 및 변동성 기준 충족)"
              )
            else:
              st.warning(
                  f"⚠️ **[{target_ticker}] 조건 미달 (관망 권장)**"
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
                f'<div class="metric-value">{atr_val:.2f}%</div>'
                "</div>"
                "</div>"
            )
            st.markdown(card_html_1, unsafe_allow_html=True)
          else:
            st.warning(
                "선택한 날짜에 해당하는 거래 데이터가 없습니다 (휴장일 등)."
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
          df_clean = fetch_and_process_universe_data()
          t_df = df_clean[df_clean["Ticker"] == pos_ticker]

          if t_df.empty or len(t_df) < 30:
            st.error(f"'{pos_ticker}' 종목 데이터를 충분히 가져오지 못했습니다.")
          else:
            latest_row = t_df.iloc[[-1]]
            curr_price = float(latest_row["Close"].values[0])
            holding_return = (
                (curr_price - pos_price) / pos_price * 100
            )

            x_val = latest_row[FEATURE_COLS]
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
