# -------------------------------------------------------------
  # 1. 당일 종목 추천 (스캐너) - 코랩 추출 로직 100% 동기화 버전
  # -------------------------------------------------------------
  if menu == "🔥 당일 종목 추천 (스캐너)":
    st.header(
        f"🔥 당일 기준 정예 챔피언 스캐너 ({HOLDING_PERIOD}거래일 스윙 기준)"
    )
    st.write(
        "코랩의 실전 추론 셀과 완전히 동일한 로직(상위 10% 및 생존율 58% 이상)"
        "으로 스캔합니다."
    )

    run_scan = st.button("🔄 코랩 동일 기준 실시간 정밀 스캔 실행")

    df_display = None
    scan_timestamp = None

    with st.spinner(
        f"총 {len(UNIVERSE_TICKERS)}개 종목 데이터 수집 및 코랩 동일 앙상블"
        " 추론 중..."
    ):
      progress_bar = st.progress(0)
      status_text = st.empty()

      results = []
      total_tickers = len(UNIVERSE_TICKERS)

      # 1단계: 코랩 df_clean처럼 전체 유니버스의 최신 데이터프레임 행들을 수집
      latest_rows_list = []

      for idx, ticker in enumerate(UNIVERSE_TICKERS):
        status_text.text(
            f"({idx+1}/{total_tickers}) {ticker} 데이터 수집 및 피처 연산 중..."
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
          # 날짜 컬럼 인덱스를 일반 컬럼으로 변환 (코랩 df_clean 호환)
          df_t = df_t.reset_index()
          if "Date" not in df_t.columns and "Datetime" in df_t.columns:
            df_t = df_t.rename(columns={"Datetime": "Date"})
          elif "Date" not in df_t.columns:
            df_t["Date"] = df_t.iloc[:, 0]

          # 가장 마지막 행(최신 마감일) 추출
          latest_row = df_t.iloc[[-1]].copy()
          latest_row["Ticker"] = ticker
          latest_rows_list.append(latest_row)
        except Exception:
          continue

      progress_bar.empty()
      status_text.empty()

      if latest_rows_list:
        df_clean_all = pd.concat(latest_rows_list, ignore_index=True)

        # 2단계: 가장 최신 장 마감 날짜 추출 (코랩 코드 동일)
        latest_date = df_clean_all["Date"].max()
        scan_timestamp = pd.to_datetime(latest_date).strftime("%Y-%m-%d")

        # 3단계: Max ATR(4.5%) 안정형 변동성 필터 적용
        MAX_DAILY_ATR_PCT = 4.5
        latest_subset = df_clean_all[
            (df_clean_all["Date"] == latest_date)
            & (df_clean_all["ATR_Pct"] * 100 <= MAX_DAILY_ATR_PCT)
        ].copy()

        if len(latest_subset) > 0 and latest_subset[FEATURE_COLS].notna().all().all():
          # 4단계: 모델 추론 수행 (Booster .predict 사용)
          latest_subset["Pred_Score"] = ranker_lgb.predict(
              latest_subset[FEATURE_COLS]
          )
          # 코랩의 predict_proba[:, 1]과 동일한 Booster 이진분류 확률 예측
          latest_subset["Survive_Prob"] = classifier_lgb.predict(
              latest_subset[FEATURE_COLS]
          )
          latest_subset["Predicted_TP"] = tp_regressor.predict(
              latest_subset[FEATURE_COLS]
          )

          # 5단계: 랭킹 상위 10% 선별 후 생존 확률 58% 이상 필터링 (코랩 동일)
          group_sorted = latest_subset.sort_values(
              by="Pred_Score", ascending=False
          )
          top_k = max(1, int(len(group_sorted) * 0.10))
          top_candidates = group_sorted.head(top_k)

          MIN_SURVIVE_PROB_COLAB = 0.58  # 코랩 기준 58%
          filtered_latest = top_candidates[
              top_candidates["Survive_Prob"] >= MIN_SURVIVE_PROB_COLAB
          ].reset_index(drop=True)

          if len(filtered_latest) > 0:
            MAX_SL_LIMIT = -8.0
            display_results = []

            for idx, row in filtered_latest.iterrows():
              rank_num = idx + 1
              entry_price = float(row["Close"])
              atr_val = float(row["ATR_Pct"])

              # 코랩과 동일한 손절폭 계산 공식
              sl_pct = max(-(atr_val * 1.5 * 100), MAX_SL_LIMIT)
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
            st.success("✨ 코랩 추출 로직과 100% 일치하는 정예 챔피언 선별 완료!")
          else:
            st.warning(
                "⚠️ 오늘 마감 기준 모든 필터(상위 10% 랭킹, 생존율 58% 이상)를"
                " 통과한 종목이 없습니다."
            )
            df_display = pd.DataFrame()
        else:
          st.warning(
              "⚠️ 최신 마감일에 조건을 충족하는 유효한 데이터가 없습니다."
          )
          df_display = pd.DataFrame()
      else:
        st.error("데이터를 수집하지 못했습니다.")
        df_display = pd.DataFrame()

    if df_display is not None and not df_display.empty:
      if scan_timestamp:
        st.markdown(
            f"📅 **타겟 기준일 (최신 마감일):** `{scan_timestamp}` (코랩 동기화)"
        )
      st.dataframe(df_display, use_container_width=True)
