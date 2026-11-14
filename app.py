# -------------------------------------------------------------
  # 1. 당일 종목 추천 (스캐너)
  # -------------------------------------------------------------
  if menu == "🔥 당일 종목 추천 (스캐너)":
    st.header("🔥 당일 기준 정예 챔피언 스캐너")
    st.write(
        f"총 {len(UNIVERSE_TICKERS)}개 종목의 유니버스를 대상으로 최신 데이터를"
        " 수집하여 트레이딩 시그널을 도출합니다."
    )

    if st.button("실시간 스캔 실행"):
      progress_bar = st.progress(0)
      status_text = st.empty()

      results = []
      error_logs = []  # 👈 에러 로그를 담을 리스트
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
          # 👈 어떤 종목에서 어떤 에러가 났는지 상세히 기록
          error_logs.append(f"[{ticker}] 에러 내용: {str(e)}")
          continue

      progress_bar.empty()
      status_text.empty()

      if results:
        res_df = pd.DataFrame(results)
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
        st.error(
            "데이터를 불러오지 못했습니다. 아래 디버깅 로그를 확인해주세요."
        )

        # 💡 실패했을 때 수집된 에러 로그를 화면에 그대로 출력
        if error_logs:
          st.error(
              f"🚨 상세 에러 로그 총 {len(error_logs)}건 발생 (상위 10개 표시):"
          )
          st.code("\n".join(error_logs[:10]), language="text")
