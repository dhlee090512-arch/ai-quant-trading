# AI Quant Daily Automated Pipeline
import os
import json
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor

print("🚀 [1/4] 데이터 수집 및 피처 엔지니어링 시작...")
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'LLY', 'AVGO',
    'JPM', 'XOM', 'UNH', 'V', 'PG', 'JNJ', 'MA', 'HD', 'COST', 'MRK',
    'ABBV', 'NFLX', 'AMD', 'BAC', 'CRM', 'CVX', 'WMT', 'COIN', 'PLTR', 'TSM',
    'INTC', 'QCOM', 'IBM', 'AMAT', 'LRCX', 'MU', 'ARM', 'MRVL', 'SMCI', 'SOXL',
    'TQQQ', 'SQQQ', 'SPY', 'QQQ', 'IWM', 'ARKK', 'BA', 'DIS', 'NKE', 'SBUX',
    'PYPL', 'SHOP', 'UBER', 'ABNB', 'SNOW', 'NET', 'CRWD', 'DDOG', 'ZS', 'PANW',
    'NOW', 'MDB', 'TEAM', 'ADBE', 'INTU', 'ORCL', 'CSCO', 'ACN', 'TXN', 'NEE',
    'UNP', 'HON', 'UPS', 'CAT', 'GE', 'RTX', 'LOW', 'SPGI', 'BLK', 'MARA'
]

df_raw_yf = yf.download(tickers + ['^VIX', 'SPY'], start="2018-01-01", progress=False)

if isinstance(df_raw_yf.columns, pd.MultiIndex):
    df_fresh = df_raw_yf.stack(future_stack=True).reset_index()
    if 'level_1' in df_fresh.columns:
        df_fresh = df_fresh.rename(columns={'level_1': 'Ticker'})
else:
    df_fresh = df_raw_yf.reset_index()

df_fresh.columns = [str(col[0]).lower() if isinstance(col, tuple) else str(col).lower() for col in df_fresh.columns]
rename_map = {}
for c in df_fresh.columns:
    if 'date' in c: rename_map[c] = 'Date'
    elif 'ticker' in c: rename_map[c] = 'Ticker'
    elif 'open' in c: rename_map[c] = 'Open'
    elif 'high' in c: rename_map[c] = 'High'
    elif 'low' in c: rename_map[c] = 'Low'
    elif 'close' in c and 'adj' not in c: rename_map[c] = 'Close'
    elif 'volume' in c: rename_map[c] = 'Volume'
df_fresh = df_fresh.rename(columns=rename_map)
df_fresh['Date'] = pd.to_datetime(df_fresh['Date']).dt.tz_localize(None)
df_fresh = df_fresh.dropna(subset=['Close'])

# 피처 가공
market_data = df_fresh[df_fresh['Ticker'].str.upper().isin(['^VIX', 'VIX', 'SPY'])][['Date', 'Ticker', 'Close']].copy()
market_data['Ticker'] = market_data['Ticker'].str.upper().str.replace('^', '', regex=False)
market_data = market_data.pivot(index='Date', columns='Ticker', values='Close').reset_index()
market_data.columns.name = None
market_data = market_data.sort_values('Date').ffill()

market_data['VIX_Change_1d'] = market_data['VIX'].pct_change(1)
market_data['VIX_Change_5d'] = market_data['VIX'].pct_change(5)
roll_min = market_data['VIX'].rolling(20).min()
roll_max = market_data['VIX'].rolling(20).max()
market_data['VIX_Pos_20d'] = (market_data['VIX'] - roll_min) / (roll_max - roll_min + 1e-8)

df_stocks = df_fresh[~df_fresh['Ticker'].str.upper().isin(['VIX', '^VIX', 'SPY'])].copy()
spy_df = market_data[['Date', 'SPY']].copy()
spy_df['SPY_Ret_5d'] = spy_df['SPY'].pct_change(5)

processed_dfs = []
for ticker, group in df_stocks.groupby('Ticker'):
    g = group.copy().sort_values('Date').reset_index(drop=True)
    if len(g) < 100: continue
    o, h, l, c, v = g['Open'], g['High'], g['Low'], g['Close'], g['Volume']
    prev_c = c.shift(1)
    candle_range = (h - l) + 1e-8
    
    g['Body_Ratio'] = (c - o) / candle_range
    g['Upper_Shadow'] = (h - np.maximum(o, c)) / candle_range
    g['Lower_Shadow'] = (np.minimum(o, c) - l) / candle_range
    g['Close_Pos'] = (c - l) / candle_range
    g['Gap_Pct'] = (o - prev_c) / (prev_c + 1e-8)
    
    for win in [5, 20, 60]:
        r_min = l.rolling(win).min()
        r_max = h.rolling(win).max()
        g[f'Pos_Box_{win}d'] = (c - r_min) / (r_max - r_min + 1e-8)
        g[f'Ret_{win}d'] = c.pct_change(win)
        
    g['Vol_Ratio_5d'] = v / (v.rolling(5).mean() + 1e-8)
    g['Vol_Ratio_20d'] = v / (v.rolling(20).mean() + 1e-8)
    g['Vol_Trend_5_20'] = v.rolling(5).mean() / (v.rolling(20).mean() + 1e-8)
    
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    g['ATR_Pct'] = atr14 / (c + 1e-8)
    g['Range_Expansion'] = candle_range / (atr14 + 1e-8)
    
    future_days = 10
    future_returns = []
    closes = c.values
    for i in range(len(g) - future_days):
        entry = closes[i]
        max_future_close = np.max(closes[i+1 : i+future_days+1])
        future_returns.append((max_future_close - entry) / entry * 100)
    future_returns.extend([np.nan] * future_days)
    g['Target_Return'] = future_returns
    processed_dfs.append(g)

df_all = pd.concat(processed_dfs, ignore_index=True)
df_all = pd.merge(df_all, market_data, on='Date', how='inner')
df_all = pd.merge(df_all, spy_df[['Date', 'SPY_Ret_5d']], on='Date', how='inner')
df_all['Rel_Strength_5d'] = df_all['Ret_5d'] - df_all['SPY_Ret_5d']

FEATURE_COLS = [
    'Body_Ratio', 'Upper_Shadow', 'Lower_Shadow', 'Close_Pos', 'Gap_Pct',
    'Pos_Box_5d', 'Ret_5d', 'Pos_Box_20d', 'Ret_20d', 'Pos_Box_60d', 'Ret_60d',
    'Vol_Ratio_5d', 'Vol_Ratio_20d', 'Vol_Trend_5_20',
    'ATR_Pct', 'Range_Expansion',
    'VIX', 'VIX_Change_1d', 'VIX_Change_5d', 'VIX_Pos_20d', 'Rel_Strength_5d'
]
df_clean = df_all.dropna(subset=FEATURE_COLS).reset_index(drop=True)

print("🚀 [2/4] 회귀 앙상블 모델 학습 중...")
df_trainable = df_clean.dropna(subset=['Target_Return']).reset_index(drop=True)

train_list, val_list = [], []
for _, group in df_trainable.groupby('Ticker'):
    split_idx = int(len(group) * 0.85)
    train_list.append(group.iloc[:split_idx])
    val_list.append(group.iloc[split_idx:])

df_train = pd.concat(train_list, ignore_index=True)
df_val = pd.concat(val_list, ignore_index=True)

X_train, y_train = df_train[FEATURE_COLS], df_train['Target_Return']
X_val, y_val = df_val[FEATURE_COLS], df_val['Target_Return']

lgb_reg = lgb.LGBMRegressor(
    objective='huber', n_estimators=400, learning_rate=0.03,
    max_depth=5, num_leaves=25, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
)
lgb_reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])

cat_reg = CatBoostRegressor(
    loss_function='Huber:delta=1.0', iterations=500, learning_rate=0.03,
    depth=5, l2_leaf_reg=5.0, random_seed=42, verbose=0
)
cat_reg.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)

print("🚀 [3/4] 추천 종목 및 대시보드 JSON 생성 중...")
latest_rows = df_clean.sort_values('Date').groupby('Ticker').last().reset_index()
latest_date = latest_rows['Date'].max().strftime('%Y-%m-%d')
current_vix = latest_rows['VIX'].iloc[0]

X_latest = latest_rows[FEATURE_COLS]
latest_rows['Expected_Return_Pct'] = (lgb_reg.predict(X_latest) * 0.5) + (cat_reg.predict(X_latest) * 0.5)

filtered_picks = latest_rows[latest_rows['ATR_Pct'] <= 0.10].sort_values(
    by='Expected_Return_Pct', ascending=False
).head(5).copy()

trading_plan = []
for _, row in filtered_picks.iterrows():
    ticker = row['Ticker']
    close = row['Close']
    atr_val = row['ATR_Pct'] * close
    exp_ret = row['Expected_Return_Pct']
    
    entry_price = close
    stop_loss = entry_price - (1.0 * atr_val)
    stop_pct = - (1.0 * row['ATR_Pct']) * 100
    take_profit = entry_price + (1.5 * atr_val)
    tp_pct = (1.5 * row['ATR_Pct']) * 100
    confidence = min(max(55.0 + (exp_ret * 2), 50.0), 85.0)
    
    trading_plan.append({
        "Ticker": ticker,
        "진입가 ($)": round(entry_price, 2),
        "손절가 ($)": round(stop_loss, 2),
        "손절폭 (%)": round(stop_pct, 1),
        "최종익절가 ($)": round(take_profit, 2),
        "익절폭 (%)": round(tp_pct, 1),
        "AI예상수익(%)": round(exp_ret, 1),
        "추정승률(%)": round(confidence, 1)
    })

dashboard_payload = {
    "last_date": latest_date,
    "market_vix": float(current_vix),
    "vix_warning": bool(current_vix > 24.0),
    "recommendations": trading_plan
}

with open("quant_dashboard_data.json", "w", encoding="utf-8") as f:
    json.dump(dashboard_payload, f, ensure_ascii=False, indent=4)

print("✅ [4/4] main.py 파이프라인 실행 및 JSON 저장 완료!")
