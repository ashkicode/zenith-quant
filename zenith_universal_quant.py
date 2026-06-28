# zenith_universal_quant.py
# Zenith Universal Quant Engine - Crypto, Forex, Stocks, Indices & DEX Tokens
# Features: Pure ATR Scaling, Dynamic Decimals, MTF, Geometric Patterns, Twelve Data + DEX Integration

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from scipy.stats import linregress
import math
import warnings
import requests
import time

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Zenith Universal", layout="wide", page_icon="🌐")

# -----------------------------------------------------------------------------
# 1. Universal Live Price & Data Engine (Waterfall: TwelveData -> yFinance -> DEX)
# -----------------------------------------------------------------------------
TWELVE_DATA_API_KEY = "25b5bc8b6be74ef487a77d1923ebe0ed"
HEADERS = {'Accept': 'application/json', 'User-Agent': 'ZenithQuant/1.0'}

def format_symbol_for_twelvedata(symbol):
    s = symbol.upper()
    if "-USD" in s:
        return s.replace("-USD", "/USD")
    elif "=X" in s:
        base = s.replace("=X", "")
        if len(base) == 6:
            return f"{base[:3]}/{base[3:]}"
        return base
    elif s == "GC=F":
        return "XAU/USD" 
    return s

def get_live_price(symbol):
    """دریافت قیمت زنده با ساختار آبشاری: TwelveData -> yFinance -> DexScreener"""
    td_symbol = format_symbol_for_twelvedata(symbol)
    url_twelve = f"https://api.twelvedata.com/price?symbol={td_symbol}&apikey={TWELVE_DATA_API_KEY}"
    
    # 1. Twelve Data (Main Markets)
    try:
        res = requests.get(url_twelve, timeout=3).json()
        if 'price' in res: return float(res['price'])
    except: pass

    # 2. yFinance (Fallback)
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.get('last_price')
        if price: return price
    except: pass

    # 3. DexScreener (Shitcoins & DeFi)
    try:
        clean_symbol = symbol.split('-')[0].split('/')[0].upper()
        url_dex = f"https://api.dexscreener.com/latest/dex/search?q={clean_symbol}"
        data = requests.get(url_dex, timeout=5).json()
        
        if data.get('pairs'):
            valid_pairs = [p for p in data['pairs'] if 'priceUsd' in p and 'liquidity' in p]
            if valid_pairs:
                valid_pairs.sort(key=lambda x: float(x.get('liquidity', {}).get('usd', 0)), reverse=True)
                return float(valid_pairs[0]['priceUsd'])
    except: pass
        
    return None

def fetch_dex_ohlcv(symbol, tf_type, aggregate, limit=500):
    """دریافت کندل‌های تاریخی از GeckoTerminal برای توکن‌های DEX"""
    try:
        clean_symbol = symbol.split('-')[0].split('/')[0].upper()
        search_url = f"https://api.geckoterminal.com/api/v2/search/pools?query={clean_symbol}"
        res = requests.get(search_url, headers=HEADERS, timeout=10).json()
        
        if not res.get('data'): return None
        
        # انتخاب اولین استخر (معمولاً پرحجم‌ترین است)
        pool = res['data'][0]
        network = pool['relationships']['network']['data']['id']
        pool_address = pool['attributes']['address']
        
        ohlcv_url = f"https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}/ohlcv/{tf_type}?aggregate={aggregate}&limit={limit}"
        ohlcv_res = requests.get(ohlcv_url, headers=HEADERS, timeout=10).json()
        
        if not ohlcv_res.get('data') or not ohlcv_res['data'].get('attributes'): return None
        
        raw_data = ohlcv_res['data']['attributes']['ohlcv_list']
        df = pd.DataFrame(raw_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        df = df.sort_index()
        return df
    except Exception as e:
        return None

@st.cache_data(ttl=30)
def fetch_mtf_data(symbol, mode):
    df_base, df_htf = None, None
    
    # تنظیم تایم فریم‌ها برای yfinance
    if "Scalp" in mode:
        yf_base_tf, yf_base_per, yf_htf, yf_htf_per = "5m", "3d", "30m", "1mo"
        dex_base_tf, dex_base_agg, dex_htf_tf, dex_htf_agg = "minute", 5, "minute", 30
    else: 
        yf_base_tf, yf_base_per, yf_htf, yf_htf_per = "1h", "1mo", "1d", "1y"
        dex_base_tf, dex_base_agg, dex_htf_tf, dex_htf_agg = "hour", 1, "day", 1
        
    # تلاش اول: yFinance
    try:
        df_base_temp = yf.download(symbol, interval=yf_base_tf, period=yf_base_per, progress=False)
        df_htf_temp = yf.download(symbol, interval=yf_htf, period=yf_htf_per, progress=False)
        
        if not df_base_temp.empty and not df_htf_temp.empty:
            if isinstance(df_base_temp.columns, pd.MultiIndex):
                df_base_temp.columns = df_base_temp.columns.get_level_values(0)
                df_htf_temp.columns = df_htf_temp.columns.get_level_values(0)
            df_base_temp.dropna(subset=['Close'], inplace=True)
            df_htf_temp.dropna(subset=['Close'], inplace=True)
            
            if len(df_base_temp) > 20: 
                df_base, df_htf = df_base_temp, df_htf_temp
    except: pass
    
    # تلاش دوم: اگر yFinance دیتایی نداشت، برو سراغ DEX (GeckoTerminal)
    if df_base is None or df_base.empty:
        df_base = fetch_dex_ohlcv(symbol, dex_base_tf, dex_base_agg)
        time.sleep(0.5) # جلوگیری از بلاک شدن توسط API
        df_htf = fetch_dex_ohlcv(symbol, dex_htf_tf, dex_htf_agg)
        
    return df_base, df_htf

# -----------------------------------------------------------------------------
# 2. Geometry Engine (S&R & Trendlines)
# -----------------------------------------------------------------------------
def detect_pivots(df, window=5):
    df['Pivot_High'] = df['High'][(df['High'].shift(window) < df['High']) & (df['High'].shift(-window) < df['High'])]
    df['Pivot_Low'] = df['Low'][(df['Low'].shift(window) > df['Low']) & (df['Low'].shift(-window) > df['Low'])]
    return df

def calculate_smart_sr(df, atr_val):
    highs = df['Pivot_High'].dropna().values
    lows = df['Pivot_Low'].dropna().values
    pivots = np.concatenate([highs, lows])
    pivots.sort()
    
    if len(pivots) < 2: return [], []
    
    zones = []
    current_zone = [pivots[0]]
    for p in pivots[1:]:
        if p - current_zone[-1] <= atr_val * 0.5:
            current_zone.append(p)
        else:
            if len(current_zone) >= 2:
                zones.append((min(current_zone), max(current_zone)))
            current_zone = [p]
            
    current_price = df['Close'].iloc[-1]
    resistances = sorted([z for z in zones if z[0] > current_price], key=lambda x: x[0])[:2]
    supports = sorted([z for z in zones if z[1] < current_price], key=lambda x: x[1], reverse=True)[:2]
    return supports, resistances

def calculate_geometric_trendlines(df, atr_val, lookback=120):
    recent_df = df.iloc[-lookback:].copy()
    highs = recent_df[['Pivot_High']].dropna()
    lows = recent_df[['Pivot_Low']].dropna()
    
    res_line, sup_line = None, None
    
    def find_touches(m, b, is_high=True):
        touches = []
        for idx in range(len(recent_df)):
            x = idx
            y = recent_df['High'].iloc[idx] if is_high else recent_df['Low'].iloc[idx]
            expected_y = m * x + b
            dist = abs(m * x - y + b) / math.sqrt(m**2 + 1)
            if dist < atr_val * 0.3:
                if (is_high and recent_df['High'].iloc[idx] >= expected_y - (atr_val*0.2)) or \
                   (not is_high and recent_df['Low'].iloc[idx] <= expected_y + (atr_val*0.2)):
                    touches.append(recent_df.index[idx])
        return touches

    if len(highs) >= 3:
        x = np.arange(len(highs))
        y = highs['Pivot_High'].values
        slope, intercept, r, _, _ = linregress(x, y)
        if r < -0.3:
            start_x_offset = recent_df.index.get_loc(highs.index[0])
            mapped_intercept = intercept - (slope * start_x_offset)
            res_touches = find_touches(slope, mapped_intercept, is_high=True)
            res_line = {"slope": slope, "intercept": mapped_intercept, "touches": res_touches}
            
    if len(lows) >= 3:
        x = np.arange(len(lows))
        y = lows['Pivot_Low'].values
        slope, intercept, r, _, _ = linregress(x, y)
        if r > 0.3:
            start_x_offset = recent_df.index.get_loc(lows.index[0])
            mapped_intercept = intercept - (slope * start_x_offset)
            sup_touches = find_touches(slope, mapped_intercept, is_high=False)
            sup_line = {"slope": slope, "intercept": mapped_intercept, "touches": sup_touches}
            
    return sup_line, res_line

# -----------------------------------------------------------------------------
# 3. Pattern & Wave Engine
# -----------------------------------------------------------------------------
def detect_candlestick_patterns(df):
    df['Pattern'] = None
    df['Pattern_Type'] = 0 
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    for i in range(3, len(df)):
        c0, c1, c2 = df.iloc[i], df.iloc[i-1], df.iloc[i-2]
        atr = df['ATR'].iloc[i]
        if pd.isna(atr): continue
        
        b0 = abs(c0['Close'] - c0['Open'])
        b1 = abs(c1['Close'] - c1['Open'])
        r0 = c0['High'] - c0['Low']
        uw0 = c0['High'] - max(c0['Close'], c0['Open'])
        lw0 = min(c0['Close'], c0['Open']) - c0['Low']
        
        if r0 == 0: continue

        if lw0 > b0 * 2 and uw0 < b0 * 0.2:
            df.loc[df.index[i], 'Pattern'] = 'HAMMER 🔨'; df.loc[df.index[i], 'Pattern_Type'] = 1
        elif uw0 > b0 * 2 and lw0 < b0 * 0.2:
            df.loc[df.index[i], 'Pattern'] = 'SHOOTING STAR ☄️'; df.loc[df.index[i], 'Pattern_Type'] = -1
        elif c1['Close'] < c1['Open'] and c0['Close'] > c0['Open'] and c0['Close'] > c1['Open'] and c0['Open'] < c1['Close']:
            df.loc[df.index[i], 'Pattern'] = 'BULL_ENGULFING 🟢'; df.loc[df.index[i], 'Pattern_Type'] = 1
        elif c1['Close'] > c1['Open'] and c0['Close'] < c0['Open'] and c0['Close'] < c1['Open'] and c0['Open'] > c1['Close']:
            df.loc[df.index[i], 'Pattern'] = 'BEAR_ENGULFING 🔴'; df.loc[df.index[i], 'Pattern_Type'] = -1
        elif c2['Close'] < c2['Open'] and b1 < atr*0.3 and c0['Close'] > c0['Open'] and c0['Close'] > (c2['Open']+c2['Close'])/2:
            df.loc[df.index[i], 'Pattern'] = 'MORNING_STAR 🌟'; df.loc[df.index[i], 'Pattern_Type'] = 1
        elif c2['Close'] > c2['Open'] and b1 < atr*0.3 and c0['Close'] < c0['Open'] and c0['Close'] < (c2['Open']+c2['Close'])/2:
            df.loc[df.index[i], 'Pattern'] = 'EVENING_STAR 🌙'; df.loc[df.index[i], 'Pattern_Type'] = -1

    return df

def detect_advanced_chart_patterns(df, atr_val):
    highs = df[df['Pivot_High'].notna()]['Pivot_High']
    lows = df[df['Pivot_Low'].notna()]['Pivot_Low']
    
    detected_patterns = []
    if len(highs) < 3 or len(lows) < 3: return detected_patterns

    h_vals = highs.values[-3:]
    l_vals = lows.values[-3:]
    buffer = atr_val * 0.4

    if abs(h_vals[-1] - h_vals[-2]) < buffer:
        if len(h_vals) == 3 and abs(h_vals[-2] - h_vals[-3]) < buffer:
            detected_patterns.append("TRIPLE TOP 🏔️🏔️🏔️ (Bearish)")
        else:
            detected_patterns.append("DOUBLE TOP 🏔️🏔️ (Bearish)")

    if abs(l_vals[-1] - l_vals[-2]) < buffer:
        if len(l_vals) == 3 and abs(l_vals[-2] - l_vals[-3]) < buffer:
            detected_patterns.append("TRIPLE BOTTOM 🕳️🕳️🕳️ (Bullish)")
        else:
            detected_patterns.append("DOUBLE BOTTOM 🕳️🕳️ (Bullish)")

    if len(h_vals) >= 3 and len(l_vals) >= 2:
        ls, h, rs = h_vals[-3], h_vals[-2], h_vals[-1]
        if h > ls and h > rs and abs(ls - rs) < atr_val:
            if abs(l_vals[-2] - l_vals[-1]) < atr_val: 
                detected_patterns.append("HEAD & SHOULDERS 👤 (Bearish)")
                
    if len(l_vals) >= 3 and len(h_vals) >= 2:
        ls, h_inv, rs = l_vals[-3], l_vals[-2], l_vals[-1]
        if h_inv < ls and h_inv < rs and abs(ls - rs) < atr_val:
            detected_patterns.append("INVERTED H&S 👤 (Bullish)")

    if len(h_vals) >= 2 and len(l_vals) >= 2:
        if abs(h_vals[-1] - h_vals[-2]) < buffer and l_vals[-1] > l_vals[-2] + buffer:
            detected_patterns.append("ASCENDING TRIANGLE 🔺 (Bullish)")
        elif abs(l_vals[-1] - l_vals[-2]) < buffer and h_vals[-1] < h_vals[-2] - buffer:
            detected_patterns.append("DESCENDING TRIANGLE 🔻 (Bearish)")

    if len(h_vals) == 3 and len(l_vals) >= 2:
        if h_vals[2] > h_vals[1] > h_vals[0] and l_vals[-1] > l_vals[-2]:
            if l_vals[-1] > h_vals[0]: 
                detected_patterns.append("ELLIOTT 5-WAVE UP 🌊 (Impulse)")

    if len(l_vals) == 3 and len(h_vals) >= 2:
        if l_vals[2] < l_vals[1] < l_vals[0] and h_vals[-1] < h_vals[-2]:
            if h_vals[-1] < l_vals[0]:
                detected_patterns.append("ELLIOTT 5-WAVE DOWN 🌊 (Impulse)")

    return detected_patterns

# -----------------------------------------------------------------------------
# 4. Universal Signal Generator
# -----------------------------------------------------------------------------
def format_price(price):
    if price > 1000:
        return f"{price:.2f}"
    elif price > 1:
        return f"{price:.4f}"
    elif price > 0.001:
        return f"{price:.5f}"
    else:
        return f"{price:.8f}"

def generate_multi_signals(df_base, df_htf, symbol, mode, live_price, supports, resistances, sup_line, res_line):
    df_htf['EMA_50'] = ta.ema(df_htf['Close'], length=50)
    
    if len(df_htf) > 50 and pd.notna(df_htf['EMA_50'].iloc[-1]):
        htf_trend = 1 if df_htf['Close'].iloc[-1] > df_htf['EMA_50'].iloc[-1] else -1
    else:
        htf_trend = 0
        
    atr = df_base['ATR'].iloc[-1] if pd.notna(df_base['ATR'].iloc[-1]) else (live_price * 0.002)
    buffer = atr * 0.6 
    
    asset_name = symbol.split('-')[0].split('/')[0].replace("=X", "").upper()

    recent_df = df_base.iloc[-3:]
    recent_types = recent_df['Pattern_Type'].tolist()
    
    chart_patterns = detect_advanced_chart_patterns(df_base, atr)
    signals = []

    near_support = any([z[0] - buffer <= live_price <= z[1] + buffer for z in supports])
    near_res = any([z[0] - buffer <= live_price <= z[1] + buffer for z in resistances])
    
    tl_bull_bounce, tl_bear_bounce = False, False
    if sup_line and any(t in recent_df.index for t in sup_line['touches']): tl_bull_bounce = True
    if res_line and any(t in recent_df.index for t in res_line['touches']): tl_bear_bounce = True

    bull_score = 3 if near_support else 0
    bull_score += 3 if tl_bull_bounce else 0
    bull_score += 2 if 1 in recent_types else 0
    bull_score += 1.5 if htf_trend == 1 else 0

    bear_score = 3 if near_res else 0
    bear_score += 3 if tl_bear_bounce else 0
    bear_score += 2 if -1 in recent_types else 0
    bear_score += 1.5 if htf_trend == -1 else 0
    
    for cp in chart_patterns:
        if "Bullish" in cp or "UP" in cp: bull_score += 4
        elif "Bearish" in cp or "DOWN" in cp: bear_score += 4

    sl_dist = atr * 1.5 if "Scalp" in mode else atr * 2.5

    if bull_score >= 5.0:
        signals.append(build_telegram_msg(asset_name, "BUY", live_price, sl_dist, "Instant Market Exec"))
    elif bear_score >= 5.0:
        signals.append(build_telegram_msg(asset_name, "SELL", live_price, sl_dist, "Instant Market Exec"))

    if supports:
        best_sup = supports[0][1] 
        if live_price > best_sup + (atr * 0.5):
            signals.append(build_telegram_msg(asset_name, "BUY LIMIT", best_sup, sl_dist, "Pending Pullback"))

    if resistances:
        best_res = resistances[0][0] 
        if live_price < best_res - (atr * 0.5):
            signals.append(build_telegram_msg(asset_name, "SELL LIMIT", best_res, sl_dist, "Pending Fade"))

    return signals, chart_patterns

def build_telegram_msg(asset, action, entry_price, sl_dist, signal_type):
    sl = entry_price - sl_dist if "BUY" in action else entry_price + sl_dist
    tp1 = entry_price + (sl_dist * 1.2) if "BUY" in action else entry_price - (sl_dist * 1.2)
    tp2 = entry_price + (sl_dist * 2.5) if "BUY" in action else entry_price - (sl_dist * 2.5)

    e_s = entry_price - (sl_dist*0.2) if "BUY" in action else entry_price
    e_e = entry_price if "BUY" in action else entry_price + (sl_dist*0.2)

    msg = f"""**[{signal_type}]**
{asset} {action}

{format_price(e_s)}_{format_price(e_e)}

Tp{format_price(tp1)}
Tp{format_price(tp2)}

Sl{format_price(sl)}"""
    return msg

# -----------------------------------------------------------------------------
# 5. UI & Advanced Chart Rendering
# -----------------------------------------------------------------------------
st.title("🌐 Zenith Universal: Any Asset, Any Market")
st.markdown("از این بخش می‌توانید هر نمادی (کریپتو، فارکس، سهام و حتی شت‌کوین‌های DEX) را وارد کنید.")

c1, c2, c3 = st.columns([1.5, 1, 1])
symbol_input = c1.text_input("Enter Asset Ticker (e.g. BTC-USD, AAPL, EURUSD=X, RAVE)", value="BTC-USD").upper()
mode = c2.radio("Trading Mode", ["⚡ Scalp (5m / 30m)", "🌊 Swing (1h / 1d)"])

live_price = get_live_price(symbol_input)
if live_price:
    c3.success(f"### 🟢 LIVE PRICE:\n**${format_price(live_price)}**")
else:
    c3.warning("Waiting for live data...")

if st.button("🚀 Analyze & Generate Signals", type="primary"):
    if not symbol_input:
        st.error("Please enter a valid ticker symbol.")
    else:
        with st.spinner(f"Scanning the Matrix for {symbol_input} (Checking CEX & DEX)..."):
            df_base, df_htf = fetch_mtf_data(symbol=symbol_input, mode=mode)
            
            if df_base is not None and not df_base.empty and len(df_base) > 20 and df_htf is not None and not df_htf.empty:
                df_base = detect_pivots(df_base, window=6)
                df_base = detect_candlestick_patterns(df_base)
                
                if 'ATR' not in df_base.columns or df_base['ATR'].isna().all():
                    st.error("❌ دیتای کافی برای محاسبه نوسانات (ATR) دریافت نشد.")
                else:
                    atr_val = df_base['ATR'].iloc[-1]
                    supports, resistances = calculate_smart_sr(df_base, atr_val)
                    sup_line, res_line = calculate_geometric_trendlines(df_base, atr_val, lookback=120)
                    
                    eval_price = live_price if live_price else df_base['Close'].iloc[-1]
                    signals, chart_patterns = generate_multi_signals(df_base, df_htf, symbol_input, mode, eval_price, supports, resistances, sup_line, res_line)
                    
                    st.markdown("---")
                    col_sig, col_chart = st.columns([1, 2.5])
                    
                    with col_sig:
                        st.subheader("📲 Actionable Signals")
                        if not signals:
                            st.warning("No high-probability zones identified. Waiting for setup.")
                        else:
                            for sig_msg in signals:
                                if "Instant" in sig_msg:
                                    st.success(sig_msg)
                                else:
                                    st.info(sig_msg)
                                    
                        if chart_patterns:
                            st.markdown("### 👁️ Detected Patterns")
                            for p in chart_patterns:
                                st.success(p)

                    with col_chart:
                        st.subheader("📊 TradingView-Grade Analysis")
                        recent_df = df_base.iloc[-120:].copy()
                        
                        x_vals_int = np.arange(len(recent_df))
                        
                        fig = go.Figure(data=[go.Candlestick(
                            x=x_vals_int, open=recent_df['Open'], high=recent_df['High'],
                            low=recent_df['Low'], close=recent_df['Close'], name=symbol_input,
                            increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
                        )])
                        
                        for z in supports:
                            fig.add_shape(type="rect", x0=0, y0=z[0], x1=len(recent_df)+5, y1=z[1],
                                          line=dict(width=0), fillcolor="rgba(38, 166, 154, 0.15)", layer="below")
                        for z in resistances:
                            fig.add_shape(type="rect", x0=0, y0=z[0], x1=len(recent_df)+5, y1=z[1],
                                          line=dict(width=0), fillcolor="rgba(239, 83, 80, 0.15)", layer="below")

                        if res_line:
                            y_tl = res_line['slope'] * x_vals_int + res_line['intercept']
                            fig.add_trace(go.Scatter(x=x_vals_int, y=y_tl, mode='lines', line=dict(color='orange'), name='Resist Line'))
                            touch_x = [recent_df.index.get_loc(idx) for idx in res_line['touches'] if idx in recent_df.index]
                            touch_y = [recent_df['High'].loc[idx] for idx in res_line['touches'] if idx in recent_df.index]
                            if touch_x: fig.add_trace(go.Scatter(x=touch_x, y=touch_y, mode='markers', marker=dict(symbol='star', size=12, color='yellow'), name='Bounce'))

                        if sup_line:
                            y_tl = sup_line['slope'] * x_vals_int + sup_line['intercept']
                            fig.add_trace(go.Scatter(x=x_vals_int, y=y_tl, mode='lines', line=dict(color='cyan'), name='Support Line'))
                            touch_x = [recent_df.index.get_loc(idx) for idx in sup_line['touches'] if idx in recent_df.index]
                            touch_y = [recent_df['Low'].loc[idx] for idx in sup_line['touches'] if idx in recent_df.index]
                            if touch_x: fig.add_trace(go.Scatter(x=touch_x, y=touch_y, mode='markers', marker=dict(symbol='star', size=12, color='yellow'), name='Bounce'))

                        pattern_df = recent_df.dropna(subset=['Pattern'])
                        for idx, row in pattern_df.iterrows():
                            x_pos = recent_df.index.get_loc(idx)
                            fig.add_annotation(x=x_pos, y=row['High'] if 'BEAR' in row['Pattern'] or 'STAR ☄️' in row['Pattern'] else row['Low'],
                                               text=row['Pattern'].split(' ')[0], showarrow=False, font=dict(size=10, color="white"),
                                               yshift=-18 if 'BULL' in row['Pattern'] or 'HAMMER' in row['Pattern'] else 18)

                        if eval_price:
                            fig.add_hline(y=eval_price, line_dash="dot", line_color="yellow", annotation_text="LIVE", annotation_position="right")

                        fig.update_layout(
                            template="plotly_dark", 
                            height=700, 
                            xaxis_rangeslider_visible=False,
                            dragmode='pan',
                            hovermode='x unified',
                            margin=dict(l=10, r=50, t=30, b=10),
                            plot_bgcolor='#131722',
                            paper_bgcolor='#131722',
                            xaxis=dict(showgrid=True, gridcolor='#363c4e', tickmode='array', 
                                       tickvals=np.arange(0, len(recent_df), 20), 
                                       ticktext=recent_df.index[::20].strftime('%b %d %H:%M')),
                            yaxis=dict(showgrid=True, gridcolor='#363c4e', side='right')
                        )
                        
                        st.plotly_chart(fig, use_container_width=True, config={
                            'scrollZoom': True,
                            'displayModeBar': True,
                            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                            'displaylogo': False
                        })
            else:
                st.warning(f"⚠️ دیتایی برای نماد '{symbol_input}' دریافت نشد. مطمئن شوید که تیکر (Ticker) را صحیح وارد کرده‌اید یا استخر نقدینگی آن وجود دارد.")
