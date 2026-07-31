#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf

from hk_universe import fetch_hk_universe, hk_yahoo_symbol
UA = "Mozilla/5.0 (X11; Linux x86_64) Hermes-Agent/1.0"
SWING_WINDOW = 3
STAGE1_LIQUIDITY_LOOKBACK_DAYS = 20
STAGE1_LIQUIDITY_MIN_AVG_DOLLAR_VOLUME_HKD = 20_000_000
DEFAULT_STAGE1_PERIOD = '1mo'
SHORT_TREND_LOOKBACK = 30
LONG_TREND_LOOKBACK = 90
LONG_TERM_TREND_BONUS = 5
MIN_DOUBLE_STRUCTURE_GAP = 20
DOUBLE_STRUCTURE_WIDE_GAP_BONUS = 5
DOUBLE_STRUCTURE_WIDE_GAP_THRESHOLD = 60


def _hard_timeout_handler(signum, frame):
    raise TimeoutError('hard timeout waiting for yfinance download')


def run_with_hard_timeout(seconds, fn):
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _hard_timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": UA})
    with urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_nasdaq_listed(text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(text), sep="|")
    df = df[df["Symbol"].notna()]
    df = df[df["Symbol"] != "File Creation Time"]
    df["source"] = "nasdaq"
    df["name"] = df["Security Name"].fillna("")
    df["etf"] = df["ETF"].fillna("N").astype(str).str.upper().eq("Y")
    df["test_issue"] = df["Test Issue"].fillna("N").astype(str).str.upper().eq("Y")
    return df[["Symbol", "name", "etf", "test_issue", "source"]]


def parse_other_listed(text: str) -> pd.DataFrame:
    df = pd.read_csv(StringIO(text), sep="|")
    df = df[df["ACT Symbol"].notna()]
    df = df[df["ACT Symbol"] != "File Creation Time"]
    df["source"] = "other"
    df["name"] = df["Security Name"].fillna("")
    df["etf"] = df["ETF"].fillna("N").astype(str).str.upper().eq("Y")
    df["test_issue"] = df["Test Issue"].fillna("N").astype(str).str.upper().eq("Y")
    return df.rename(columns={"ACT Symbol": "Symbol"})[["Symbol", "name", "etf", "test_issue", "source"]]


BAD_PATTERNS = [
    " warrant", " warrants", " right", " rights", " unit", " units", " preferred", " depositary", " depository",
    " adr", " ads", " note", " notes", " bond", " etn", " nextshares", " when issued", " due ", " rate ",
    " income cap", " preferred stock", " preference", " senior note", " trust preferred"
]
GOOD_STOCK_PATTERNS = [
    " common stock", " common shares", " ordinary shares", " common share", " ordinary share", " class a common", " class b common"
]
DEFAULT_BAD_SYMBOLS_FILE = Path(__file__).resolve().parent / 'data' / 'universe' / 'yahoo_bad_symbols.txt'
BAD_SYMBOL_SUFFIXES = (
    '-V', '.V',
    '-WI', '.WI',
    '-WD', '.WD',
    '-WS', '.WS',
    '-W', '.W',
    '-U', '.U',
    '-R', '.R',
    '-RT', '.RT',
    '-P', '.P',
)
BAD_SYMBOL_SUBSTRINGS = ('^', '/', '=')


def load_known_bad_symbols(path: Path | None = None) -> set[str]:
    target = Path(path) if path else DEFAULT_BAD_SYMBOLS_FILE
    if not target.exists():
        return set()
    out = set()
    for raw in target.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        upper = line.upper()
        out.add(upper)
        out.add(upper.replace('.', '-'))
    return out


KNOWN_BAD_SYMBOLS = load_known_bad_symbols()


def is_probably_yahoo_friendly_symbol(symbol: str) -> bool:
    sym = str(symbol or '').upper().strip()
    if not sym:
        return False
    yahoo_sym = yahoo_symbol(sym)
    if sym in KNOWN_BAD_SYMBOLS or yahoo_sym in KNOWN_BAD_SYMBOLS:
        return False
    if any(ch in sym for ch in ["$", "+", "*"]):
        return False
    if any(token in sym for token in BAD_SYMBOL_SUBSTRINGS):
        return False
    if any(sym.endswith(suffix) for suffix in BAD_SYMBOL_SUFFIXES):
        return False
    if any(yahoo_sym.endswith(suffix.replace('.', '-')) for suffix in BAD_SYMBOL_SUFFIXES):
        return False
    return True


def is_regular_security(symbol: str, name: str, is_etf: bool, test_issue: bool) -> bool:
    if not symbol or test_issue:
        return False
    sym = symbol.upper()
    if not is_probably_yahoo_friendly_symbol(sym):
        return False
    lname = f" {str(name).lower()} "
    if is_etf:
        if any(x in lname for x in ["etn", "exchange traded note", "nextshares", "trust preferred"]):
            return False
        return True
    if any(p in lname for p in BAD_PATTERNS):
        return False
    if " fund" in lname or " trust" in lname or " acquisition" in lname or " acquisition corp" in lname:
        return False
    if any(p in lname for p in GOOD_STOCK_PATTERNS):
        return True
    if any(token in lname for token in [" class a", " class b", " class c", " ordinary", " common"]):
        return True
    return False



def yahoo_symbol(sym: str) -> str:
    raw = str(sym or '').strip().upper()
    if not raw:
        return ''
    if raw.endswith('.HK'):
        return raw
    digits = ''.join(ch for ch in raw if ch.isdigit())
    if digits:
        return hk_yahoo_symbol(digits)
    return raw.replace('.', '-')


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def split_into_shards(seq, shard_count):
    if shard_count <= 1 or len(seq) <= 1:
        return [list(seq)] if seq else []
    shard_count = max(1, min(int(shard_count), len(seq)))
    base = len(seq) // shard_count
    extra = len(seq) % shard_count
    out = []
    start = 0
    for idx in range(shard_count):
        size = base + (1 if idx < extra else 0)
        end = start + size
        if start < len(seq):
            out.append(list(seq[start:end]))
        start = end
    return out


def append_log(stderr_path: str, message: str):
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with open(stderr_path, 'a', encoding='utf-8') as f:
        f.write(f"[{ts}] {message}\n")


def download_bars(symbols, period, stderr_path, batch=200, phase='DOWNLOAD'):
    frames = []
    misses = set()
    total_batches = max(1, math.ceil(len(symbols) / batch)) if symbols else 0
    for group_idx, group in enumerate(chunked(symbols, batch), start=1):
        batch_start = time.time()
        append_log(
            stderr_path,
            f"{phase}_BATCH_START period={period} batch={group_idx}/{total_batches} size={len(group)} accumulated_ok={len(frames)} accumulated_miss={len(misses)}"
        )
        tickers = ' '.join(group)
        time.sleep(0.35 + random.uniform(0.0, 0.55))
        data = None
        last_error = None
        for attempt in range(1, 4):
            try:
                data = run_with_hard_timeout(
                    45,
                    lambda: yf.download(
                        tickers=tickers,
                        period=period,
                        interval='1d',
                        auto_adjust=False,
                        group_by='ticker',
                        progress=False,
                        threads=False,
                        prepost=False,
                        timeout=30,
                    )
                )
                if data is not None and len(data) != 0:
                    break
                last_error = RuntimeError('empty download result')
            except Exception as e:
                last_error = e
            wait_s = 0.8 * attempt + random.uniform(0.6, 1.8)
            append_log(
                stderr_path,
                f"{phase}_RETRY period={period} batch={group_idx}/{total_batches} attempt={attempt} size={len(group)} wait={wait_s:.2f}s error={last_error}"
            )
            if attempt < 3:
                time.sleep(wait_s)
        if data is None or len(data) == 0:
            append_log(
                stderr_path,
                f"{phase}_ERROR period={period} batch={group_idx}/{total_batches} sample={group[:5]} error={last_error}"
            )
            misses.update(group)
            continue
        before_frames = len(frames)
        before_misses = len(misses)
        if isinstance(data.columns, pd.MultiIndex):
            if data.columns.nlevels == 2:
                if data.columns[0][0] in ["Adj Close", "Close", "High", "Low", "Open", "Volume"]:
                    # single ticker shape from yfinance sometimes
                    if len(group) == 1:
                        sym = group[0]
                        df = data.copy()
                        df.columns = [c[0] for c in df.columns]
                        df = df.reset_index().rename(columns={df.index.name or 'Date': 'Date'})
                        frames.append((sym, df))
                    else:
                        # unexpected; try extract by top-level names if possible
                        for sym in group:
                            try:
                                sdf = data[sym].reset_index()
                                frames.append((sym, sdf))
                            except Exception:
                                misses.add(sym)
                    continue
                for sym in group:
                    try:
                        sdf = data[sym].copy().reset_index()
                        if len(sdf.dropna(how='all')) == 0:
                            misses.add(sym)
                        else:
                            frames.append((sym, sdf))
                    except Exception:
                        misses.add(sym)
            else:
                misses.update(group)
        else:
            if len(group) == 1:
                sdf = data.reset_index()
                frames.append((group[0], sdf))
            else:
                misses.update(group)
        batch_ok = len(frames) - before_frames
        batch_miss = len(misses) - before_misses
        elapsed = time.time() - batch_start
        append_log(
            stderr_path,
            f"{phase}_BATCH_DONE period={period} batch={group_idx}/{total_batches} size={len(group)} ok={batch_ok} miss={batch_miss} cumulative_ok={len(frames)} cumulative_miss={len(misses)} elapsed={elapsed:.2f}s"
        )
        time.sleep(0.15)
    out = {}
    for sym, df in frames:
        cols = {c.lower(): c for c in df.columns}
        needed = [cols.get('date'), cols.get('open'), cols.get('high'), cols.get('low'), cols.get('close'), cols.get('volume')]
        if any(c is None for c in needed):
            misses.add(sym)
            continue
        sdf = df[[cols['date'], cols['open'], cols['high'], cols['low'], cols['close'], cols['volume']]].copy()
        sdf.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        sdf = sdf.dropna(subset=['Date']).sort_values('Date')
        if len(sdf) == 0 or sdf[['Open','High','Low','Close']].dropna(how='all').empty:
            misses.add(sym)
            continue
        sdf['Date'] = pd.to_datetime(sdf['Date']).dt.tz_localize(None)
        out[sym] = sdf.reset_index(drop=True)
    return out, misses


def local_extrema(df: pd.DataFrame, kind: str, lookback=90, window=SWING_WINDOW):
    sdf = df.tail(lookback).reset_index(drop=True)
    arr = sdf['Low'].to_numpy() if kind == 'low' else sdf['High'].to_numpy()
    idxs = []
    for i in range(window, len(sdf)-window):
        segment = arr[i-window:i+window+1]
        if kind == 'low':
            if arr[i] == np.nanmin(segment):
                if int(np.argmin(segment)) == window:
                    idxs.append(i)
        else:
            if arr[i] == np.nanmax(segment):
                if int(np.argmax(segment)) == window:
                    idxs.append(i)
    return sdf, idxs[-10:]


def avg_body(df):
    s = (df['Close'] - df['Open']).abs()
    return float(s.mean()) if len(s) else 0.0


def avg_tr(df):
    prev_close = df['Close'].shift(1)
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - prev_close).abs(),
        (df['Low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.mean()) if len(tr.dropna()) else 0.0


def score_confirm_day(df, idx, bullish=True):
    if idx <= 0 or idx >= len(df):
        return 0.0
    row = df.iloc[idx]
    trailing = df.iloc[max(0, idx-20):idx]
    body = abs(row['Close'] - row['Open'])
    avg20_body = avg_body(trailing)
    avg20_vol = float(trailing['Volume'].mean()) if len(trailing) else 0.0
    score = 0.0
    if bullish and row['Close'] > row['Open']:
        score += 12
    if (not bullish) and row['Close'] < row['Open']:
        score += 12
    if avg20_body > 0 and body > avg20_body:
        score += 8
    if avg20_vol > 0 and row['Volume'] > avg20_vol:
        score += 8
    return score


def reference_close_n_trading_days_ago(df, idx, days=20):
    ref_idx = idx - days
    if idx < 0 or idx >= len(df) or ref_idx < 0:
        return None
    return float(df.iloc[ref_idx]['Close'])


def passes_direction_filter_on_idx(df, idx, bullish=True, days=20, min_pct=0.0):
    ref_close = reference_close_n_trading_days_ago(df, idx, days=days)
    if ref_close is None:
        return False
    current_close = float(df.iloc[idx]['Close'])
    pct = (current_close / ref_close - 1.0) * 100.0
    if bullish:
        return pct >= min_pct
    return pct <= -min_pct


def trailing_avg_dollar_volume(df, idx, days=5):
    if idx < 0 or idx >= len(df):
        return None
    left = max(0, idx - days + 1)
    seg = df.iloc[left:idx+1].dropna(subset=['Close', 'Volume'])
    if len(seg) == 0:
        return None
    dv = seg['Close'].astype(float) * seg['Volume'].astype(float)
    if len(dv) == 0:
        return None
    return float(dv.mean())


def passes_stage1_liquidity(df, idx=None, *, days=STAGE1_LIQUIDITY_LOOKBACK_DAYS, min_avg_dollar_volume=STAGE1_LIQUIDITY_MIN_AVG_DOLLAR_VOLUME_HKD):
    if idx is None:
        idx = len(df) - 1
    avg_dollar_volume = trailing_avg_dollar_volume(df, idx, days=days)
    if avg_dollar_volume is None or not np.isfinite(avg_dollar_volume):
        return False, avg_dollar_volume
    return avg_dollar_volume >= float(min_avg_dollar_volume), float(avg_dollar_volume)


def liquidity_band_from_avg_dollar_volume(avg_dollar_volume):
    if avg_dollar_volume is None or not np.isfinite(avg_dollar_volume):
        return None
    if avg_dollar_volume >= 50_000_000:
        return '50m_plus'
    if avg_dollar_volume >= 20_000_000:
        return '20m_to_50m'
    return None


def filter_recent_windows_by_direction(df, windows, bullish=True, days=20, min_pct=1.0):
    filtered = []
    if not windows:
        return filtered
    date_to_idx = {
        pd.Timestamp(row['Date']).strftime('%Y-%m-%d'): i
        for i, row in df.iterrows()
    }
    for w in windows:
        rep_date = w.get('representative_date')
        if not rep_date:
            continue
        idx = date_to_idx.get(str(rep_date))
        if idx is None:
            continue
        avg_dollar_volume = trailing_avg_dollar_volume(df, idx, days=20)
        liquidity_band = liquidity_band_from_avg_dollar_volume(avg_dollar_volume)
        if not liquidity_band:
            continue
        if passes_direction_filter_on_idx(df, idx, bullish=bullish, days=days, min_pct=min_pct):
            new_w = dict(w)
            avg_dollar_volume_val = float(avg_dollar_volume if avg_dollar_volume is not None else 0.0)
            new_w['avg_20d_dollar_volume'] = round(avg_dollar_volume_val, 2)
            new_w['liquidity_band'] = liquidity_band
            filtered.append(new_w)
    return filtered


def date_to_index(df, date_str):
    matches = df.index[df['Date'].dt.strftime('%Y-%m-%d') == str(date_str)].tolist()
    return matches[0] if matches else None


def find_platform_zone(series, around_idx, direction='long'):
    left = max(0, around_idx - 10)
    right = min(len(series), around_idx + 1)
    seg = series.iloc[left:right]
    if len(seg) == 0:
        return None
    return float(seg.quantile(0.35)), float(seg.quantile(0.65))


def find_chip_dense_zone(df, around_idx, lookback=30, bins=24):
    left = max(0, around_idx - lookback + 1)
    seg = df.iloc[left:around_idx+1].dropna(subset=['High', 'Low', 'Close', 'Volume'])
    if len(seg) < 8:
        return None
    price_low = float(seg['Low'].min())
    price_high = float(seg['High'].max())
    if not np.isfinite(price_low) or not np.isfinite(price_high) or price_high <= price_low:
        return None
    edges = np.linspace(price_low, price_high, bins + 1)
    weights = np.zeros(bins, dtype=float)
    for _, row in seg.iterrows():
        lo = float(row['Low'])
        hi = float(row['High'])
        vol = max(float(row['Volume']), 0.0)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi < lo:
            continue
        if hi == lo:
            idx = int(np.clip(np.searchsorted(edges, lo, side='right') - 1, 0, bins - 1))
            weights[idx] += vol
            continue
        touched = np.where((edges[:-1] < hi) & (edges[1:] > lo))[0]
        if len(touched) == 0:
            idx = int(np.clip(np.searchsorted(edges, (lo + hi) / 2.0, side='right') - 1, 0, bins - 1))
            weights[idx] += vol
            continue
        span = hi - lo
        for idx in touched:
            overlap = max(0.0, min(hi, edges[idx + 1]) - max(lo, edges[idx]))
            if overlap > 0:
                weights[idx] += vol * (overlap / span)
    if float(weights.sum()) <= 0:
        return None
    peak_idx = int(np.argmax(weights))
    peak_mid = float((edges[peak_idx] + edges[peak_idx + 1]) / 2.0)
    width = max((price_high - price_low) / bins * 1.5, peak_mid * 0.006)
    return peak_mid - width, peak_mid + width, peak_mid


def find_recent_desc_trendline_break(df, confirm_idx, lookback=SHORT_TREND_LOOKBACK, window=SWING_WINDOW):
    start = max(0, confirm_idx - lookback)
    seg = df.iloc[start:confirm_idx+1].reset_index(drop=True)
    _, highs = local_extrema(seg, 'high', lookback=len(seg), window=window)
    anchors = []
    for idx in highs:
        anchors.append((start + idx, float(seg.iloc[idx]['High'])))
    if len(anchors) < 2:
        return None
    for b in range(len(anchors) - 1, 0, -1):
        for a in range(b - 1, -1, -1):
            idx1, p1 = anchors[a]
            idx2, p2 = anchors[b]
            if idx2 <= idx1 or p2 >= p1:
                continue
            slope = (p2 - p1) / (idx2 - idx1)
            line_at_confirm = p1 + slope * (confirm_idx - idx1)
            if float(df.iloc[confirm_idx]['Close']) > line_at_confirm:
                return {'anchor1': idx1, 'anchor2': idx2, 'line_value': line_at_confirm}
    return None


def find_recent_asc_trendline_break(df, confirm_idx, lookback=SHORT_TREND_LOOKBACK, window=SWING_WINDOW):
    start = max(0, confirm_idx - lookback)
    seg = df.iloc[start:confirm_idx+1].reset_index(drop=True)
    _, lows = local_extrema(seg, 'low', lookback=len(seg), window=window)
    anchors = []
    for idx in lows:
        anchors.append((start + idx, float(seg.iloc[idx]['Low'])))
    if len(anchors) < 2:
        return None
    for b in range(len(anchors) - 1, 0, -1):
        for a in range(b - 1, -1, -1):
            idx1, p1 = anchors[a]
            idx2, p2 = anchors[b]
            if idx2 <= idx1 or p2 <= p1:
                continue
            slope = (p2 - p1) / (idx2 - idx1)
            line_at_confirm = p1 + slope * (confirm_idx - idx1)
            if float(df.iloc[confirm_idx]['Close']) < line_at_confirm:
                return {'anchor1': idx1, 'anchor2': idx2, 'line_value': line_at_confirm}
    return None


def qualifies_reclaim_after_fib_break_long(df, fib618, idx, max_days=5):
    close = float(df.iloc[idx]['Close'])
    low = float(df.iloc[idx]['Low'])
    if not (close < fib618 and low < fib618):
        return True, False
    end = min(len(df) - 1, idx + max_days)
    for j in range(idx + 1, end + 1):
        row = df.iloc[j]
        prev = df.iloc[j - 1]
        bullish_candle = float(row['Close']) > float(row['Open']) and float(row['Close']) >= fib618
        gap_reclaim = float(row['Open']) >= fib618 and float(prev['Close']) < fib618
        if bullish_candle or gap_reclaim:
            return True, True
    return False, False


def qualifies_reclaim_after_fib_break_short(df, fib618, idx, max_days=5):
    close = float(df.iloc[idx]['Close'])
    high = float(df.iloc[idx]['High'])
    if not (close > fib618 and high > fib618):
        return True, False
    end = min(len(df) - 1, idx + max_days)
    for j in range(idx + 1, end + 1):
        row = df.iloc[j]
        prev = df.iloc[j - 1]
        bearish_candle = float(row['Close']) < float(row['Open']) and float(row['Close']) <= fib618
        gap_reclaim = float(row['Open']) <= fib618 and float(prev['Close']) > fib618
        if bearish_candle or gap_reclaim:
            return True, True
    return False, False


def nearest_swing_high(df, start_idx, end_idx):
    if end_idx <= start_idx:
        return None
    seg = df.iloc[start_idx:end_idx+1]
    if len(seg) == 0:
        return None
    idx = int(seg['High'].idxmax())
    return idx


def nearest_swing_low(df, start_idx, end_idx):
    if end_idx <= start_idx:
        return None
    seg = df.iloc[start_idx:end_idx+1]
    if len(seg) == 0:
        return None
    idx = int(seg['Low'].idxmin())
    return idx


def pct_diff(a, b):
    denom = (abs(a)+abs(b))/2.0
    return abs(a-b)/denom if denom else 999


def valid_double_bottom_structure(sdf: pd.DataFrame, li: int, lj: int) -> bool:
    if lj <= li:
        return False
    if (lj - li) < MIN_DOUBLE_STRUCTURE_GAP:
        return False
    middle = sdf.iloc[li+1:lj]
    if len(middle) == 0:
        return False
    left_low = float(sdf.iloc[li]['Low'])
    right_low = float(sdf.iloc[lj]['Low'])
    threshold = min(left_low, right_low)
    if float(middle['Low'].min()) < threshold:
        return False
    return True


def valid_double_top_structure(sdf: pd.DataFrame, hi: int, hj: int) -> bool:
    if hj <= hi:
        return False
    if (hj - hi) < MIN_DOUBLE_STRUCTURE_GAP:
        return False
    middle = sdf.iloc[hi+1:hj]
    if len(middle) == 0:
        return False
    left_high = float(sdf.iloc[hi]['High'])
    right_high = float(sdf.iloc[hj]['High'])
    threshold = max(left_high, right_high)
    if float(middle['High'].max()) > threshold:
        return False
    return True


def make_result(symbol, direction, pattern, zone, event_date, confirm_date, pullback_date, price, fib618, volume_feature, slowdown_feature, score, logic, recent_windows=None):
    return {
        'symbol': symbol,
        'direction': direction,
        'pattern': pattern,
        'zone': zone,
        'event_date': event_date.strftime('%Y-%m-%d'),
        'confirm_date': confirm_date.strftime('%Y-%m-%d'),
        'pullback_date': pullback_date.strftime('%Y-%m-%d'),
        'price': round(float(price), 2),
        'fib618': round(float(fib618), 2),
        'volume_feature': volume_feature,
        'slowdown_feature': slowdown_feature,
        'score': round(float(score), 1),
        'logic': logic,
        'recent_windows': recent_windows or [],
        '_sort_pullback': pullback_date,
        '_sort_event': event_date,
        '_sort_confirm': confirm_date,
    }


def build_recent_windows(df, points, bullish=True, max_windows=3, max_gap_days=3):
    if not points:
        return []
    ordered = sorted(points, key=lambda x: x['idx'])
    grouped = []
    current = [ordered[0]]
    for point in ordered[1:]:
        prev_date = df.iloc[current[-1]['idx']]['Date']
        point_date = df.iloc[point['idx']]['Date']
        gap_days = int((point_date - prev_date).days)
        if gap_days <= max_gap_days:
            current.append(point)
        else:
            grouped.append(current)
            current = [point]
    grouped.append(current)

    out = []
    for group in grouped[-max_windows:]:
        if bullish:
            rep = min(group, key=lambda x: (x['price_level'], x['idx']))
        else:
            rep = max(group, key=lambda x: (x['price_level'], -x['idx']))
        out.append({
            'start_date': df.iloc[group[0]['idx']]['Date'].strftime('%Y-%m-%d'),
            'end_date': df.iloc[group[-1]['idx']]['Date'].strftime('%Y-%m-%d'),
            'representative_date': df.iloc[rep['idx']]['Date'].strftime('%Y-%m-%d'),
            'representative_price': round(float(rep['price_level']), 2),
            'count': len(group),
        })
    return out


def clone_row_for_liquidity_band(row, band_key):
    band_windows = [w for w in (row.get('recent_windows') or []) if w.get('liquidity_band') == band_key]
    if not band_windows:
        return None
    new_row = dict(row)
    new_row['recent_windows'] = band_windows
    new_row['liquidity_band'] = band_key
    new_row['pullback_date'] = band_windows[-1]['representative_date']
    new_row['_sort_pullback'] = pd.Timestamp(band_windows[-1]['representative_date'])
    return new_row


def render_markdown_report(out: dict) -> str:
    lines = []
    lines.append("# 港股假突破 / 破底翻形態簡報")
    lines.append("")
    miss_total = int(out.get('stage1_misses', 0)) + int(out.get('stage2_misses', 0))
    miss_note = f"；数据下载失败 {miss_total} 个" if miss_total else ""
    lines.append(
        f"摘要：共扫描 {out.get('universe_total', 0)} 个标的，"
        f"通过流动性过滤 {out.get('liquid_count', 0)} 个，"
        f"深度扫描 {out.get('deep_scan_count', 0)} 个，"
        f"形成候选 {out.get('candidate_total', 0)} 个，"
        f"其中做多 {out.get('long_candidates', 0)} 个、做空 {out.get('short_candidates', 0)} 个，"
        f"最终输出前 {len(out.get('top10', []))} 个{miss_note}。"
    )
    lines.append("")
    top10 = out.get('top10', []) or []
    if not top10:
        lines.append("今日无符合假突破 / 破底翻确认条件的标的。")
        if out.get('stderr_log'):
            lines.append("")
            lines.append(f"日志：`{out['stderr_log']}`")
        return "\n".join(lines)

    lines.append("| 代码 | 方向 | 形态 | 支撑/阻力区 | 母形态事件日 | 确认日 | 代表确认日 | 现价 | 0.618参考位 | 量能特征 | 备注 | 质量分 | 一句话逻辑 |")
    lines.append("|---|---|---|---|---|---|---|---:|---:|---|---|---:|---|")

    def display_pullback_dates(row: dict) -> str:
        windows = row.get('recent_windows') or []
        if windows:
            return ' / '.join(w.get('representative_date', '') for w in windows if w.get('representative_date'))
        return row['pullback_date']

    for row in top10:
        lines.append(
            f"| {row['symbol']} | {row['direction']} | {row['pattern']} | {row['zone']} | {row['event_date']} | {row['confirm_date']} | {display_pullback_dates(row)} | {row['price']:.2f} | {row['fib618']:.2f} | {row['volume_feature']} | {row['slowdown_feature']} | {row['score']:.1f} | {row['logic']} |"
        )

    lines.append("")
    lines.append("## 观察要点")
    lines.append("")
    long_top = [x for x in top10 if x['direction'] == '做多']
    short_top = [x for x in top10 if x['direction'] == '做空']
    qty_confirm = sum(1 for x in top10 if x['volume_feature'] == '确认放量')
    newest = top10[0]
    lines.append(f"- 今日最优先关注的是最新确认的标的：**{newest['symbol']}**（{newest['direction']} / {newest['pattern']}）。")
    lines.append(f"- 前10中确认日出现放量的共有 **{qty_confirm}** 个，可用作确认力度参考。")
    lines.append(f"- 多头候选 **{len(long_top)}** 个，空头候选 **{len(short_top)}** 个，可用来判断当天偏风险偏好还是偏防守。")
    lines.append("- 本版本只保留假突破 / 破底翻本體确认，不再混入确认后回调、趋势线二次介入、减速或量缩回踩等延伸条件。")
    lines.append("- 支撑/阻力区与前一日阴阳烛区间均为日線近似计算，适合做盘后筛选，不替代盘中确认。")
    return "\n".join(lines)


def scan_stage2_dataset(stage2, mapped, stderr_path):
    results = []
    long_count = 0
    short_count = 0
    for ys, df in stage2.items():
        try:
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume']).reset_index(drop=True)
            if len(df) < 120:
                continue
            long_r = scan_long(mapped[ys], df)
            short_r = scan_short(mapped[ys], df)
            if long_r:
                results.append(long_r)
                long_count += 1
            if short_r:
                results.append(short_r)
                short_count += 1
        except Exception as e:
            append_log(stderr_path, f"SCAN_ERROR {ys} {e}\\n{traceback.format_exc()}")
    return results, long_count, short_count


def scan_long(symbol, df):
    if len(df) < 140:
        return None
    sdf, lows = local_extrema(df, 'low', 90, SWING_WINDOW)
    candidates = []
    base_offset = len(df) - len(sdf)
    for i in range(len(lows)):
        for j in range(i+1, len(lows)):
            li, lj = lows[i], lows[j]
            if not valid_double_bottom_structure(sdf, li, lj):
                continue
            p1, p2 = float(sdf.iloc[li]['Low']), float(sdf.iloc[lj]['Low'])
            if pct_diff(p1, p2) > 0.03:
                continue
            zone_low = min(p1, p2)
            post = sdf.iloc[lj+1:].copy()
            if len(post) < 3:
                continue
            breakdown_idx = None
            breakdown_mag = None
            for k in range(lj+1, len(sdf)):
                low = float(sdf.iloc[k]['Low'])
                close = float(sdf.iloc[k]['Close'])
                break_price = min(low, close)
                mag = (zone_low - break_price) / zone_low
                if 0.005 <= mag <= 0.08:
                    breakdown_idx = k
                    breakdown_mag = mag
                    break
            if breakdown_idx is None or breakdown_idx <= 0:
                continue
            prev_bar = sdf.iloc[breakdown_idx - 1]
            reclaim_low = min(float(prev_bar['Open']), float(prev_bar['Close']))
            reclaim_high = max(float(prev_bar['Open']), float(prev_bar['Close']))
            confirm_idx = None
            for k in range(breakdown_idx + 1, len(sdf)):
                close = float(sdf.iloc[k]['Close'])
                high = float(sdf.iloc[k]['High'])
                if close >= reclaim_low or high >= reclaim_low:
                    confirm_idx = k
                    break
            if confirm_idx is None:
                continue
            global_breakdown = base_offset + breakdown_idx
            global_confirm = base_offset + confirm_idx
            if global_confirm < len(df) - 30:
                continue
            avg_dollar_volume = trailing_avg_dollar_volume(df, global_confirm, days=20)
            liquidity_band = liquidity_band_from_avg_dollar_volume(avg_dollar_volume)
            if not liquidity_band or avg_dollar_volume is None:
                continue
            confirm_close = float(df.iloc[global_confirm]['Close'])
            confirm_high = float(df.iloc[global_confirm]['High'])
            confirm_low = float(df.iloc[global_confirm]['Low'])
            fib618 = reclaim_low + 0.618 * (reclaim_high - reclaim_low)
            avg20_vol = float(df.iloc[max(0, global_confirm - 20):global_confirm]['Volume'].mean()) if global_confirm > 0 else 0.0
            confirm_vol = float(df.iloc[global_confirm]['Volume'])
            volume_feature = '确认放量' if avg20_vol > 0 and confirm_vol > avg20_vol else '一般'
            recent_windows = [{
                'start_date': df.iloc[global_confirm]['Date'].strftime('%Y-%m-%d'),
                'end_date': df.iloc[global_confirm]['Date'].strftime('%Y-%m-%d'),
                'representative_date': df.iloc[global_confirm]['Date'].strftime('%Y-%m-%d'),
                'representative_price': round(confirm_close, 2),
                'count': 1,
                'avg_5d_dollar_volume': round(float(avg_dollar_volume), 2),
                'liquidity_band': liquidity_band,
            }]
            score = 50
            score += max(0, 15 - pct_diff(p1, p2) * 500)
            if (lj - li) >= DOUBLE_STRUCTURE_WIDE_GAP_THRESHOLD:
                score += DOUBLE_STRUCTURE_WIDE_GAP_BONUS
            score += max(0, 10 - abs(float(breakdown_mag or 0.0) - 0.025) * 120)
            score += score_confirm_day(df, global_confirm, bullish=True)
            if confirm_close >= reclaim_high:
                score += 6
            elif confirm_high >= reclaim_high:
                score += 3
            if confirm_low >= zone_low:
                score += 4
            logic = '破底翻本體：近似雙底被向下跌破後，價格拉回至跌破日前一天陰陽燭區間；不包含突破趨勢線後再找回調的延伸條件'
            candidates.append(make_result(
                symbol=symbol,
                direction='做多',
                pattern='破底翻确认',
                zone=f"前一日陰陽燭區間 {reclaim_low:.2f}-{reclaim_high:.2f}",
                event_date=df.iloc[global_breakdown]['Date'],
                confirm_date=df.iloc[global_confirm]['Date'],
                pullback_date=df.iloc[global_confirm]['Date'],
                price=df.iloc[-1]['Close'],
                fib618=fib618,
                volume_feature=volume_feature,
                slowdown_feature='不適用',
                score=score,
                logic=logic,
                recent_windows=recent_windows,
            ))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x['score'], x['_sort_confirm'], x['_sort_event']), reverse=True)
    return candidates[0]


def scan_short(symbol, df):
    if len(df) < 140:
        return None
    sdf, highs = local_extrema(df, 'high', 90, SWING_WINDOW)
    candidates = []
    base_offset = len(df) - len(sdf)
    for i in range(len(highs)):
        for j in range(i+1, len(highs)):
            hi, hj = highs[i], highs[j]
            if not valid_double_top_structure(sdf, hi, hj):
                continue
            p1, p2 = float(sdf.iloc[hi]['High']), float(sdf.iloc[hj]['High'])
            if pct_diff(p1, p2) > 0.03:
                continue
            zone_low = min(p1, p2)
            zone_high = max(p1, p2)
            post = sdf.iloc[hj+1:].copy()
            if len(post) < 3:
                continue
            breakout_idx = None
            breakout_mag = None
            for k in range(hj+1, len(sdf)):
                high = float(sdf.iloc[k]['High'])
                close = float(sdf.iloc[k]['Close'])
                break_price = max(high, close)
                mag = (break_price - zone_high) / zone_high
                if 0.005 <= mag <= 0.08:
                    breakout_idx = k
                    breakout_mag = mag
                    break
            if breakout_idx is None:
                continue
            confirm_idx = None
            for k in range(breakout_idx + 1, len(sdf)):
                close = float(sdf.iloc[k]['Close'])
                if close <= zone_low:
                    confirm_idx = k
                    break
            if confirm_idx is None:
                continue
            global_breakout = base_offset + breakout_idx
            global_confirm = base_offset + confirm_idx
            if global_confirm < len(df) - 30:
                continue
            avg_dollar_volume = trailing_avg_dollar_volume(df, global_confirm, days=20)
            liquidity_band = liquidity_band_from_avg_dollar_volume(avg_dollar_volume)
            if not liquidity_band or avg_dollar_volume is None:
                continue
            confirm_close = float(df.iloc[global_confirm]['Close'])
            confirm_high = float(df.iloc[global_confirm]['High'])
            confirm_low = float(df.iloc[global_confirm]['Low'])
            fib618 = zone_low + 0.618 * (zone_high - zone_low)
            avg20_vol = float(df.iloc[max(0, global_confirm - 20):global_confirm]['Volume'].mean()) if global_confirm > 0 else 0.0
            confirm_vol = float(df.iloc[global_confirm]['Volume'])
            volume_feature = '确认放量' if avg20_vol > 0 and confirm_vol > avg20_vol else '一般'
            recent_windows = [{
                'start_date': df.iloc[global_confirm]['Date'].strftime('%Y-%m-%d'),
                'end_date': df.iloc[global_confirm]['Date'].strftime('%Y-%m-%d'),
                'representative_date': df.iloc[global_confirm]['Date'].strftime('%Y-%m-%d'),
                'representative_price': round(confirm_close, 2),
                'count': 1,
                'avg_5d_dollar_volume': round(float(avg_dollar_volume), 2),
                'liquidity_band': liquidity_band,
            }]
            score = 50
            score += max(0, 15 - pct_diff(p1, p2) * 500)
            if (hj - hi) >= DOUBLE_STRUCTURE_WIDE_GAP_THRESHOLD:
                score += DOUBLE_STRUCTURE_WIDE_GAP_BONUS
            score += max(0, 10 - abs(float(breakout_mag) - 0.025) * 120)
            score += score_confirm_day(df, global_confirm, bullish=False)
            if confirm_close <= zone_low:
                score += 6
            elif confirm_low <= zone_low:
                score += 3
            if confirm_high <= zone_high:
                score += 4
            logic = '假突破本體：近似雙頂被向上假突破後，價格重新跌回原阻力區下方；不包含跌破趨勢線後再等回抽的延伸條件'
            candidates.append(make_result(
                symbol=symbol,
                direction='做空',
                pattern='假突破确认',
                zone=f"原阻力區 {zone_low:.2f}-{zone_high:.2f}",
                event_date=df.iloc[global_breakout]['Date'],
                confirm_date=df.iloc[global_confirm]['Date'],
                pullback_date=df.iloc[global_confirm]['Date'],
                price=df.iloc[-1]['Close'],
                fib618=fib618,
                volume_feature=volume_feature,
                slowdown_feature='不適用',
                score=score,
                logic=logic,
                recent_windows=recent_windows,
            ))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x['score'], x['_sort_confirm'], x['_sort_event']), reverse=True)
    return candidates[0]



def load_universe(max_symbols: int):
    uni, _universe_source = fetch_hk_universe(prefer_cache=True)
    uni = uni.drop_duplicates(subset=['symbol']).reset_index(drop=True)
    if max_symbols and max_symbols > 0:
        uni = uni.head(max_symbols).copy()
    original_symbols = uni['stock_code'].astype(str).str.zfill(4).tolist()
    yahoo_symbols = uni['symbol'].astype(str).str.upper().tolist()
    mapped = dict(zip(yahoo_symbols, original_symbols))
    return original_symbols, mapped, yahoo_symbols


def run_stage1_only(args, stderr_path: str, artifact_dir: Path):
    original_symbols, mapped, yahoo_symbols = load_universe(args.max_symbols)
    append_log(stderr_path, f"STAGE1_START universe={len(yahoo_symbols)}")
    stage1, miss1 = download_bars(yahoo_symbols, args.stage1_period, stderr_path, batch=args.stage1_batch, phase='STAGE1')
    liquid = []
    for ys, df in stage1.items():
        x = df.dropna(subset=['Close', 'Volume']).reset_index(drop=True)
        if len(x) == 0:
            continue
        passed, _avg_dollar_vol = passes_stage1_liquidity(x)
        if passed:
            liquid.append(ys)
    out = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'universe_total': len(yahoo_symbols),
        'liquid_count': len(liquid),
        'stage1_misses': len(miss1),
        'liquid_symbols': liquid,
        'universe_source': 'local_cache',
        'stage1_liquidity_days': STAGE1_LIQUIDITY_LOOKBACK_DAYS,
        'stage1_liquidity_min_avg_turnover_hkd': STAGE1_LIQUIDITY_MIN_AVG_DOLLAR_VOLUME_HKD,
    }
    append_log(stderr_path, f"STAGE1_DONE ok={len(stage1)} liquid={len(liquid)} misses={len(miss1)}")
    (artifact_dir / 'liquid_symbols.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    return out, original_symbols, mapped, liquid


def run_stage2_shards(liquid, mapped, args, stderr_path: str, artifact_dir: Path, *, shard_count_override=None, selected_shard_index=None):
    if selected_shard_index is not None:
        shard_count = max(1, int(shard_count_override or 1))
        all_shards = split_into_shards(liquid, shard_count)
        shard_symbols = all_shards[selected_shard_index - 1] if 1 <= selected_shard_index <= len(all_shards) else []
        shard_lists = [shard_symbols]
        shard_numbers = [selected_shard_index]
        shard_total = shard_count
    else:
        shard_total = max(1, int(shard_count_override or args.shards))
        shard_lists = split_into_shards(liquid, shard_total)
        shard_numbers = list(range(1, len(shard_lists) + 1))

    results = []
    long_count = 0
    short_count = 0
    deep_scan_count = 0
    miss2 = set()
    shard_summaries = []

    for shard_number, shard_symbols in zip(shard_numbers, shard_lists):
        append_log(stderr_path, f"STAGE2_SHARD_START shard={shard_number}/{shard_total} symbols={len(shard_symbols)}")
        stage2, shard_miss = download_bars(shard_symbols, '8mo', stderr_path, batch=args.stage2_batch, phase=f'STAGE2_SHARD_{shard_number:02d}')
        shard_results, shard_long, shard_short = scan_stage2_dataset(stage2, mapped, stderr_path)
        deep_scan_count += len(stage2)
        miss2.update(shard_miss)
        results.extend(shard_results)
        long_count += shard_long
        short_count += shard_short
        shard_summary = {
            'shard': shard_number,
            'input_symbols': len(shard_symbols),
            'downloaded_symbols': len(stage2),
            'misses': len(shard_miss),
            'candidates': len(shard_results),
            'long_candidates': shard_long,
            'short_candidates': shard_short,
        }
        shard_summaries.append(shard_summary)
        shard_path = artifact_dir / f'shard_{shard_number:02d}.json'
        shard_path.write_text(
            json.dumps({
                'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'summary': shard_summary,
                'deep_scan_count': len(stage2),
                'results': shard_results,
                'miss_symbols': sorted(list(shard_miss)),
            }, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8',
        )
        append_log(stderr_path, f"STAGE2_SHARD_DONE shard={shard_number}/{shard_total} downloaded={len(stage2)} misses={len(shard_miss)} candidates={len(shard_results)}")

    return {
        'results': results,
        'long_count': long_count,
        'short_count': short_count,
        'deep_scan_count': deep_scan_count,
        'miss2': miss2,
        'shard_summaries': shard_summaries,
        'shard_count': shard_total if selected_shard_index is None else 1,
    }


def build_full_output(original_symbols, liquid, stage1_misses, stage2_payload, stderr_path: str, artifact_dir: Path):
    results = stage2_payload['results']
    long_count = stage2_payload['long_count']
    short_count = stage2_payload['short_count']
    deep_scan_count = stage2_payload['deep_scan_count']
    miss2 = stage2_payload['miss2']
    shard_summaries = stage2_payload['shard_summaries']

    results.sort(key=lambda x: (x['_sort_pullback'], x['score'], x['_sort_event'], x['_sort_confirm']), reverse=True)
    deduped = []
    seen_symbols = set()
    for row in results:
        if row['symbol'] in seen_symbols:
            continue
        deduped.append(row)
        seen_symbols.add(row['symbol'])
    top10 = deduped[:10]
    top10_long = [row for row in deduped if row['direction'] == '做多'][:10]
    top10_short = [row for row in deduped if row['direction'] == '做空'][:10]

    band_rows_20m_to_50m = []
    band_rows_50m_plus = []
    for row in deduped:
        row_20m_to_50m = clone_row_for_liquidity_band(row, '20m_to_50m')
        row_50m_plus = clone_row_for_liquidity_band(row, '50m_plus')
        if row_20m_to_50m:
            band_rows_20m_to_50m.append(row_20m_to_50m)
        if row_50m_plus:
            band_rows_50m_plus.append(row_50m_plus)

    band_rows_20m_to_50m.sort(key=lambda x: (x['_sort_pullback'], x['score'], x['_sort_event'], x['_sort_confirm']), reverse=True)
    band_rows_50m_plus.sort(key=lambda x: (x['_sort_pullback'], x['score'], x['_sort_event'], x['_sort_confirm']), reverse=True)

    top10_long_20m_to_50m = [row for row in band_rows_20m_to_50m if row['direction'] == '做多'][:10]
    top10_short_20m_to_50m = [row for row in band_rows_20m_to_50m if row['direction'] == '做空'][:10]
    top10_long_50m_plus = [row for row in band_rows_50m_plus if row['direction'] == '做多'][:10]
    top10_short_50m_plus = [row for row in band_rows_50m_plus if row['direction'] == '做空'][:10]

    out = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'data_sources': [
            'HKEX ListOfSecurities.xlsx / 本地月更港股股票池快取（HKD 櫃台；掃描前先排除每月小流動性/失效代號池）',
            'Yahoo Finance / yfinance 日線 OHLCV',
            '收復日過去20個交易日平均成交額分組（2000萬-5000萬港元；5000萬港元以上）',
        ],
        'universe_total': int(len(original_symbols)),
        'liquid_count': int(len(liquid)),
        'deep_scan_count': int(deep_scan_count),
        'stage1_misses': int(stage1_misses),
        'stage2_misses': int(len(miss2)),
        'candidate_total': int(len(results)),
        'long_candidates': int(long_count),
        'short_candidates': int(short_count),
        'stderr_log': stderr_path,
        'artifact_dir': str(artifact_dir),
        'shard_count': len(shard_summaries),
        'shards': shard_summaries,
        'top10': top10,
        'top10_long': top10_long,
        'top10_short': top10_short,
        'top10_long_20m_to_50m': top10_long_20m_to_50m,
        'top10_short_20m_to_50m': top10_short_20m_to_50m,
        'top10_long_50m_plus': top10_long_50m_plus,
        'top10_short_50m_plus': top10_short_50m_plus,
    }
    (artifact_dir / 'final_output.json').write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    append_log(stderr_path, f"SCAN_DONE deep_scan={deep_scan_count} candidates={len(results)} deduped={len(deduped)}")
    return out


def main():
    parser = argparse.ArgumentParser(description='Hong Kong fake-breakout / breakdown-reclaim pattern scan')
    parser.add_argument('--format', choices=['json', 'markdown'], default='json')
    parser.add_argument('--mode', choices=['full', 'stage1', 'stage2'], default='full')
    parser.add_argument('--max-symbols', type=int, default=0, help='Optional cap on universe size for smoke tests')
    parser.add_argument('--stderr-path', default='/tmp/hk_pattern_scan_yf_stderr.log')
    parser.add_argument('--shards', type=int, default=int(os.environ.get('HERMES_SCAN_SHARDS', '4')), help='Number of internal stage2 shards')
    parser.add_argument('--artifact-dir', default=os.environ.get('HERMES_SCAN_ARTIFACT_DIR', ''), help='Optional directory for shard artifacts')
    parser.add_argument('--stage1-period', default=os.environ.get('HERMES_SCAN_STAGE1_PERIOD', DEFAULT_STAGE1_PERIOD), help='Short lookback window used for liquidity screening')
    parser.add_argument('--stage1-batch', type=int, default=int(os.environ.get('HERMES_SCAN_STAGE1_BATCH', '90')), help='Batch size for stage1 liquidity download')
    parser.add_argument('--stage2-batch', type=int, default=int(os.environ.get('HERMES_SCAN_STAGE2_BATCH', '120')), help='Batch size for stage2 deep-history download')
    parser.add_argument('--symbols-file', default='')
    parser.add_argument('--shard-index', type=int, default=1)
    parser.add_argument('--shard-count', type=int, default=0)
    args = parser.parse_args()

    stderr_path = args.stderr_path
    open(stderr_path, 'w').close()
    artifact_dir = Path(args.artifact_dir).expanduser() if args.artifact_dir else Path(stderr_path).resolve().parent / (Path(stderr_path).stem + '.artifacts')
    artifact_dir.mkdir(parents=True, exist_ok=True)

    append_log(
        stderr_path,
        f"SCAN_START mode={args.mode} format={args.format} max_symbols={args.max_symbols or 'all'} shards={max(1, args.shards)} stage1_period={args.stage1_period} stage1_batch={args.stage1_batch} stage2_batch={args.stage2_batch}"
    )

    if args.mode == 'stage1':
        out, _original_symbols, _mapped, _liquid = run_stage1_only(args, stderr_path, artifact_dir)
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
        return

    if args.mode == 'stage2':
        if not args.symbols_file:
            raise SystemExit('--symbols-file is required for --mode stage2')
        stage1_payload = json.loads(Path(args.symbols_file).read_text(encoding='utf-8'))
        liquid = list(stage1_payload.get('liquid_symbols') or [])
        shard_count = max(1, int(args.shard_count or args.shards or 1))
        _original_symbols, mapped, _yahoo_symbols = load_universe(0)
        stage2_payload = run_stage2_shards(liquid, mapped, args, stderr_path, artifact_dir, shard_count_override=shard_count, selected_shard_index=args.shard_index)
        if not stage2_payload['shard_summaries']:
            out = {
                'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'summary': {
                    'shard': args.shard_index,
                    'input_symbols': 0,
                    'downloaded_symbols': 0,
                    'misses': 0,
                    'candidates': 0,
                    'long_candidates': 0,
                    'short_candidates': 0,
                },
                'deep_scan_count': 0,
                'results': [],
                'miss_symbols': [],
            }
        else:
            shard_summary = stage2_payload['shard_summaries'][0]
            shard_path = artifact_dir / f"shard_{args.shard_index:02d}.json"
            out = json.loads(shard_path.read_text(encoding='utf-8')) if shard_path.exists() else {
                'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'summary': shard_summary,
                'deep_scan_count': stage2_payload['deep_scan_count'],
                'results': stage2_payload['results'],
                'miss_symbols': sorted(list(stage2_payload['miss2'])),
            }
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
        return

    stage1_out, original_symbols, mapped, liquid = run_stage1_only(args, stderr_path, artifact_dir)
    stage2_payload = run_stage2_shards(liquid, mapped, args, stderr_path, artifact_dir)
    out = build_full_output(original_symbols, liquid, stage1_out['stage1_misses'], stage2_payload, stderr_path, artifact_dir)
    if args.format == 'markdown':
        print(render_markdown_report(out))
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    main()
