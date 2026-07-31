#!/usr/bin/env python3
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

HKEX_SECURITIES_URL = "https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx"
KEEP_CATEGORIES = {
    "Equity",
    "Exchange Traded Products",
    "Real Estate Investment Trusts",
}
EXCLUDE_SUBCATEGORIES = {
    "Depositary Receipts",
    "Trading Only Securities",
}
MONTHLY_EXCLUSION_FILENAME = 'monthly_excluded_symbols.json'
MANUAL_EXCLUSION_FILENAME = 'exclude_symbols.txt'


def normalize_stock_code(value: str) -> str:
    digits = ''.join(ch for ch in str(value).strip() if ch.isdigit())
    if not digits:
        return ''
    return f"{int(digits):04d}"


def hk_yahoo_symbol(stock_code: str) -> str:
    code = normalize_stock_code(stock_code)
    if not code:
        return ''
    return f"{code}.HK"


def default_universe_cache_dir() -> Path:
    return Path(__file__).resolve().parent / 'data' / 'universe'


def default_config_dir() -> Path:
    return Path(__file__).resolve().parent / 'config'


def parse_hkex_workbook(xlsx_path: Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, header=2)
    df = df.rename(columns={
        'Stock Code': 'stock_code',
        'Name of Securities': 'name',
        'Category': 'category',
        'Sub-Category': 'sub_category',
        'Trading Currency': 'currency',
    })
    df = df[df['stock_code'].notna()].copy()
    df['stock_code'] = df['stock_code'].astype(str).apply(normalize_stock_code)
    df = df[df['stock_code'].ne('')].copy()
    df['category'] = df['category'].fillna('').astype(str).str.strip()
    df['sub_category'] = df['sub_category'].fillna('').astype(str).str.strip()
    df['currency'] = df['currency'].fillna('').astype(str).str.strip().str.upper()
    df = df[df['category'].isin(KEEP_CATEGORIES)].copy()
    df = df[~df['sub_category'].isin(EXCLUDE_SUBCATEGORIES)].copy()
    df = df[df['currency'].eq('HKD')].copy()
    df['symbol'] = df['stock_code'].apply(hk_yahoo_symbol)
    df = df[df['symbol'].ne('')].copy()
    df = df.drop_duplicates(subset=['symbol']).reset_index(drop=True)
    cols = ['stock_code', 'symbol', 'name', 'category', 'sub_category', 'currency']
    return df[cols]


def download_hkex_workbook(target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(HKEX_SECURITIES_URL, timeout=60) as resp:
                data = resp.read()
            if not data:
                raise RuntimeError('empty HKEX securities workbook')
            target_path.write_bytes(data)
            return target_path
        except Exception as exc:
            last_error = exc
            if target_path.exists() and target_path.stat().st_size > 0:
                return target_path
            if attempt < 3:
                time.sleep(2 * attempt)
    raise RuntimeError(f'failed to download HKEX securities workbook: {last_error}')


def load_cached_universe(cache_dir: Path) -> pd.DataFrame | None:
    csv_path = cache_dir / 'hk_symbols.csv'
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    df = pd.read_csv(csv_path, dtype=str).fillna('')
    required = ['stock_code', 'symbol', 'name', 'category', 'sub_category', 'currency']
    if not set(required).issubset(set(df.columns)):
        return None
    df = df[required].copy()
    df['stock_code'] = df['stock_code'].astype(str).apply(normalize_stock_code)
    df['symbol'] = df['symbol'].astype(str).str.strip().str.upper()
    df = df[df['symbol'].str.endswith('.HK')].reset_index(drop=True)
    return df


def _normalize_exclusion_entry(raw: str) -> tuple[str, str] | None:
    text = str(raw).strip()
    if not text or text.startswith('#'):
        return None
    text = text.split('#', 1)[0].strip()
    if not text:
        return None
    symbol = ''
    stock_code = normalize_stock_code(text)
    upper = text.upper()
    if upper.endswith('.HK'):
        symbol = upper
        if not stock_code:
            stock_code = normalize_stock_code(upper[:-3])
    elif stock_code:
        symbol = hk_yahoo_symbol(stock_code)
    if not stock_code and not symbol:
        return None
    return stock_code, symbol


def load_symbol_exclusions(cache_dir: Path | None = None, config_dir: Path | None = None) -> dict:
    cache_dir = (cache_dir or default_universe_cache_dir()).expanduser().resolve()
    config_dir = (config_dir or default_config_dir()).expanduser().resolve()
    stock_codes: set[str] = set()
    symbols: set[str] = set()

    manual_path = config_dir / MANUAL_EXCLUSION_FILENAME
    if manual_path.exists():
        for line in manual_path.read_text(encoding='utf-8').splitlines():
            normalized = _normalize_exclusion_entry(line)
            if not normalized:
                continue
            stock_code, symbol = normalized
            if stock_code:
                stock_codes.add(stock_code)
            if symbol:
                symbols.add(symbol)

    generated_path = cache_dir / MONTHLY_EXCLUSION_FILENAME
    if generated_path.exists() and generated_path.stat().st_size > 0:
        payload = json.loads(generated_path.read_text(encoding='utf-8'))
        for raw in payload.get('generated_symbols') or []:
            normalized = _normalize_exclusion_entry(raw)
            if not normalized:
                continue
            stock_code, symbol = normalized
            if stock_code:
                stock_codes.add(stock_code)
            if symbol:
                symbols.add(symbol)

    return {
        'stock_codes': stock_codes,
        'symbols': symbols,
        'count': len(symbols | {hk_yahoo_symbol(code) for code in stock_codes if code}),
        'manual_path': str(manual_path),
        'generated_path': str(generated_path),
    }


def apply_symbol_exclusions(df: pd.DataFrame, exclusions: dict | None = None) -> pd.DataFrame:
    exclusions = exclusions or {}
    stock_codes = set(exclusions.get('stock_codes') or [])
    symbols = {str(x).strip().upper() for x in (exclusions.get('symbols') or []) if str(x).strip()}
    if not stock_codes and not symbols:
        return df.reset_index(drop=True)
    out = df.copy()
    out['stock_code'] = out['stock_code'].astype(str).apply(normalize_stock_code)
    out['symbol'] = out['symbol'].astype(str).str.strip().str.upper()
    mask = (~out['stock_code'].isin(stock_codes)) & (~out['symbol'].isin(symbols))
    return out.loc[mask].reset_index(drop=True)


def fetch_hk_universe(cache_dir: Path | None = None, prefer_cache: bool = True) -> tuple[pd.DataFrame, str]:
    cache_dir = (cache_dir or default_universe_cache_dir()).expanduser().resolve()
    exclusions = load_symbol_exclusions(cache_dir=cache_dir)
    if prefer_cache:
        cached = load_cached_universe(cache_dir)
        if cached is not None and len(cached) > 0:
            return apply_symbol_exclusions(cached, exclusions), 'local_cache'
    xlsx_path = download_hkex_workbook(cache_dir / 'hkex_ListOfSecurities.xlsx')
    df = parse_hkex_workbook(xlsx_path)
    return apply_symbol_exclusions(df, exclusions), 'live_hkex'
