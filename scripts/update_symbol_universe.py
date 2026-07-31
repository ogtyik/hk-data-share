#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hk_universe import (
    HKEX_SECURITIES_URL,
    MANUAL_EXCLUSION_FILENAME,
    MONTHLY_EXCLUSION_FILENAME,
    default_config_dir,
    download_hkex_workbook,
    parse_hkex_workbook,
)
import hk_pattern_scan as scan

SMALLCAP_AVG_TURNOVER_30D_HKD = 15_000_000
DEFAULT_SCAN_PERIOD = '2mo'
DEFAULT_BATCH = 80


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_exclusion_rows(df: pd.DataFrame, *, stderr_path: str, period: str, batch: int) -> tuple[list[dict], list[str], list[str]]:
    symbols = df['symbol'].tolist()
    bars, misses = scan.download_bars(symbols, period, stderr_path, batch=batch, phase='MONTHLY_EXCLUSION')
    meta_by_symbol = {row['symbol']: row for row in df.to_dict('records')}
    rows: list[dict] = []
    smallcap_symbols: list[str] = []
    missing_symbols: list[str] = []

    for symbol in sorted(misses):
        meta = meta_by_symbol.get(symbol, {})
        missing_symbols.append(symbol)
        rows.append({
            'stock_code': meta.get('stock_code', ''),
            'symbol': symbol,
            'name': meta.get('name', ''),
            'category': meta.get('category', ''),
            'sub_category': meta.get('sub_category', ''),
            'reason': 'download_miss_or_possibly_delisted',
            'avg_turnover_30d_hkd': None,
            'valid_days': 0,
        })

    for symbol, price_df in bars.items():
        meta = meta_by_symbol.get(symbol, {})
        x = price_df.dropna(subset=['Close', 'Volume']).reset_index(drop=True)
        if len(x) == 0:
            missing_symbols.append(symbol)
            rows.append({
                'stock_code': meta.get('stock_code', ''),
                'symbol': symbol,
                'name': meta.get('name', ''),
                'category': meta.get('category', ''),
                'sub_category': meta.get('sub_category', ''),
                'reason': 'empty_bars_or_possibly_delisted',
                'avg_turnover_30d_hkd': None,
                'valid_days': 0,
            })
            continue
        avg_turnover_30d = scan.trailing_avg_dollar_volume(x, len(x) - 1, days=30)
        if avg_turnover_30d is not None and avg_turnover_30d < SMALLCAP_AVG_TURNOVER_30D_HKD:
            smallcap_symbols.append(symbol)
            rows.append({
                'stock_code': meta.get('stock_code', ''),
                'symbol': symbol,
                'name': meta.get('name', ''),
                'category': meta.get('category', ''),
                'sub_category': meta.get('sub_category', ''),
                'reason': 'avg_turnover_30d_below_15m_hkd',
                'avg_turnover_30d_hkd': round(float(avg_turnover_30d), 2),
                'valid_days': int(len(x)),
            })

    generated_symbols = sorted(set(smallcap_symbols) | set(missing_symbols))
    rows.sort(key=lambda row: (row['reason'], row['symbol']))
    return rows, generated_symbols, sorted(set(missing_symbols))


def main() -> None:
    parser = argparse.ArgumentParser(description='Update HK universe cache and monthly exclusion list.')
    parser.add_argument('--max-symbols', type=int, default=0, help='Optional cap for smoke tests.')
    parser.add_argument('--batch', type=int, default=DEFAULT_BATCH)
    parser.add_argument('--period', default=DEFAULT_SCAN_PERIOD)
    parser.add_argument('--stderr-path', default='')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = root / 'data' / 'universe'
    out_dir.mkdir(parents=True, exist_ok=True)
    config_dir = default_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    manual_exclude_path = config_dir / MANUAL_EXCLUSION_FILENAME
    if not manual_exclude_path.exists():
        manual_exclude_path.write_text('# 一行一個股票代號，可寫 0001 或 0001.HK\n', encoding='utf-8')

    stderr_path = args.stderr_path or str(out_dir / 'monthly_excluded_symbols.stderr.log')

    xlsx_path = download_hkex_workbook(out_dir / 'hkex_ListOfSecurities.xlsx')
    workbook_bytes = xlsx_path.read_bytes()
    df = parse_hkex_workbook(xlsx_path)
    if args.max_symbols and args.max_symbols > 0:
        df = df.head(args.max_symbols).copy()

    csv_path = out_dir / 'hk_symbols.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8')
    csv_bytes = csv_path.read_bytes()

    rows, generated_symbols, missing_symbols = build_exclusion_rows(
        df,
        stderr_path=stderr_path,
        period=args.period,
        batch=args.batch,
    )
    smallcap_symbols = sorted({row['symbol'] for row in rows if row['reason'] == 'avg_turnover_30d_below_15m_hkd'})

    exclusion_payload = {
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'thresholds': {
            'smallcap_avg_turnover_30d_hkd': SMALLCAP_AVG_TURNOVER_30D_HKD,
            'market_data_period': args.period,
            'market_data_batch': args.batch,
        },
        'counts': {
            'symbols': int(len(df)),
            'smallcap_symbols': int(len(smallcap_symbols)),
            'missing_symbols': int(len(missing_symbols)),
            'generated_symbols': int(len(generated_symbols)),
        },
        'generated_symbols': generated_symbols,
        'smallcap_symbols': smallcap_symbols,
        'missing_symbols': missing_symbols,
        'rows': rows,
    }

    exclusion_json_path = out_dir / MONTHLY_EXCLUSION_FILENAME
    exclusion_json_path.write_text(json.dumps(exclusion_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    exclusion_csv_path = out_dir / 'monthly_excluded_symbols.csv'
    pd.DataFrame(rows).to_csv(exclusion_csv_path, index=False, encoding='utf-8')
    exclusion_txt_path = out_dir / 'monthly_excluded_symbols.txt'
    exclusion_txt_path.write_text('\n'.join(generated_symbols) + ('\n' if generated_symbols else ''), encoding='utf-8')

    manifest = {
        'updated_at_utc': datetime.now(timezone.utc).isoformat(),
        'sources': {
            'hkex_list_of_securities_xlsx': HKEX_SECURITIES_URL,
            'yahoo_finance_daily_bars': 'yfinance',
        },
        'rules': {
            'keep_categories': ['Equity', 'Exchange Traded Products', 'Real Estate Investment Trusts'],
            'exclude_subcategories': ['Depositary Receipts', 'Trading Only Securities'],
            'pre_scan_generated_exclusions': '30日平均成交額 < 1500萬港元，或 Yahoo 對不到 / 疑似已下市代號',
        },
        'counts': {
            'symbols': int(len(df)),
            'generated_exclusions': int(len(generated_symbols)),
            'smallcap_symbols': int(len(smallcap_symbols)),
            'missing_symbols': int(len(missing_symbols)),
        },
        'files': {
            'hkex_ListOfSecurities.xlsx': {
                'sha256': sha256_bytes(workbook_bytes),
                'bytes': len(workbook_bytes),
            },
            'hk_symbols.csv': {
                'sha256': sha256_bytes(csv_bytes),
                'bytes': len(csv_bytes),
            },
            'monthly_excluded_symbols.json': {
                'sha256': sha256_bytes(exclusion_json_path.read_bytes()),
                'bytes': exclusion_json_path.stat().st_size,
            },
            'monthly_excluded_symbols.csv': {
                'sha256': sha256_bytes(exclusion_csv_path.read_bytes()),
                'bytes': exclusion_csv_path.stat().st_size,
            },
            'monthly_excluded_symbols.txt': {
                'sha256': sha256_bytes(exclusion_txt_path.read_bytes()),
                'bytes': exclusion_txt_path.stat().st_size,
            },
        },
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'manifest': manifest,
        'manual_exclude_path': str(manual_exclude_path),
        'stderr_path': stderr_path,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
