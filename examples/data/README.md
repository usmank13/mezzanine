# Example datasets

## `spy_daily_ohlcv.csv`

- Source: Stooq daily bars for `SPY.US` (CSV download endpoint).
- Retrieved: 2026-02-09.
- Shape: header + 2000 daily OHLCV rows (`Date,Open,High,Low,Close,Volume`).

To refresh the file:
```bash
tmp=$(mktemp)
curl -fsSL 'https://stooq.com/q/d/l/?s=spy.us&i=d' > "$tmp"
{ head -n 1 "$tmp"; tail -n 2000 "$tmp"; } > examples/data/spy_daily_ohlcv.csv
rm -f "$tmp"
```

