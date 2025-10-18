from pathlib import Path
import pandas as pd
import os

timeframes = {
    "D1": "1D",
    "4H": "4h",
    "1H": "1h",
    "30M": "30min",
    "15M": "15min",
    "5M": "5min",
    "2M": "2min",
    "1M": "1min"
}
DEFAULT_TIMEFRAMES = timeframes.copy()



def get_file_type(file_path):
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.csv':
        return 'csv'
    elif ext.lower() == '.pkl':
        return 'pkl'
    else:
        return 'unknown'

def load_and_resample(file: str | Path,
                      timeframes: dict[str, str] = DEFAULT_TIMEFRAMES,
                      days_back: int = 60) -> dict[str, pd.DataFrame]:
    """
    Loads OHLCV data from a CSV, resamples it to various timeframes,
    and ensures column names are capitalized for plotting.
    """
    path = Path(file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)    

    file_type = get_file_type(path)
    if file_type == 'pkl':
        # cols = ["open", "high", "low", "close", "volume"]
        df = pd.read_pickle(path)
        df.columns=['open', 'high', 'low', 'close', 'volume']
        # df.columns = cols  # Ensure correct column names
        # df.astype(float)[cols[:]]  # enforce numeric types
    elif file_type == 'csv':
        cols = ["datetime", "open", "high", "low", "close", "volume"]
        df = pd.read_csv(path, names=cols, header=None)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = (df.set_index("datetime")
          .astype(float)[cols[1:]])  # enforce numeric types
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    cutoff = df.index.max() - pd.Timedelta(days=days_back)
    df = df[df.index >= cutoff]

    return {k: (resample_ohlcv(df, rule=freq, base_freq='1min')).rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'})
                  for k, freq in timeframes.items()}


from collections import OrderedDict
OHLCV_AGG = OrderedDict((
    ('open', 'first'),
    ('high', 'max'),
    ('low', 'min'),
    ('close', 'last'),
    ('volume', 'sum'),
    ))

def resample_ohlcv(df, rule='1h', base_freq='1min', drop_partial_tail=False):
    """
    Resample OHLCV-DataFrame mit garantierter Vollständigkeit je Intervall.
    
    Parameter:
    - df: Pandas DataFrame mit datetime-indexiertem OHLCV-Format
    - rule: Ziel-Zeiteinheit (z.B. '15Min', '1h')
    - base_freq: Frequenz des Ausgangs-DataFrames (z.B. '1min' für M1)
    - drop_partial_tail: Wenn True, wird die letzte unvollständige Periode entfernt
    """
    
    # Gruppiere nach der Ziel-Zeiteinheit
    grouped = df.resample(rule, label='left', closed='left')

    # Anzahl der Zeilen pro Gruppe zählen
    counts = grouped.size()
    required = pd.Timedelta(rule) // pd.Timedelta(base_freq)

    # Nur Gruppen mit vollständiger Anzahl Kerzen behalten
    full_intervals = counts[counts >= required].index
    resampled = grouped.agg(OHLCV_AGG).loc[full_intervals]

    # Optional: letzte Zeile (unvollständige Periode) entfernen
    if drop_partial_tail and not resampled.empty:
        last_index = resampled.index[-1]
        last_end = last_index + pd.Timedelta(rule)
        if df.index.max() < last_end:
            resampled = resampled.iloc[:-1]

    return resampled