from pathlib import Path
import pandas as pd

timeframes = {
    "D1": "1D",
    "4H": "4h",
    "1H": "1h",
    "30M": "30min",
    "15M": "15min",
    "5M": "5min",
    "1M": "1min"
}
DEFAULT_TIMEFRAMES = timeframes.copy()

def load_and_resample(file: str | Path,
                      timeframes: dict[str, str] = DEFAULT_TIMEFRAMES,
                      days_back: int = 60) -> dict[str, pd.DataFrame]:
    path = Path(file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    cols = ["datetime", "open", "high", "low", "close", "volume"]
    df = pd.read_csv(path, names=cols, header=None)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = (df.set_index("datetime")
            .astype(float)[cols[1:]])  # enforce numeric types

    cutoff = df.index.max() - pd.Timedelta(days=days_back)
    df = df[df.index >= cutoff]

    return {
        k: (df.resample(freq)
                .agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum"
                }).dropna())
        for k, freq in timeframes.items()
    }
