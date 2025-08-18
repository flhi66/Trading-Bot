from __future__ import annotations

import pandas as pd


class RiskManager:
    """
    Centralized risk management utilities.

    - Dynamic ATR calculation for volatility-aware stops
    - Automated position sizing to risk a fixed fraction of capital
    """

    def __init__(self,
                 risk_per_trade: float = 0.02,
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0):
        self.risk_per_trade = risk_per_trade
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

    # --- ATR ---
    def calculate_atr(self, data: pd.DataFrame, period: int | None = None) -> pd.Series:
        """Return ATR series for given OHLCV DataFrame."""
        if data is None or data.empty:
            return pd.Series(dtype=float)

        period_to_use = period if period is not None else self.atr_period

        high = data['High']
        low = data['Low']
        close = data['Close']

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period_to_use).mean()
        return atr

    # --- Stops / Targets ---
    def compute_stop_and_target_from_atr(self,
                                         entry_price: float,
                                         direction: str,
                                         atr_value: float,
                                         reward_risk_ratio: float) -> tuple[float, float]:
        """Compute stop-loss and take-profit using ATR and configured multiplier.

        direction: "BUY" or "SELL"
        """
        if pd.isna(atr_value) or atr_value <= 0:
            return entry_price, entry_price

        risk_distance = atr_value * self.atr_multiplier

        if direction.upper() == "BUY":
            stop_loss = entry_price - risk_distance
            take_profit = entry_price + risk_distance * reward_risk_ratio
        else:
            stop_loss = entry_price + risk_distance
            take_profit = entry_price - risk_distance * reward_risk_ratio

        return stop_loss, take_profit

    # --- Position sizing ---
    def calculate_position_size(self,
                                entry_price: float,
                                stop_loss: float,
                                risk_amount: float) -> float:
        """Return position size such that loss at stop equals risk_amount."""
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return 0.0
        return risk_amount / risk_per_unit

    def risk_amount_for_balance(self, account_balance: float) -> float:
        return account_balance * self.risk_per_trade


