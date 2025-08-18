import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

from .trend_detector import detect_trend, detect_swing_points
from .smart_money_concepts import MarketStructureAnalyzer, MarketEvent, EventType
from .data_loader import load_and_resample
from .risk_manager import RiskManager

warnings.filterwarnings('ignore')

@dataclass
class TradeSignal:
    """Represents a complete trade signal with all confirmations"""
    timestamp: pd.Timestamp
    direction: Literal["BUY", "SELL"]
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timeframe_1h_trend: str
    timeframe_15m_entry: str
    timeframe_1m_confirmation: str
    risk_reward_ratio: float
    stop_loss_pips: float
    take_profit_pips: float

@dataclass
class TradeExecution:
    """Represents an executed trade"""
    signal: TradeSignal
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    status: Literal["OPEN", "CLOSED", "CANCELLED"]
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None

class MultiTimeframeTradingExecutor:
    """
    Advanced trading executor that implements the multi-timeframe strategy:
    1. Check trend in 1H timeframe
    2. Mark A+ entries in 15M timeframe
    3. Confirm with 1M candle (green for buy, red for sell)
    4. Execute with 20 pip stop loss and 1:2 risk-reward ratio
    """
    
    def __init__(self, 
                 symbol: str = "EURUSD",
                 risk_per_trade: float = 0.02,  # 2% risk per trade
                 stop_loss_pips: float = 20.0,  # Legacy SL used if ATR unavailable
                 risk_reward_ratio: float = 2.0,  # 1:2 risk-reward
                 confidence_threshold: float = 0.7,  # A+ entry threshold
                 pip_value: float = 0.0001,  # Standard pip value for major pairs
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0):
        
        self.symbol = symbol
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pips = stop_loss_pips
        self.risk_reward_ratio = risk_reward_ratio
        self.confidence_threshold = confidence_threshold
        self.pip_value = pip_value
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        
        # Initialize analyzers
        self.market_analyzer = MarketStructureAnalyzer(confidence_threshold=confidence_threshold)
        self.risk_manager = RiskManager(risk_per_trade=risk_per_trade,
                                        atr_period=atr_period,
                                        atr_multiplier=atr_multiplier)
        
        # Track trades and signals
        self.signals: List[TradeSignal] = []
        self.executed_trades: List[TradeExecution] = []
        self.open_trades: List[TradeExecution] = []
        self._resampled_data: Optional[Dict[str, pd.DataFrame]] = None
        
    def analyze_1h_trend(self, data_1h: pd.DataFrame) -> str:
        """
        Analyze trend in 1H timeframe using swing point analysis
        
        Args:
            data_1h: 1H timeframe OHLCV data
            
        Returns:
            Trend direction: "uptrend", "downtrend", or "sideways"
        """
        if data_1h.empty or len(data_1h) < 20:
            return "sideways"
        
        # Use last 50 candles for trend analysis
        recent_data = data_1h.tail(50)
        
        # Detect swing points
        swing_highs, swing_lows = detect_swing_points(recent_data, window=3)
        
        # Determine trend
        trend = detect_trend(swing_highs, swing_lows)
        
        return trend
    
    def find_a_plus_entries_15m(self, data_15m: pd.DataFrame, trend_1h: str) -> List[MarketEvent]:
        """
        Find A+ quality entries in 15M timeframe that align with 1H trend
        
        Args:
            data_15m: 15M timeframe OHLCV data
            trend_1h: Trend from 1H timeframe
            
        Returns:
            List of high-quality market events
        """
        if data_15m.empty or len(data_15m) < 20:
            return []
        
        # Get market structure analysis
        structure = self._build_market_structure(data_15m)
        
        # Find market events
        events = self.market_analyzer.get_market_events(structure)
        
        # Filter for A+ quality entries that align with 1H trend
        a_plus_events = []
        
        for event in events:
            if event.confidence >= self.confidence_threshold:
                # Check trend alignment
                if self._is_trend_aligned(event, trend_1h):
                    a_plus_events.append(event)
        
        return a_plus_events
    
    def confirm_1m_signal(self, 
                          data_1m: pd.DataFrame, 
                          signal_direction: str, 
                          entry_time: pd.Timestamp) -> bool:
        """
        Confirm signal with 1M candle confirmation
        
        Args:
            data_1m: 1M timeframe OHLCV data
            signal_direction: "BUY" or "SELL"
            entry_time: Timestamp of the entry signal
            
        Returns:
            True if confirmation is valid, False otherwise
        """
        if data_1m.empty:
            return False
        
        # Find the 1M candle that starts after the entry time
        # Look for the next 1M candle that opens after the entry time
        entry_mask = data_1m.index > entry_time
        if not entry_mask.any():
            return False
        
        # Get the first 1M candle after entry
        next_candle_idx = entry_mask.idxmax()
        if next_candle_idx is None:
            return False
        
        # Check if we have enough data
        if next_candle_idx >= len(data_1m):
            return False
        
        next_candle = data_1m.loc[next_candle_idx]
        
        # Determine if it's a green (bullish) or red (bearish) candle
        is_green = next_candle['Close'] > next_candle['Open']
        is_red = next_candle['Close'] < next_candle['Open']
        
        # For BUY signals: wait for green 1M candle
        if signal_direction == "BUY":
            return is_green
        
        # For SELL signals: wait for red 1M candle
        elif signal_direction == "SELL":
            return is_red
        
        return False
    
    def generate_trade_signal(self, 
                             event: MarketEvent, 
                             trend_1h: str,
                             current_price: float) -> Optional[TradeSignal]:
        """
        Generate a complete trade signal with all confirmations
        
        Args:
            event: Market event from 15M timeframe
            trend_1h: Trend from 1H timeframe
            current_price: Current market price
            
        Returns:
            Complete trade signal or None if invalid
        """
        # Determine trade direction
        if event.direction == "Bullish" and event.event_type == EventType.BOS:
            direction = "BUY"
        elif event.direction == "Bearish" and event.event_type == EventType.BOS:
            direction = "SELL"
        elif event.direction == "Bullish" and event.event_type == EventType.CHOCH:
            direction = "BUY"
        elif event.direction == "Bearish" and event.event_type == EventType.CHOCH:
            direction = "SELL"
        else:
            return None
        
        # Calculate stop loss and take profit
        if direction == "BUY":
            stop_loss = current_price - (self.stop_loss_pips * self.pip_value)
            take_profit = current_price + (self.stop_loss_pips * self.pip_value * self.risk_reward_ratio)
        else:  # SELL
            stop_loss = current_price + (self.stop_loss_pips * self.pip_value)
            take_profit = current_price - (self.stop_loss_pips * self.pip_value * self.risk_reward_ratio)
        
        # Create trade signal
        signal = TradeSignal(
            timestamp=event.timestamp,
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=event.confidence,
            timeframe_1h_trend=trend_1h,
            timeframe_15m_entry=f"{event.event_type.value} - {event.direction}",
            timeframe_1m_confirmation="PENDING",
            risk_reward_ratio=self.risk_reward_ratio,
            stop_loss_pips=self.stop_loss_pips,
            take_profit_pips=self.stop_loss_pips * self.risk_reward_ratio
        )
        
        return signal
    
    def execute_trade(self, 
                     signal: TradeSignal, 
                     account_balance: float,
                     data_1m: pd.DataFrame) -> Optional[TradeExecution]:
        """
        Execute a trade after 1M confirmation
        
        Args:
            signal: Complete trade signal
            account_balance: Current account balance
            data_1m: 1M timeframe data for confirmation
            
        Returns:
            Executed trade or None if execution fails
        """
        # Wait for 1M confirmation
        if not self.confirm_1m_signal(data_1m, signal.direction, signal.timestamp):
            return None
        
        # Prefer ATR-based stops from 15M timeframe without lookahead
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit
        if self._resampled_data is not None:
            data_15m = self._resampled_data.get('15M')
            if data_15m is not None and not data_15m.empty:
                data_pre_entry = data_15m.loc[data_15m.index <= signal.timestamp]
                atr_series = self.risk_manager.calculate_atr(data_pre_entry)
                if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]):
                    atr_value = atr_series.iloc[-1]
                    stop_loss, take_profit = self.risk_manager.compute_stop_and_target_from_atr(
                        entry_price=signal.entry_price,
                        direction=signal.direction,
                        atr_value=atr_value,
                        reward_risk_ratio=self.risk_reward_ratio
                    )

        # Strict 2% risk sizing
        risk_amount = self.risk_manager.risk_amount_for_balance(account_balance)
        stop_loss_distance = abs(signal.entry_price - stop_loss)
        if stop_loss_distance == 0:
            return None
        position_size = self.risk_manager.calculate_position_size(
            entry_price=signal.entry_price,
            stop_loss=stop_loss,
            risk_amount=risk_amount
        )
        
        # Create trade execution
        trade = TradeExecution(
            signal=signal,
            entry_time=signal.timestamp,
            entry_price=signal.entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            status="OPEN"
        )
        
        # Update signal confirmation
        signal.timeframe_1m_confirmation = "CONFIRMED"
        
        # Add to open trades
        self.open_trades.append(trade)
        self.executed_trades.append(trade)
        
        return trade
    
    def monitor_open_trades(self, current_data: Dict[str, pd.DataFrame]) -> List[TradeExecution]:
        """
        Monitor open trades and check for exit conditions
        
        Args:
            current_data: Current market data for all timeframes
            
        Returns:
            List of trades that were closed
        """
        closed_trades = []
        
        for trade in self.open_trades[:]:  # Copy list to avoid modification during iteration
            if trade.status != "OPEN":
                continue
            
            # Check stop loss and take profit
            current_price = self._get_current_price(current_data)
            
            if current_price is None:
                continue
            
            # Check stop loss
            if self._is_stop_loss_hit(trade, current_price):
                trade.status = "CLOSED"
                trade.exit_time = pd.Timestamp.now()
                trade.exit_price = trade.stop_loss
                trade.pnl = self._calculate_pnl(trade)
                trade.exit_reason = "Stop Loss"
                closed_trades.append(trade)
                self.open_trades.remove(trade)
                continue
            
            # Check take profit
            if self._is_take_profit_hit(trade, current_price):
                trade.status = "CLOSED"
                trade.exit_time = pd.Timestamp.now()
                trade.exit_price = trade.take_profit
                trade.pnl = self._calculate_pnl(trade)
                trade.exit_reason = "Take Profit"
                closed_trades.append(trade)
                self.open_trades.remove(trade)
                continue
        
        return closed_trades
    
    def run_strategy(self, 
                    data_file: str,
                    days_back: int = 30) -> Dict:
        """
        Run the complete multi-timeframe trading strategy
        
        Args:
            data_file: Path to CSV data file
            days_back: Number of days to analyze
            
        Returns:
            Strategy results and statistics
        """
        print(f"🚀 Starting Multi-Timeframe Trading Strategy for {self.symbol}")
        print(f"📊 Analyzing {days_back} days of data")
        
        # Load and resample data
        print("📈 Loading and resampling data...")
        resampled_data = load_and_resample(data_file, days_back=days_back)
        self._resampled_data = resampled_data
        
        # Initialize results
        results = {
            'signals_generated': 0,
            'trades_executed': 0,
            'trades_closed': 0,
            'total_pnl': 0.0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        
        # Get required timeframes
        data_1h = resampled_data.get('1H')
        data_15m = resampled_data.get('15M')
        data_1m = resampled_data.get('1M')
        
        if data_1h is None or data_15m is None or data_1m is None:
            print("❌ Missing required timeframe data")
            return results
        
        print("🔍 Analyzing 1H trend...")
        trend_1h = self.analyze_1h_trend(data_1h)
        print(f"📊 1H Trend: {trend_1h.upper()}")
        
        print("🎯 Finding A+ entries in 15M timeframe...")
        a_plus_events = self.find_a_plus_entries_15m(data_15m, trend_1h)
        print(f"✨ Found {len(a_plus_events)} A+ quality entries")
        
        # Process each A+ entry
        for event in a_plus_events:
            print(f"\n🎯 Processing {event.event_type.value} - {event.direction} entry")
            print(f"   Confidence: {event.confidence:.2f}")
            print(f"   Price: {event.price:.5f}")
            
            # Generate trade signal
            current_price = event.price
            signal = self.generate_trade_signal(event, trend_1h, current_price)
            
            if signal is None:
                print("   ❌ Failed to generate trade signal")
                continue
            
            results['signals_generated'] += 1
            self.signals.append(signal)
            
            print(f"   ✅ Signal generated: {signal.direction} at {signal.entry_price:.5f}")
            print(f"   🛑 Stop Loss: {signal.stop_loss:.5f} ({signal.stop_loss_pips} pips)")
            print(f"   🎯 Take Profit: {signal.take_profit:.5f} ({signal.take_profit_pips} pips)")
            
            # Execute trade (simulate with current data)
            # In real implementation, you'd wait for live data
            trade = self.execute_trade(signal, 10000, data_1m)  # Simulate $10k account
            
            if trade:
                results['trades_executed'] += 1
                print(f"   🚀 Trade executed: {trade.position_size:.2f} units")
            else:
                print("   ⏳ Waiting for 1M confirmation...")
        
        # Monitor trades (simulate with historical data)
        print("\n📊 Monitoring trades...")
        closed_trades = self.monitor_open_trades(resampled_data)
        
        # Calculate final statistics
        for trade in closed_trades:
            results['trades_closed'] += 1
            results['total_pnl'] += trade.pnl or 0
            
            if (trade.pnl or 0) > 0:
                results['winning_trades'] += 1
            else:
                results['losing_trades'] += 1
        
        # Print final results
        print(f"\n🎉 Strategy Execution Complete!")
        print(f"📊 Results Summary:")
        print(f"   Signals Generated: {results['signals_generated']}")
        print(f"   Trades Executed: {results['trades_executed']}")
        print(f"   Trades Closed: {results['trades_closed']}")
        print(f"   Total P&L: ${results['total_pnl']:.2f}")
        print(f"   Winning Trades: {results['winning_trades']}")
        print(f"   Losing Trades: {results['losing_trades']}")
        
        if results['trades_closed'] > 0:
            win_rate = (results['winning_trades'] / results['trades_closed']) * 100
            print(f"   Win Rate: {win_rate:.1f}%")
        
        return results
    
    def _build_market_structure(self, data: pd.DataFrame) -> List:
        """Build market structure from price data"""
        # This is a simplified implementation
        # In practice, you'd use the full market structure builder
        structure = []
        
        # Find swing highs and lows
        swing_highs, swing_lows = detect_swing_points(data, window=3)
        
        # Convert to structure points
        for timestamp, price in swing_highs:
            structure.append({
                'timestamp': timestamp,
                'price': price,
                'type': 'HH'  # Simplified
            })
        
        for timestamp, price in swing_lows:
            structure.append({
                'timestamp': timestamp,
                'price': price,
                'type': 'LL'  # Simplified
            })
        
        # Sort by timestamp
        structure.sort(key=lambda x: x['timestamp'])
        return structure
    
    def _is_trend_aligned(self, event: MarketEvent, trend: str) -> bool:
        """Check if market event aligns with the trend"""
        if trend == "uptrend":
            return event.direction == "Bullish"
        elif trend == "downtrend":
            return event.direction == "Bearish"
        else:  # sideways
            return True  # Allow both directions in sideways market
    
    def _get_current_price(self, data: Dict[str, pd.DataFrame]) -> Optional[float]:
        """Get current price from the most recent data"""
        # Use 1M data for most current price
        data_1m = data.get('1M')
        if data_1m is None or data_1m.empty:
            return None
        
        return data_1m['Close'].iloc[-1]
    
    def _is_stop_loss_hit(self, trade: TradeExecution, current_price: float) -> bool:
        """Check if stop loss is hit"""
        if trade.signal.direction == "BUY":
            return current_price <= trade.stop_loss
        else:  # SELL
            return current_price >= trade.stop_loss
    
    def _is_take_profit_hit(self, trade: TradeExecution, current_price: float) -> bool:
        """Check if take profit is hit"""
        if trade.signal.direction == "BUY":
            return current_price >= trade.take_profit
        else:  # SELL
            return current_price <= trade.take_profit
    
    def _calculate_pnl(self, trade: TradeExecution) -> float:
        """Calculate P&L for a trade"""
        if trade.exit_price is None:
            return 0.0
        
        if trade.signal.direction == "BUY":
            return (trade.exit_price - trade.entry_price) * trade.position_size
        else:  # SELL
            return (trade.entry_price - trade.exit_price) * trade.position_size
