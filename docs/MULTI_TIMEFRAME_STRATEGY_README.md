# Multi-Timeframe Trading Strategy

## Overview

This trading strategy implements a sophisticated multi-timeframe approach that combines trend analysis, pattern recognition, and momentum confirmation to identify high-probability trading opportunities. The strategy follows a systematic process across three timeframes to ensure optimal entry and exit points.

## Strategy Flow

### 1. 1H Timeframe - Trend Analysis
- **Purpose**: Determine the overall market direction
- **Method**: Swing point analysis using HH, HL, LH, LL patterns
- **Lookback**: Last 50 candles for trend determination
- **Output**: Uptrend, Downtrend, or Sideways classification

### 2. 15M Timeframe - A+ Entry Detection
- **Purpose**: Identify high-quality entry points
- **Method**: BOS (Break of Structure) and CHOCH (Change of Character) pattern recognition
- **Filter**: Confidence threshold ≥ 0.7 (A+ quality)
- **Alignment**: Must align with 1H trend direction

### 3. 1M Timeframe - Momentum Confirmation
- **Purpose**: Confirm immediate momentum before execution
- **Method**: Candle color validation
- **BUY Signals**: Wait for green (bullish) 1M candle
- **SELL Signals**: Wait for red (bearish) 1M candle

## Risk Management

### Position Sizing
- **Risk per Trade**: 2% of account balance
- **Stop Loss**: 20 pips
- **Risk-Reward Ratio**: 1:2 (40 pips take profit)
- **Position Size**: Calculated based on risk amount and stop loss distance

### Exit Conditions
- **Stop Loss**: Automatic exit at 20 pip loss
- **Take Profit**: Automatic exit at 40 pip profit
- **Manual Exit**: Available for discretionary management

## Implementation Details

### Core Components

#### 1. MultiTimeframeTradingExecutor
The main class that orchestrates the entire strategy:

```python
executor = MultiTimeframeTradingExecutor(
    symbol="EURUSD",
    risk_per_trade=0.02,        # 2% risk per trade
    stop_loss_pips=20.0,        # 20 pips stop loss
    risk_reward_ratio=2.0,      # 1:2 risk-reward ratio
    confidence_threshold=0.7,   # A+ entry threshold
    pip_value=0.0001            # Standard pip value for major pairs
)
```

#### 2. TradeSignal
Represents a complete trade signal with all confirmations:

```python
@dataclass
class TradeSignal:
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
```

#### 3. TradeExecution
Represents an executed trade with full lifecycle management:

```python
@dataclass
class TradeExecution:
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
```

### Key Methods

#### analyze_1h_trend()
Analyzes trend in 1H timeframe using swing point analysis:

```python
def analyze_1h_trend(self, data_1h: pd.DataFrame) -> str:
    """
    Analyze trend in 1H timeframe using swing point analysis
    
    Returns:
        Trend direction: "uptrend", "downtrend", or "sideways"
    """
```

#### find_a_plus_entries_15m()
Finds A+ quality entries in 15M timeframe that align with 1H trend:

```python
def find_a_plus_entries_15m(self, data_15m: pd.DataFrame, trend_1h: str) -> List[MarketEvent]:
    """
    Find A+ quality entries in 15M timeframe that align with 1H trend
    
    Returns:
        List of high-quality market events
    """
```

#### confirm_1m_signal()
Confirms signal with 1M candle confirmation:

```python
def confirm_1m_signal(self, data_1m: pd.DataFrame, signal_direction: str, entry_time: pd.Timestamp) -> bool:
    """
    Confirm signal with 1M candle confirmation
    
    Returns:
        True if confirmation is valid, False otherwise
    """
```

#### execute_trade()
Executes a trade after 1M confirmation:

```python
def execute_trade(self, signal: TradeSignal, account_balance: float, data_1m: pd.DataFrame) -> Optional[TradeExecution]:
    """
    Execute a trade after 1M confirmation
    
    Returns:
        Executed trade or None if execution fails
    """
```

## Usage Examples

### Basic Strategy Execution

```python
from core.trading_executor import MultiTimeframeTradingExecutor

# Initialize executor
executor = MultiTimeframeTradingExecutor(
    symbol="EURUSD",
    risk_per_trade=0.02,
    stop_loss_pips=20.0,
    risk_reward_ratio=2.0
)

# Run strategy on historical data
results = executor.run_strategy("data/EURUSD_M1.csv", days_back=30)

# Display results
print(f"Signals Generated: {results['signals_generated']}")
print(f"Trades Executed: {results['trades_executed']}")
print(f"Total P&L: ${results['total_pnl']:.2f}")
```

### Custom Configuration

```python
# Custom risk parameters
executor = MultiTimeframeTradingExecutor(
    symbol="GBPUSD",
    risk_per_trade=0.01,        # 1% risk per trade
    stop_loss_pips=15.0,        # 15 pips stop loss
    risk_reward_ratio=3.0,      # 1:3 risk-reward ratio
    confidence_threshold=0.8,   # Higher confidence threshold
    pip_value=0.0001
)
```

### Individual Component Testing

```python
# Test trend analysis
trend_1h = executor.analyze_1h_trend(data_1h)
print(f"1H Trend: {trend_1h}")

# Test A+ entry detection
a_plus_events = executor.find_a_plus_entries_15m(data_15m, trend_1h)
print(f"Found {len(a_plus_events)} A+ entries")

# Test 1M confirmation
is_confirmed = executor.confirm_1m_signal(data_1m, "BUY", entry_time)
print(f"1M Confirmation: {is_confirmed}")
```

## Data Requirements

### Required Timeframes
- **1H**: For trend analysis (minimum 50 candles)
- **15M**: For entry detection (minimum 20 candles)
- **1M**: For momentum confirmation (minimum 20 candles)

### Data Format
CSV files with the following columns:
- `datetime`: Timestamp in UTC
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Volume (optional)

### File Naming Convention
```
data/SYMBOL_TIMEFRAME.csv
Examples:
- data/EURUSD_M1.csv
- data/GBPUSD_1H.csv
- data/USDJPY_15M.csv
```

## Performance Metrics

### Key Performance Indicators
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Average Win**: Average profit per winning trade
- **Average Loss**: Average loss per losing trade
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return measure

### Risk Metrics
- **Risk per Trade**: 2% of account balance
- **Maximum Risk**: 6% total risk (3 concurrent trades)
- **Risk-Reward Ratio**: 1:2 minimum
- **Stop Loss**: 20 pips maximum

## Live Trading Considerations

### Real-Time Implementation
1. **Data Feed**: Connect to live market data provider
2. **Order Execution**: Integrate with broker API
3. **Risk Monitoring**: Real-time position and P&L tracking
4. **Alert System**: Notifications for signals and executions

### Risk Controls
1. **Daily Loss Limit**: Maximum daily loss threshold
2. **Position Limits**: Maximum concurrent positions
3. **Correlation Limits**: Avoid highly correlated pairs
4. **News Filter**: Avoid trading during major news events

### Monitoring and Maintenance
1. **Performance Review**: Weekly strategy performance analysis
2. **Parameter Optimization**: Monthly parameter fine-tuning
3. **Market Regime Detection**: Adapt to changing market conditions
4. **Backup Systems**: Redundant execution and monitoring

## Troubleshooting

### Common Issues

#### 1. No Signals Generated
- Check data quality and timeframe availability
- Verify confidence threshold settings
- Ensure sufficient historical data for analysis

#### 2. Low Win Rate
- Review trend alignment logic
- Adjust confidence threshold
- Check stop loss and take profit levels

#### 3. High Drawdown
- Reduce risk per trade
- Implement tighter stop losses
- Add correlation filters

### Debug Mode
Enable debug output for detailed analysis:

```python
# Enable debug mode in market analyzer
executor.market_analyzer.confidence_threshold = 0.5  # Lower threshold for more signals
```

## Future Enhancements

### Planned Features
1. **Machine Learning Integration**: Pattern recognition using ML models
2. **Multi-Asset Support**: Extend to stocks, commodities, and crypto
3. **Advanced Risk Management**: Dynamic position sizing and correlation analysis
4. **Performance Analytics**: Comprehensive reporting and visualization
5. **Automated Optimization**: Genetic algorithm parameter optimization

### Research Areas
1. **Market Regime Detection**: Automatic identification of trending vs. ranging markets
2. **Volatility Adjustment**: Dynamic stop loss based on market volatility
3. **News Impact Analysis**: Integration with economic calendar and news sentiment
4. **Cross-Timeframe Validation**: Additional timeframe confirmations

## Support and Documentation

### Additional Resources
- **API Documentation**: Complete method and class documentation
- **Example Scripts**: Ready-to-run demonstration scripts
- **Video Tutorials**: Step-by-step implementation guides
- **Community Forum**: User community and support

### Contact Information
For questions, suggestions, or support:
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides and examples
- **Email Support**: Direct technical support

---

**Note**: This strategy is designed for educational and research purposes. Always test thoroughly on historical data before live trading. Past performance does not guarantee future results. Trading involves risk, and you should only trade with capital you can afford to lose.
