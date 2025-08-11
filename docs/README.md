# 🚀 BOS/CHOCH Trading Bot Backtester

A comprehensive backtesting system for Smart Money Concepts (SMC) trading strategies, specifically designed for Break of Structure (BOS) and Change of Character (CHOCH) patterns. This system generates detailed P/L ratio charts, measures trading accuracy, and provides comprehensive performance analytics.

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Features](#core-features)
- [File Structure](#file-structure)
- [Installation & Setup](#installation--setup)
- [Quick Start Guide](#quick-start-guide)
- [Configuration Options](#configuration-options)
- [Output & Reports](#output--reports)
- [Performance Metrics](#performance-metrics)
- [Trading Strategy Details](#trading-strategy-details)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 Overview

The BOS/CHOCH Trading Bot Backtester is a sophisticated system that:

- **Identifies A+ Entry Signals**: Automatically detects high-confidence BOS and CHOCH patterns using advanced market structure analysis
- **Simulates Real Trading**: Executes trades with realistic risk management, position sizing, and exit strategies
- **Generates Comprehensive Reports**: Creates interactive HTML charts showing P/L ratios, accuracy metrics, and performance analytics
- **Provides Risk Management**: Implements ATR-based stop-losses, position sizing, and drawdown protection
- **Supports Multiple Strategies**: Offers conservative, aggressive, and balanced trading presets

## 🏗️ System Architecture

The system is built with a modular architecture consisting of:

- **Core Analysis Engine**: Market structure detection and BOS/CHOCH pattern recognition
- **Backtesting Engine**: Trade simulation with realistic market conditions
- **Visualization System**: Interactive Plotly-based charts and dashboards
- **Configuration Management**: Centralized parameter control with strategy presets
- **Data Processing**: Efficient handling of multi-timeframe market data

## ⭐ Core Features

### 🎯 A+ Entry Detection
- **Confidence Scoring**: Multi-factor analysis for entry quality assessment
- **Pattern Recognition**: Automatic BOS and CHOCH pattern identification
- **Trend Alignment**: Ensures entries align with overall market direction
- **Filtering System**: Configurable thresholds for signal quality

### 💼 Risk Management
- **ATR-Based Stop Losses**: Dynamic stop-loss placement using Average True Range
- **Position Sizing**: Risk-based position calculation (default: 2% risk per trade)
- **Reward/Risk Ratios**: Configurable profit targets (default: 2:1)
- **Drawdown Protection**: Maximum loss limits and correlation controls

### 📊 Performance Analytics
- **Win Rate Analysis**: Detailed breakdown of winning vs. losing trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Sharpe Ratio**: Risk-adjusted return measurement
- **Maximum Drawdown**: Peak-to-trough capital decline analysis
- **Monthly Returns**: Performance breakdown by time periods

### 📈 Visualization & Reporting
- **Interactive Charts**: Plotly-based HTML visualizations
- **Comprehensive Dashboard**: All metrics in a single view
- **P/L Analysis**: Cumulative returns and trade-by-trade performance
- **Entry Type Comparison**: BOS vs CHOCH performance analysis
- **Risk Metrics**: Drawdown curves and equity analysis

## 📁 File Structure

```
Trading Bot/
├── 📊 Core Analysis Files
│   ├── core/
│   │   ├── smart_money_concepts.py      # BOS/CHOCH pattern detection
│   │   ├── structure_builder.py          # Market structure analysis
│   │   └── data_loader.py               # Data loading and resampling
│   └── utils/
│       ├── level_plotter.py             # Support/resistance visualization
│       ├── pattern_plotter.py           # Pattern charting
│       ├── candlestick_plotter.py       # Price action charts
│       ├── structure_trend_plotter.py   # Trend analysis charts
│       └── trend_plotter.py             # Trend visualization
│
├── 🧪 Testing & Examples
│   ├── test_bos_choch_plot.py          # Example usage and testing
│   ├── test_backtester.py              # Backtester functionality tests
│   └── test_structure_builder.py       # Structure analysis tests
│
├── ⚙️ Configuration & Main System
│   ├── backtester.py                    # Main backtesting engine
│   ├── backtester_config.py            # Configuration management
│   └── requirements_backtester.txt     # Python dependencies
│
├── 📈 Generated Reports
│   └── generated_backtest_plots/       # HTML charts and dashboards
│
└── 📚 Documentation
    ├── README.md                        # This comprehensive guide
    └── BACKTESTER_README.md            # Detailed backtester documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+ 
- Windows 10/11 (tested on Windows 10.0.26100)
- 4GB+ RAM recommended
- Internet connection for data loading

### Installation Steps

1. **Clone/Download the Project**
   ```bash
   # Navigate to your desired directory
   cd "D:\Code\Trading Bot"
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements_backtester.txt
   ```

3. **Verify Installation**
   ```bash
   python test_backtester.py
   ```

### Required Dependencies
```
pandas>=1.5.0          # Data manipulation and analysis
numpy>=1.21.0          # Numerical computing
plotly>=5.0.0          # Interactive charting
scipy>=1.9.0           # Scientific computing
scikit-learn>=1.1.0    # Machine learning utilities
matplotlib>=3.5.0      # Basic plotting
seaborn>=0.11.0        # Statistical visualization
```

## 🎯 Quick Start Guide

### 1. Basic Backtest Run
```bash
python backtester.py
```

This will:
- Load 60 days of XAUUSD H1 data
- Detect A+ entries (confidence ≥ 70%)
- Execute simulated trades with risk management
- Generate comprehensive HTML reports
- Open the main dashboard automatically

### 2. Custom Configuration
```python
from backtester_config import get_config

# Load aggressive strategy
config = get_config("aggressive")

# Modify specific parameters
config['trading']['risk_per_trade'] = 0.05  # 5% risk per trade
config['trading']['stop_loss_atr_multiplier'] = 3.0  # Wider stops
```

### 3. Data Analysis Only
```python
from test_bos_choch_plot import main
main()  # Run pattern detection without backtesting
```

## ⚙️ Configuration Options

### Trading Parameters (`backtester_config.py`)

#### Strategy Presets
- **Conservative**: 1% risk, 1.5:1 reward/risk, 80% confidence threshold
- **Balanced**: 2% risk, 2:1 reward/risk, 70% confidence threshold  
- **Aggressive**: 5% risk, 3:1 reward/risk, 60% confidence threshold

#### Risk Management
```python
TRADING_CONFIG = {
    "initial_capital": 10000,        # Starting capital in USD
    "risk_per_trade": 0.02,         # 2% risk per trade
    "reward_risk_ratio": 2.0,       # 2:1 reward to risk ratio
    "max_trade_duration": 48,       # Maximum trade duration in hours
    "confidence_threshold": 0.7,    # Minimum confidence for A+ entries
    "stop_loss_atr_multiplier": 2.0, # ATR multiplier for stop loss
}
```

#### Data Configuration
```python
DATA_CONFIG = {
    "symbol": "XAUUSD_H1.csv",      # Symbol to backtest
    "days_back": 60,                # Number of days of historical data
    "timeframe": "1H",              # Timeframe for analysis
}
```

#### Advanced Settings
```python
ADVANCED_CONFIG = {
    "slippage": 0.0001,             # Slippage per trade (1 pip)
    "commission": 0.0001,           # Commission per trade (1 pip)
    "spread": 0.0002,               # Spread cost (2 pips)
    "fill_rate": 0.95,              # Fill rate assumption (95%)
}
```

## 📊 Output & Reports

### Generated Files
The system creates a `generated_backtest_plots/` directory containing:

1. **`comprehensive_dashboard.html`** - Main dashboard with all metrics
2. **`pl_ratio_analysis.html`** - P/L ratio charts and analysis
3. **`trading_accuracy.html`** - Win rate and performance metrics
4. **`monthly_returns.html`** - Monthly performance heatmap
5. **`entry_type_analysis.html`** - BOS vs CHOCH comparison

### Dashboard Sections
- **Equity Curve**: Account balance over time
- **Drawdown Analysis**: Peak-to-trough capital decline
- **Trade Distribution**: P&L histogram and statistics
- **Monthly Returns**: Performance by month (heatmap)
- **Entry Type Performance**: BOS vs CHOCH success rates
- **Performance Summary**: Overall win rate gauge

## 📈 Performance Metrics

### Key Performance Indicators (KPIs)

#### Profitability Metrics
- **Total P&L**: Absolute dollar gain/loss
- **Total Return**: Percentage return on initial capital
- **Profit Factor**: Ratio of gross profits to gross losses
- **Average Win/Loss**: Mean performance of winning/losing trades

#### Risk Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return measure
- **Win Rate**: Percentage of profitable trades
- **Average Trade Duration**: Mean time in trades

#### Trade Analysis
- **Total Trades**: Number of completed trades
- **Winning Trades**: Count of profitable trades
- **Losing Trades**: Count of unprofitable trades
- **Largest Win/Loss**: Best and worst individual trades

### Calculation Methods

#### Sharpe Ratio
```
Sharpe Ratio = √252 × (Mean Return / Standard Deviation)
```
- Annualized using 252 trading days
- Higher values indicate better risk-adjusted returns

#### Profit Factor
```
Profit Factor = |Sum of Wins| / |Sum of Losses|
```
- Values > 1 indicate profitable strategy
- Values > 2 indicate excellent performance

#### Maximum Drawdown
```
Max Drawdown = (Peak Value - Trough Value) / Peak Value
```
- Expressed as percentage
- Lower values indicate better capital preservation

## 🎯 Trading Strategy Details

### A+ Entry Criteria

#### Confidence Thresholds
- **Minimum Confidence**: 70% (configurable)
- **Pattern Quality**: Based on multiple technical factors
- **Trend Alignment**: Must align with overall market direction
- **Volume Confirmation**: Optional volume-based filtering

#### BOS (Break of Structure) Entries
- **Bullish BOS**: Higher High (HH) breaking previous resistance
- **Bearish BOS**: Lower Low (LL) breaking previous support
- **Confidence Factors**: 
  - Strength of break
  - Volume confirmation
  - Previous structure quality
  - Market context alignment

#### CHOCH (Change of Character) Entries
- **Bullish CHOCH**: Break above downtrend resistance
- **Bearish CHOCH**: Break below uptrend support
- **Confidence Factors**:
  - Trend strength before change
  - Break magnitude
  - Support/resistance quality
  - Market momentum shift

### Risk Management System

#### Stop Loss Placement
```
Long Trade: Stop Loss = Entry Price - (ATR × Multiplier)
Short Trade: Stop Loss = Entry Price + (ATR × Multiplier)
```
- **ATR Period**: 14 periods (configurable)
- **Default Multiplier**: 2.0 (configurable)
- **Dynamic Adjustment**: Based on market volatility

#### Position Sizing
```
Position Size = Risk Amount / (Entry Price - Stop Loss)
Risk Amount = Capital × Risk Per Trade
```
- **Default Risk**: 2% of capital per trade
- **Maximum Risk**: 5% per trade (aggressive preset)
- **Correlation Limits**: Prevents over-concentration

#### Take Profit Targets
```
Take Profit = Entry Price ± (Stop Loss Distance × Reward/Risk Ratio)
```
- **Default Ratio**: 2:1 reward to risk
- **Conservative**: 1.5:1 ratio
- **Aggressive**: 3:1 ratio

### Exit Conditions

#### Automatic Exits
1. **Stop Loss Hit**: Price reaches stop loss level
2. **Take Profit Hit**: Price reaches profit target
3. **Time Exit**: Maximum trade duration exceeded
4. **End of Data**: Close remaining trades at final price

#### Exit Reasons Tracking
- **SL**: Stop loss triggered
- **TP**: Take profit reached
- **Time**: Maximum duration exceeded
- **End of Data**: Backtest period ended

## 🔧 API Reference

### Main Classes

#### `BOSCHOCHBacktester`
Main backtesting engine class.

```python
class BOSCHOCHBacktester:
    def __init__(self, 
                 initial_capital: float = 10000,
                 risk_per_trade: float = 0.02,
                 reward_risk_ratio: float = 2.0,
                 max_trade_duration: int = 48,
                 confidence_threshold: float = 0.7,
                 stop_loss_atr_multiplier: float = 2.0):
```

**Key Methods:**
- `run_backtest(data, events)`: Execute complete backtest
- `calculate_statistics()`: Generate performance metrics
- `execute_trade(event, data, index)`: Execute individual trade
- `check_exit_conditions(data, index)`: Monitor exit conditions

#### `BacktestVisualizer`
Creates interactive charts and visualizations.

```python
class BacktestVisualizer:
    def __init__(self, backtest_result: BacktestResult):
```

**Key Methods:**
- `create_pl_ratio_chart()`: P/L analysis dashboard
- `create_accuracy_chart()`: Performance metrics
- `create_monthly_returns_chart()`: Monthly performance heatmap
- `create_entry_type_analysis()`: BOS vs CHOCH comparison

#### `MarketStructureAnalyzer`
Detects BOS and CHOCH patterns.

```python
class MarketStructureAnalyzer:
    def __init__(self, confidence_threshold: float = 0.7):
```

**Key Methods:**
- `get_market_events(structure)`: Detect market events
- `_get_trend_state(structure, index)`: Analyze trend context
- `_calculate_confidence(event, structure)`: Score event confidence

### Data Structures

#### `Trade` Dataclass
```python
@dataclass
class Trade:
    entry_time: pd.Timestamp      # Trade entry timestamp
    entry_price: float            # Entry price
    exit_time: pd.Timestamp       # Trade exit timestamp
    exit_price: float             # Exit price
    direction: str                # "Long" or "Short"
    entry_type: str               # "BOS" or "CHOCH"
    confidence: float             # Entry confidence (0.0-1.0)
    pnl: float                    # Profit/Loss amount
    pnl_pct: float                # P&L percentage
    duration: timedelta           # Time in trade
    stop_loss: float              # Stop loss price
    take_profit: float            # Take profit price
    exit_reason: str              # Exit reason
```

#### `BacktestResult` Dataclass
```python
@dataclass
class BacktestResult:
    trades: List[Trade]           # List of all trades
    total_trades: int             # Total number of trades
    winning_trades: int           # Number of winning trades
    losing_trades: int            # Number of losing trades
    win_rate: float               # Win rate (0.0-1.0)
    total_pnl: float              # Total P&L
    total_pnl_pct: float          # Total return percentage
    max_drawdown: float           # Maximum drawdown
    max_drawdown_pct: float       # Maximum drawdown percentage
    sharpe_ratio: float           # Sharpe ratio
    profit_factor: float          # Profit factor
    equity_curve: pd.Series       # Account balance over time
    drawdown_curve: pd.Series     # Drawdown over time
```

### Configuration Functions

#### `get_config(preset_name)`
```python
def get_config(preset_name: str = None) -> dict:
    """Get configuration with optional preset override"""
```

**Parameters:**
- `preset_name`: Strategy preset ("conservative", "balanced", "aggressive")

**Returns:**
- Complete configuration dictionary with all parameters

#### `print_config_summary(config)`
```python
def print_config_summary(config: dict):
    """Print a summary of the current configuration"""
```

## 🚨 Troubleshooting

### Common Issues & Solutions

#### 1. Import Errors
**Problem**: Module not found errors
**Solution**: Ensure all dependencies are installed
```bash
pip install -r requirements_backtester.txt
```

#### 2. Data Loading Issues
**Problem**: No data loaded for timeframe
**Solution**: Check data file exists and format
```bash
# Verify data file exists
ls data/XAUUSD_H1.csv

# Check file format (should have OHLCV columns)
head -5 data/XAUUSD_H1.csv
```

#### 3. Chart Generation Errors
**Problem**: HTML files not created
**Solution**: Check write permissions and disk space
```bash
# Check directory permissions
dir generated_backtest_plots

# Verify disk space
df -h
```

#### 4. Performance Issues
**Problem**: Slow backtesting execution
**Solution**: Reduce data period or optimize parameters
```python
# Reduce data period
config['data']['days_back'] = 30  # Instead of 60

# Increase confidence threshold
config['trading']['confidence_threshold'] = 0.8  # Fewer trades
```

### Debug Mode

Enable debug output by modifying the main function:

```python
# Add debug prints
print(f"Debug: Processing candle {i}/{len(data)}")
print(f"Debug: Open trades: {len(self.open_trades)}")
print(f"Debug: Current equity: ${current_equity:,.2f}")
```

### Logging

The system includes built-in logging capabilities:

```python
LOGGING_CONFIG = {
    "log_level": "INFO",             # DEBUG, INFO, WARNING, ERROR
    "log_to_file": True,             # Log to file
    "log_to_console": True,          # Log to console
    "log_file": "backtester.log",    # Log file name
}
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints for all function parameters
- Add docstrings for all classes and methods
- Include error handling for edge cases

### Testing
Run the test suite before submitting changes:
```bash
python test_backtester.py
python test_structure_builder.py
python -m pytest tests/  # If using pytest
```

## 📞 Support & Contact

### Getting Help
1. **Check Documentation**: Review this README and BACKTESTER_README.md
2. **Run Tests**: Verify system functionality with test files
3. **Check Logs**: Review backtester.log for error details
4. **Review Configuration**: Ensure parameters are set correctly

### Feature Requests
- Submit detailed feature requests with use cases
- Include example data or scenarios
- Specify expected behavior and outputs

### Bug Reports
When reporting bugs, include:
- Python version and OS details
- Complete error messages and stack traces
- Steps to reproduce the issue
- Sample data if applicable

## 📄 License

This project is provided as-is for educational and research purposes. Use at your own risk in live trading environments.

## 🔮 Future Enhancements

### Planned Features
- **Multi-Asset Support**: Backtest multiple symbols simultaneously
- **Portfolio Optimization**: Risk parity and correlation analysis
- **Machine Learning Integration**: AI-powered entry/exit signals
- **Real-Time Trading**: Live market connection and execution
- **Advanced Analytics**: Monte Carlo simulations and stress testing

### Performance Improvements
- **Parallel Processing**: Multi-threaded backtesting for large datasets
- **Memory Optimization**: Efficient data handling for long timeframes
- **Caching System**: Store and reuse calculation results
- **Cloud Integration**: Distributed backtesting on cloud platforms

---

**🎯 Happy Trading!** 

Remember: Past performance does not guarantee future results. Always use proper risk management and never risk more than you can afford to lose.
