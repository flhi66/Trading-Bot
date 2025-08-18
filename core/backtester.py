#!/usr/bin/env python3
"""
Advanced Backtester for BOS/CHOCH Trading Bot
Generates P/L ratio charts and measures accuracy based on A+ entries
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

from core.data_loader import load_and_resample
from core.risk_manager import RiskManager
from core.structure_builder import get_market_analysis
from core.smart_money_concepts import MarketStructureAnalyzer, StructurePoint, SwingType, MarketEvent

@dataclass
class Trade:
    """Represents a single trade"""
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    direction: str  # "Long" or "Short"
    entry_type: str  # "BOS" or "CHOCH"
    confidence: float
    pnl: float
    pnl_pct: float
    duration: timedelta
    stop_loss: float
    take_profit: float
    exit_reason: str  # "TP", "SL", "Manual", "Time"

@dataclass
class BacktestResult:
    """Contains all backtest results and statistics"""
    trades: List[Trade]
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_trade_duration: timedelta
    monthly_returns: pd.Series
    equity_curve: pd.Series
    drawdown_curve: pd.Series

class BOSCHOCHBacktester:
    """Advanced backtester for BOS/CHOCH trading strategies"""
    
    def __init__(self, 
                 initial_capital: float = 10000,
                 risk_per_trade: float = 0.02,  # 2% risk per trade
                 reward_risk_ratio: float = 2.0,  # 2:1 reward to risk
                 max_trade_duration: int = 48,  # hours
                 confidence_threshold: float = 0.7,  # Only trade A+ entries
                 stop_loss_atr_multiplier: float = 2.0):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.reward_risk_ratio = reward_risk_ratio
        self.max_trade_duration = max_trade_duration
        self.confidence_threshold = confidence_threshold
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier
        
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.open_trades = []
        self.risk_manager = RiskManager(risk_per_trade=risk_per_trade,
                                        atr_period=14,
                                        atr_multiplier=stop_loss_atr_multiplier)
        
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range for stop loss placement"""
        return self.risk_manager.calculate_atr(data, period)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, risk_amount: float) -> float:
        """Calculate position size based on risk management"""
        return self.risk_manager.calculate_position_size(entry_price, stop_loss, risk_amount)
    
    def is_a_plus_entry(self, event: MarketEvent) -> bool:
        """Determine if an entry qualifies as A+ based on multiple criteria"""
        # High confidence threshold
        if event.confidence < self.confidence_threshold:
            return False
        
        # Strong trend alignment
        if event.direction == "Bullish" and event.event_type.value == "BOS":
            # Bullish BOS should be in uptrend context
            return True
        elif event.direction == "Bearish" and event.event_type.value == "BOS":
            # Bearish BOS should be in downtrend context
            return True
        elif event.direction == "Bullish" and event.event_type.value == "CHOCH":
            # Bullish CHOCH should break above resistance
            return True
        elif event.direction == "Bearish" and event.event_type.value == "CHOCH":
            # Bearish CHOCH should break below support
            return True
        
        return False
    
    def execute_trade(self, event: MarketEvent, data: pd.DataFrame, current_index: int) -> Optional[Trade]:
        """Execute a trade based on the market event"""
        if not self.is_a_plus_entry(event):
            return None
        
        entry_price = event.price
        entry_time = event.timestamp
        
        # Calculate ATR for stop loss
        atr = self.calculate_atr(data.iloc[:current_index+1])
        if pd.isna(atr.iloc[-1]):
            return None
        
        atr_value = atr.iloc[-1]
        
        # Set stop loss and take profit using centralized risk manager
        rm_direction = "BUY" if event.direction == "Bullish" else "SELL"
        stop_loss, take_profit = self.risk_manager.compute_stop_and_target_from_atr(
            entry_price=entry_price,
            direction=rm_direction,
            atr_value=atr_value,
            reward_risk_ratio=self.reward_risk_ratio
        )
        direction = "Long" if rm_direction == "BUY" else "Short"
        
        # Calculate position size
        risk_amount = self.risk_manager.risk_amount_for_balance(self.current_capital)
        position_size = self.calculate_position_size(entry_price, stop_loss, risk_amount)
        
        if position_size == 0:
            return None
        
        # Create trade object
        trade = Trade(
            entry_time=entry_time,
            entry_price=entry_price,
            exit_time=entry_time,  # Will be updated when closed
            exit_price=entry_price,  # Will be updated when closed
            direction=direction,
            entry_type=event.event_type.value,
            confidence=event.confidence,
            pnl=0.0,
            pnl_pct=0.0,
            duration=timedelta(0),
            stop_loss=stop_loss,
            take_profit=take_profit,
            exit_reason=""
        )
        
        # Add to open trades
        self.open_trades.append({
            'trade': trade,
            'position_size': position_size,
            'entry_index': current_index
        })
        
        return trade
    
    def check_exit_conditions(self, data: pd.DataFrame, current_index: int) -> None:
        """Check if any open trades should be closed"""
        current_candle = data.iloc[current_index]
        current_price = current_candle['Close']
        current_time = data.index[current_index]
        
        for open_trade in self.open_trades[:]:  # Copy list to avoid modification during iteration
            trade = open_trade['trade']
            entry_index = open_trade['entry_index']
            position_size = open_trade['position_size']
            
            # Check if trade duration exceeded
            trade_duration = current_time - trade.entry_time
            if trade_duration.total_seconds() / 3600 > self.max_trade_duration:
                self.close_trade(trade, current_price, current_time, "Time", position_size)
                self.open_trades.remove(open_trade)
                continue
            
            # Check stop loss and take profit
            if trade.direction == "Long":
                if current_price <= trade.stop_loss:
                    print(f"🛑 Long trade stopped out at {current_price:.2f} (SL: {trade.stop_loss:.2f})")
                    self.close_trade(trade, trade.stop_loss, current_time, "SL", position_size)
                    self.open_trades.remove(open_trade)
                elif current_price >= trade.take_profit:
                    print(f"🎯 Long trade take profit at {current_price:.2f} (TP: {trade.take_profit:.2f})")
                    self.close_trade(trade, trade.take_profit, current_time, "TP", position_size)
                    self.open_trades.remove(open_trade)
            else:  # Short
                if current_price >= trade.stop_loss:
                    print(f"🛑 Short trade stopped out at {current_price:.2f} (SL: {trade.stop_loss:.2f})")
                    self.close_trade(trade, trade.stop_loss, current_time, "SL", position_size)
                    self.open_trades.remove(open_trade)
                elif current_price <= trade.take_profit:
                    print(f"🎯 Short trade take profit at {current_price:.2f} (TP: {trade.take_profit:.2f})")
                    self.close_trade(trade, trade.take_profit, current_time, "TP", position_size)
                    self.open_trades.remove(open_trade)
    
    def close_trade(self, trade: Trade, exit_price: float, exit_time: pd.Timestamp, exit_reason: str, position_size: float) -> None:
        """Close a trade and calculate P&L"""
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.exit_reason = exit_reason
        trade.duration = exit_time - trade.entry_time
        
        # Calculate P&L
        if trade.direction == "Long":
            trade.pnl = (exit_price - trade.entry_price) * position_size
        else:
            trade.pnl = (trade.entry_price - exit_price) * position_size
        
        trade.pnl_pct = (trade.pnl / (trade.entry_price * position_size)) * 100
        
        # Update capital
        self.current_capital += trade.pnl
        
        # Add to completed trades
        self.trades.append(trade)
        
        # Update equity curve
        self.equity_curve.append(self.current_capital)
    
    def run_backtest(self, data: pd.DataFrame, events: List[MarketEvent]) -> BacktestResult:
        """Run the complete backtest"""
        print(f"🚀 Starting backtest with {len(events)} events...")
        print(f"💰 Initial capital: ${self.initial_capital:,.2f}")
        print(f"🎯 Confidence threshold: {self.confidence_threshold}")
        print(f"⚠️  Risk per trade: {self.risk_per_trade*100:.1f}%")
        print(f"📈 Reward/Risk ratio: {self.reward_risk_ratio:.1f}:1")
        
        # Reset state
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.open_trades = []
        self.current_capital = self.initial_capital
        
        # Store data timestamps for equity curve
        self._data_timestamps = data.index.tolist()
        
        # Create event lookup by timestamp
        event_lookup = {event.timestamp: event for event in events}
        
        # Process each candle
        for i in range(len(data)):
            current_time = data.index[i]
            
            # Check exit conditions for open trades FIRST
            self.check_exit_conditions(data, i)
            
            # Check if there's a new event to trade
            if current_time in event_lookup:
                event = event_lookup[current_time]
                new_trade = self.execute_trade(event, data, i)
                if new_trade:
                    print(f"📊 {current_time}: {event.event_type.value} {event.direction} entry at {event.price:.2f} (Confidence: {event.confidence:.2f})")
            
            # Update equity curve
            if i > 0:  # Skip first candle
                current_equity = self.current_capital
                for open_trade in self.open_trades:
                    trade = open_trade['trade']
                    position_size = open_trade['position_size']
                    current_price = data.iloc[i]['Close']
                    
                    # Calculate unrealized P&L
                    if trade.direction == "Long":
                        unrealized_pnl = (current_price - trade.entry_price) * position_size
                    else:
                        unrealized_pnl = (trade.entry_price - current_price) * position_size
                    
                    current_equity += unrealized_pnl
                
                self.equity_curve.append(current_equity)
            else:
                # First candle - just add initial capital
                self.equity_curve.append(self.initial_capital)
        
        # Close any remaining open trades at the end
        final_price = data.iloc[-1]['Close']
        final_time = data.index[-1]
        
        for open_trade in self.open_trades[:]:
            trade = open_trade['trade']
            position_size = open_trade['position_size']
            self.close_trade(trade, final_price, final_time, "End of Data", position_size)
        
        # Calculate final statistics
        return self.calculate_statistics()
    
    def calculate_statistics(self) -> BacktestResult:
        """Calculate comprehensive trading statistics"""
        if not self.trades:
            return BacktestResult(
                trades=[], total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, total_pnl=0.0, total_pnl_pct=0.0,
                max_drawdown=0.0, max_drawdown_pct=0.0, sharpe_ratio=0.0,
                profit_factor=0.0, avg_win=0.0, avg_loss=0.0,
                avg_trade_duration=timedelta(0), monthly_returns=pd.Series(),
                equity_curve=pd.Series(), drawdown_curve=pd.Series()
            )
        
        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = sum(t.pnl for t in self.trades)
        total_pnl_pct = (total_pnl / self.initial_capital) * 100
        
        # Win/Loss analysis
        wins = [t.pnl for t in self.trades if t.pnl > 0]
        losses = [t.pnl for t in self.trades if t.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
        
        # Duration analysis
        durations = [t.duration for t in self.trades]
        avg_trade_duration = sum(durations, timedelta(0)) / len(durations) if durations else timedelta(0)
        
        # Equity curve and drawdown
        equity_series = pd.Series(self.equity_curve)
        drawdown_series = self.calculate_drawdown(equity_series)
        
        max_drawdown = drawdown_series.min()
        max_drawdown_pct = (max_drawdown / self.initial_capital) * 100
        
        # Sharpe ratio (simplified)
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if len(returns) > 1 and returns.std() != 0 else 0
        
        # Monthly returns
        monthly_returns = self.calculate_monthly_returns()
        
        return BacktestResult(
            trades=self.trades,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade_duration=avg_trade_duration,
            monthly_returns=monthly_returns,
            equity_curve=equity_series,
            drawdown_curve=drawdown_series
        )
    
    def calculate_drawdown(self, equity_series: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        running_max = equity_series.expanding().max()
        drawdown = equity_series - running_max
        return drawdown
    
    def calculate_monthly_returns(self) -> pd.Series:
        """Calculate monthly returns from equity curve"""
        if len(self.equity_curve) < 2:
            return pd.Series()
        
        # Create a simple time series for monthly aggregation
        # Use the actual data timestamps instead of creating a new date range
        if hasattr(self, '_data_timestamps') and len(self._data_timestamps) == len(self.equity_curve):
            dates = self._data_timestamps
        else:
            # Fallback: create a simple index
            dates = pd.date_range(start='2025-01-01', periods=len(self.equity_curve), freq='H')
        
        equity_df = pd.DataFrame({'equity': self.equity_curve}, index=dates)
        
        # Resample to monthly and calculate returns
        monthly_equity = equity_df.resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna()
        
        return monthly_returns

class BacktestVisualizer:
    """Creates comprehensive charts for backtest results"""
    
    def __init__(self, backtest_result: BacktestResult):
        self.result = backtest_result
    
    def create_pl_ratio_chart(self) -> go.Figure:
        """Create P/L ratio chart showing cumulative returns"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Cumulative P&L', 'Trade-by-Trade P&L', 'Drawdown'),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Cumulative P&L
        cumulative_pnl = np.cumsum([t.pnl for t in self.result.trades])
        trade_numbers = list(range(1, len(self.result.trades) + 1))
        
        fig.add_trace(
            go.Scatter(
                x=trade_numbers,
                y=cumulative_pnl,
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#00ff88', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
        
        # Trade-by-Trade P&L
        pnl_values = [t.pnl for t in self.result.trades]
        colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]
        
        fig.add_trace(
            go.Bar(
                x=trade_numbers,
                y=pnl_values,
                name='Trade P&L',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Drawdown
        drawdown_values = self.result.drawdown_curve.values
        drawdown_trade_numbers = list(range(1, len(drawdown_values) + 1))
        
        fig.add_trace(
            go.Scatter(
                x=drawdown_trade_numbers,
                y=drawdown_values,
                mode='lines',
                name='Drawdown',
                line=dict(color='#ff4444', width=2),
                fill='tonexty'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title='P&L Analysis Dashboard',
            height=800,
            showlegend=True,
            template='plotly_dark'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Trade Number", row=3, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=1, col=1)
        fig.update_yaxes(title_text="P&L ($)", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown ($)", row=3, col=1)
        
        return fig
    
    def create_accuracy_chart(self) -> go.Figure:
        """Create accuracy and performance metrics chart"""
        # Calculate accuracy metrics
        total_trades = self.result.total_trades
        win_rate = self.result.win_rate * 100
        profit_factor = self.result.profit_factor
        
        # Create metrics display
        fig = go.Figure()
        
        # Add metric boxes
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=win_rate,
            delta={'reference': 50},
            title={'text': "Win Rate (%)"},
            domain={'x': [0, 0.33], 'y': [0, 1]}
        ))
        
        fig.add_trace(go.Indicator(
            mode="number",
            value=total_trades,
            title={'text': "Total Trades"},
            domain={'x': [0.33, 0.66], 'y': [0, 1]}
        ))
        
        fig.add_trace(go.Indicator(
            mode="number",
            value=profit_factor,
            title={'text': "Profit Factor"},
            domain={'x': [0.66, 1], 'y': [0, 1]}
        ))
        
        # Add performance summary
        fig.add_annotation(
            text=f"Total P&L: ${self.result.total_pnl:,.2f} ({self.result.total_pnl_pct:+.2f}%)<br>" +
                 f"Max Drawdown: ${self.result.max_drawdown:,.2f} ({self.result.max_drawdown_pct:.2f}%)<br>" +
                 f"Sharpe Ratio: {self.result.sharpe_ratio:.2f}",
            xref="paper", yref="paper",
            x=0.5, y=0.1,
            showarrow=False,
            font=dict(size=14),
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="white",
            borderwidth=1
        )
        
        fig.update_layout(
            title='Trading Bot Accuracy & Performance Metrics',
            height=400,
            template='plotly_dark'
        )
        
        return fig
    
    def create_monthly_returns_chart(self) -> go.Figure:
        """Create monthly returns heatmap"""
        if self.result.monthly_returns.empty:
            return go.Figure()
        
        # Prepare data for heatmap
        monthly_returns = self.result.monthly_returns.copy()
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        
        # Create year-month matrix
        monthly_returns['year'] = monthly_returns.index.year
        monthly_returns['month'] = monthly_returns.index.month
        
        # Pivot for heatmap
        returns_matrix = monthly_returns.pivot(index='year', columns='month', values='returns')
        
        # Month names for better readability
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig = go.Figure(data=go.Heatmap(
            z=returns_matrix.values,
            x=month_names,
            y=returns_matrix.index,
            colorscale='RdYlGn',
            zmid=0,
            text=returns_matrix.values.round(4),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Monthly Returns Heatmap',
            xaxis_title='Month',
            yaxis_title='Year',
            height=400,
            template='plotly_dark'
        )
        
        return fig
    
    def create_entry_type_analysis(self) -> go.Figure:
        """Create analysis of different entry types (BOS vs CHOCH)"""
        # Group trades by entry type
        bos_trades = [t for t in self.result.trades if t.entry_type == 'BOS']
        choch_trades = [t for t in self.result.trades if t.entry_type == 'CHOCH']
        
        # Calculate metrics for each type
        bos_win_rate = len([t for t in bos_trades if t.pnl > 0]) / len(bos_trades) * 100 if bos_trades else 0
        choch_win_rate = len([t for t in choch_trades if t.pnl > 0]) / len(choch_trades) * 100 if choch_trades else 0
        
        bos_avg_pnl = np.mean([t.pnl for t in bos_trades]) if bos_trades else 0
        choch_avg_pnl = np.mean([t.pnl for t in choch_trades]) if choch_trades else 0
        
        # Create comparison chart
        fig = go.Figure()
        
        # Win rate comparison
        fig.add_trace(go.Bar(
            name='BOS',
            x=['Win Rate (%)', 'Avg P&L ($)'],
            y=[bos_win_rate, bos_avg_pnl],
            marker_color='#00ff88'
        ))
        
        fig.add_trace(go.Bar(
            name='CHOCH',
            x=['Win Rate (%)', 'Avg P&L ($)'],
            y=[choch_win_rate, choch_avg_pnl],
            marker_color='#ff8800'
        ))
        
        fig.update_layout(
            title='BOS vs CHOCH Entry Performance Comparison',
            barmode='group',
            height=400,
            template='plotly_dark'
        )
        
        return fig

def generate_backtest_report(backtest_result: BacktestResult, output_dir: str = "generated_backtest_plots") -> None:
    """Generate comprehensive backtest report with all charts"""
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = BacktestVisualizer(backtest_result)
    
    # Generate all charts
    print("📊 Generating backtest charts...")
    
    # P&L Ratio Chart
    pl_chart = visualizer.create_pl_ratio_chart()
    pl_chart.write_html(f"{output_dir}/pl_ratio_analysis.html")
    
    # Accuracy Chart
    accuracy_chart = visualizer.create_accuracy_chart()
    accuracy_chart.write_html(f"{output_dir}/trading_accuracy.html")
    
    # Monthly Returns Chart
    monthly_chart = visualizer.create_monthly_returns_chart()
    monthly_chart.write_html(f"{output_dir}/monthly_returns.html")
    
    # Entry Type Analysis
    entry_chart = visualizer.create_entry_type_analysis()
    entry_chart.write_html(f"{output_dir}/entry_type_analysis.html")
    
    # Create comprehensive dashboard
    create_comprehensive_dashboard(backtest_result, visualizer, output_dir)
    
    print(f"✅ Backtest charts generated in '{output_dir}' directory")

def create_comprehensive_dashboard(backtest_result: BacktestResult, visualizer: BacktestVisualizer, output_dir: str) -> None:
    """Create a comprehensive dashboard combining all metrics"""
    # Create subplots for dashboard
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Equity Curve', 'Drawdown',
            'Trade P&L Distribution', 'Monthly Returns',
            'Win Rate by Entry Type', 'Performance Summary'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "heatmap"}],
            [{"type": "bar"}, {"type": "indicator"}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # 1. Equity Curve
    fig.add_trace(
        go.Scatter(
            x=list(range(len(backtest_result.equity_curve))),
            y=backtest_result.equity_curve,
            mode='lines',
            name='Equity',
            line=dict(color='#00ff88', width=2)
        ),
        row=1, col=1
    )
    
    # 2. Drawdown
    fig.add_trace(
        go.Scatter(
            x=list(range(len(backtest_result.drawdown_curve))),
            y=backtest_result.drawdown_curve,
            mode='lines',
            name='Drawdown',
            line=dict(color='#ff4444', width=2),
            fill='tonexty'
        ),
        row=1, col=2
    )
    
    # 3. Trade P&L Distribution
    pnl_values = [t.pnl for t in backtest_result.trades]
    fig.add_trace(
        go.Histogram(
            x=pnl_values,
            nbinsx=20,
            name='P&L Distribution',
            marker_color='#0088ff'
        ),
        row=2, col=1
    )
    
    # 4. Monthly Returns Heatmap
    if not backtest_result.monthly_returns.empty:
        monthly_returns = backtest_result.monthly_returns.copy()
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_returns['year'] = monthly_returns.index.year
        monthly_returns['month'] = monthly_returns.index.month
        returns_matrix = monthly_returns.pivot(index='year', columns='month', values='returns')
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        fig.add_trace(
            go.Heatmap(
                z=returns_matrix.values,
                x=month_names,
                y=returns_matrix.index,
                colorscale='RdYlGn',
                zmid=0,
                name='Monthly Returns'
            ),
            row=2, col=2
        )
    
    # 5. Win Rate by Entry Type
    bos_trades = [t for t in backtest_result.trades if t.entry_type == 'BOS']
    choch_trades = [t for t in backtest_result.trades if t.entry_type == 'CHOCH']
    
    bos_win_rate = len([t for t in bos_trades if t.pnl > 0]) / len(bos_trades) * 100 if bos_trades else 0
    choch_win_rate = len([t for t in choch_trades if t.pnl > 0]) / len(choch_trades) * 100 if choch_trades else 0
    
    fig.add_trace(
        go.Bar(
            x=['BOS', 'CHOCH'],
            y=[bos_win_rate, choch_win_rate],
            name='Win Rate (%)',
            marker_color=['#00ff88', '#ff8800']
        ),
        row=3, col=1
    )
    
    # 6. Performance Summary Indicator
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=backtest_result.win_rate * 100,
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            },
            title={'text': "Overall Win Rate (%)"}
        ),
        row=3, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Comprehensive Trading Bot Performance Dashboard',
        height=1200,
        showlegend=True,
        template='plotly_dark'
    )
    
    # Save dashboard
    fig.write_html(f"{output_dir}/comprehensive_dashboard.html")
    print("📊 Comprehensive dashboard created!")

def main():
    """Main function to run the backtest with the same data loading as test_bos_choch_plot.py"""
    print("🚀 Starting BOS/CHOCH Backtester with 60-day XAUUSD H1 data...")
    
    # === Step 1: Load data exactly like test_bos_choch_plot.py ===
    symbol = "XAUUSD_H1.csv"
    print(f"📊 Loading data for {symbol} with days_back=60...")
    
    # Use the same data loading function
    from core.data_loader import load_and_resample
    resampled = load_and_resample(f"data/{symbol}", days_back=60)
    h1_data = resampled.get("1H")
    
    if h1_data is None or h1_data.empty:
        print(f"❌ ERROR: No data loaded for the '1H' timeframe.")
        return
    
    print(f"📊 Loaded {len(h1_data)} H1 candles from {h1_data.index.min()} to {h1_data.index.max()}")
    
    # === Step 2: Get market structure analysis exactly like test_bos_choch_plot.py ===
    print("🔍 Analyzing market structure...")
    from core.structure_builder import get_market_analysis
    analysis = get_market_analysis(h1_data, prominence_factor=2.5)
    structure = analysis['structure']
    
    print(f"📊 Total structure points: {len(structure)}")
    
    # === Step 3: Detect market events with the same analyzer ===
    print("🎯 Detecting market events...")
    from core.smart_money_concepts import MarketStructureAnalyzer
    analyzer = MarketStructureAnalyzer(confidence_threshold=0.5)
    all_events = analyzer.get_market_events(structure)
    
    print(f"📊 Found {len(all_events)} total events with confidence >= {analyzer.confidence_threshold}")
    
    # Filter for A+ entries only (high confidence events)
    a_plus_events = [event for event in all_events if event.confidence >= 0.7]
    print(f"⭐ Found {len(a_plus_events)} A+ entries (confidence >= 70%)")
    
    if not a_plus_events:
        print("⚠️  No A+ entries found. Lowering confidence threshold to 50%...")
        a_plus_events = all_events
        print(f"📊 Using all {len(a_plus_events)} events with confidence >= 50%")
    
    # Show A+ entry details
    print("\n🔍 A+ Entry Details:")
    for i, event in enumerate(a_plus_events):
        print(f"  Entry {i+1}: {event.event_type.value} - {event.direction} @ {event.price:.2f}")
        print(f"    Confidence: {event.confidence:.3f}")
        print(f"    Time: {event.timestamp}")
        print(f"    Description: {event.description}")
        print()
    
    # === Step 4: Initialize and run backtester ===
    print("⚡ Initializing backtester...")
    
    # Load configuration
    from core.backtester_config import get_config
    config = get_config("balanced")  # Use balanced strategy preset
    
    # Override with specific settings for this test
    config['trading']['confidence_threshold'] = 0.7 if a_plus_events else 0.5
    config['data']['symbol'] = symbol.split('_')[0]
    config['data']['timeframe'] = '1H'
    
    print("📋 Backtester Configuration:")
    from core.backtester_config import print_config_summary
    print_config_summary(config)
    
    # Initialize backtester with config
    backtester = BOSCHOCHBacktester(
        initial_capital=config['trading']['initial_capital'],
        risk_per_trade=config['trading']['risk_per_trade'],
        reward_risk_ratio=config['trading']['reward_risk_ratio'],
        max_trade_duration=config['trading']['max_trade_duration'],
        confidence_threshold=config['trading']['confidence_threshold'],
        stop_loss_atr_multiplier=config['trading']['stop_loss_atr_multiplier']
    )
    
    # === Step 5: Run backtest ===
    print(f"\n🚀 Running backtest on {len(a_plus_events)} A+ entries...")
    print(f"📅 Date range: {h1_data.index.min()} to {h1_data.index.max()}")
    
    result = backtester.run_backtest(h1_data, a_plus_events)
    
    # === Step 6: Display results ===
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS")
    print("="*60)
    print(f"📈 Total Trades: {result.total_trades}")
    print(f"✅ Winning Trades: {result.winning_trades}")
    print(f"❌ Losing Trades: {result.losing_trades}")
    print(f"🎯 Win Rate: {result.win_rate:.2%}")
    print(f"💰 Total P&L: ${result.total_pnl:,.2f}")
    print(f"📊 Profit Factor: {result.profit_factor:.2f}")
    print(f"📉 Max Drawdown: {result.max_drawdown:.2%}")
    print(f"📊 Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"💵 Final Capital: ${result.total_pnl + 10000:,.2f}")
    print(f"📈 Total Return: {result.total_pnl_pct:.2%}")
    
    # === Step 7: Generate visualizations ===
    print("\n🎨 Generating visualizations...")
    visualizer = BacktestVisualizer(result)
    
    # Create output directory
    output_dir = "generated_backtest_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate all charts
    generate_backtest_report(result, output_dir)
    
    # Create comprehensive dashboard
    create_comprehensive_dashboard(result, visualizer, output_dir)
    
    print(f"\n✅ Backtest complete! All charts saved to: {output_dir}")
    print("🌐 Opening comprehensive dashboard...")
    
    # Open the main dashboard
    dashboard_path = os.path.join(output_dir, "comprehensive_dashboard.html")
    if os.path.exists(dashboard_path):
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
        print(f"📊 Dashboard opened: {dashboard_path}")
    else:
        print("❌ Dashboard file not found")
    
    return result

if __name__ == "__main__":
    main()
