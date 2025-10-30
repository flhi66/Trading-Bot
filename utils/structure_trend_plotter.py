import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Optional, Tuple
from core.smart_money_concepts import StructurePoint
from core.structure_builder import build_market_structure, detect_swing_points_scipy, get_market_analysis
from core.trend_detector import detect_swing_points, detect_trend, get_trend_from_data


class StructureTrendPlotter:
    """
    Professional plotter for market structure and trend analysis - Matching BOS/CHOCH chart style
    """
    
    def __init__(self):
        # Color scheme matching level_plotter.py
        self.color_scheme: Dict[Tuple[str, str], str] = {
            ('HH', 'Bullish'): '#26a69a', ('HH', 'Bearish'): '#ef5350',
            ('HL', 'Bullish'): '#2196F3', ('HL', 'Bearish'): '#FFA726',
            ('LH', 'Bullish'): '#9C27B0', ('LH', 'Bearish'): '#FF5722',
            ('LL', 'Bullish'): '#4CAF50', ('LL', 'Bearish'): '#F44336'
        }
        
        self.trend_colors = {
            'uptrend': '#00E676',
            'downtrend': '#F44336',
            'sideways': '#9E9E9E'
        }
        
        # Level importance weights for prioritization
        self.level_importance = {
            'HH': 5,
            'HL': 4,
            'LH': 4,
            'LL': 5,
            'Swing': 3
        }

    def _create_base_chart(self, df: pd.DataFrame, title: str, symbol: str) -> go.Figure:
        """Creates the base candlestick figure - Matching level_plotter.py style"""
        df["numeric_index"] = range(len(df))
        df.index = pd.to_datetime(df.index)
        df['hover_text'] = df.index.strftime('%Y-%m-%d %H:%M:%S') + \
                    "<br>Open: " + df['Open'].astype(str) + \
                    "<br>High: " + df['High'].astype(str) + \
                    "<br>Low: " + df['Low'].astype(str) + \
                    "<br>Close: " + df['Close'].astype(str)



        fig = go.Figure(data=go.Candlestick(
            x=df['numeric_index'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="", increasing_line_color='#77dd76', decreasing_line_color='#ef5350',
            customdata=df.index, # datetime index for hover

            hoverinfo="text"))
        fig.update_layout(
            title=title, template="plotly_dark", 
            height=900,
            xaxis_rangeslider_visible=False, showlegend=True,
            margin=dict(l=0, r=0, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='closest',  # zeigt nur den Punkt unter dem Cursor
            xaxis=dict(showspikes=False),
            yaxis=dict(showspikes=False)
            # xaxis=dict(
            #     showspikes=True,
            #     spikemode='across',      
            #     spikecolor='lightgray',   
            #     spikethickness=0.5,       
            #     spikesnap='cursor',
            #     layer='below traces'     
            # ),
            # yaxis=dict(
            #     showspikes=True,
            #     spikemode='across',
            #     spikecolor='lightgray',
            #     spikethickness=0.5,
            #     spikesnap='cursor',
            #     layer='below traces'
            # )
        )
        fig.update_xaxes(visible=True, showticklabels=True, 
                         tickvals=df["numeric_index"][::max(1, len(df)//20)],  # 20 Ticks max
                        ticktext=[d.strftime("%Y-%m-%d %H:%M") for d in df.index[::max(1, len(df)//20)]],
                        tickangle=45, 
                        title_text="Time",
                        hoverformat="%Y-%m-%d %H:%M:%S"
                        )
        
        fig.update_traces(hoverinfo="text", hovertext=df['hover_text'])

        fig.update_yaxes(visible=True, showticklabels=True, dtick=0.005 if df['Close'].max() < 1 else None)
        return fig

    def standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to handle inconsistencies between files
        """
        df_copy = df.copy()
        
        # Map common column variations to standard names
        column_mapping = {
            'high': 'High',
            'low': 'Low', 
            'open': 'Open',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        # Rename columns if they exist
        for old_col, new_col in column_mapping.items():
            if old_col in df_copy.columns:
                df_copy.rename(columns={old_col: new_col}, inplace=True)
        
        return df_copy

    def _add_swing_break_line(self, fig: go.Figure, from_point: Dict, to_point: Dict, color: str):
        """Add dotted line connecting swing points - Matching BOS/CHOCH style"""
        # Create dotted line between swing points
        fig.add_trace(go.Scatter(
            x=[from_point['timestamp'], to_point['timestamp']],
            y=[from_point['price'], to_point['price']],
            mode='lines',
            line=dict(
                color=color,
                width=2,
                dash='dot'
            ),
            name="Swing Connection",
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Add a small circle at the origin point
        fig.add_trace(go.Scatter(
            x=[from_point['timestamp']],
            y=[from_point['price']],
            mode='markers',
            marker=dict(
                symbol='circle-open',
                size=8,
                color=color,
                line=dict(width=2, color=color)
            ),
            name=f"Swing Point",
            hoverinfo='skip',
            showlegend=False
        ))

    def plot_trend_detection(self, df: pd.DataFrame, symbol: str = "SYMBOL", timeframe: str = "1H") -> go.Figure:
        """
        Chart 1: Trend Detection Analysis - Matching BOS/CHOCH chart style
        """
        # Standardize DataFrame
        df_std = self.standardize_dataframe(df)
        
        # Create base chart matching level_plotter.py style
        fig = self._create_base_chart(df_std, f"{symbol} - Trend Detection Analysis", symbol)
        
        # Detect swing points using trend_detector logic (lowercase columns for compatibility)
        df_lower = df_std.copy()
        df_lower.columns = [col.lower() for col in df_lower.columns]
        
        swing_highs, swing_lows = detect_swing_points(df_lower)
        trend = detect_trend(swing_highs, swing_lows)
        
        # Color based on detected trend
        trend_color = self.trend_colors.get(trend, '#9E9E9E')
        
        # Process swing highs
        if swing_highs:
            for i, (timestamp, price) in enumerate(swing_highs):
                # Determine symbol and position for swing marker
                symbol_icon = 'triangle-down'
                marker_y_position = price + (abs(price) * 0.002)
                arrow_y_offset = 50
                
                # Add swing high marker
                fig.add_trace(go.Scatter(
                    x=[timestamp], 
                    y=[marker_y_position], 
                    mode='markers',
                    marker=dict(
                        symbol=symbol_icon, 
                        size=12, 
                        color='#ef5350', 
                        line=dict(width=2, color='white')
                    ),
                    name=f"Swing High",
                    hovertemplate=f"<b>Swing High</b><br>" +
                                 f"Price: {price:.2f}<br>" +
                                 f"Time: {timestamp}<br>" +
                                 f"<i>Resistance level</i><extra></extra>",
                    showlegend=False
                ))
                
                # Add clean label
                fig.add_annotation(
                    x=timestamp, 
                    y=marker_y_position,
                    text=f"<b>HIGH</b><br>@{price:.2f}", 
                    showarrow=True, 
                    arrowhead=2, 
                    arrowcolor='#ef5350',
                    ax=0, 
                    ay=arrow_y_offset,
                    bgcolor='#ef5350', 
                    font=dict(color='white', size=10), 
                    borderpad=4,
                    bordercolor='white',
                    borderwidth=1
                )
                
                # Connect swing highs with lines
                if i > 0:
                    prev_timestamp, prev_price = swing_highs[i-1]
                    self._add_swing_break_line(
                        fig, 
                        {'timestamp': prev_timestamp, 'price': prev_price},
                        {'timestamp': timestamp, 'price': price},
                        '#ef5350'
                    )
        
        # Process swing lows
        if swing_lows:
            for i, (timestamp, price) in enumerate(swing_lows):
                # Determine symbol and position for swing marker
                symbol_icon = 'triangle-up'
                marker_y_position = price - (abs(price) * 0.002)
                arrow_y_offset = -50
                
                # Add swing low marker
                fig.add_trace(go.Scatter(
                    x=[timestamp], 
                    y=[marker_y_position], 
                    mode='markers',
                    marker=dict(
                        symbol=symbol_icon, 
                        size=12, 
                        color='#26a69a', 
                        line=dict(width=2, color='white')
                    ),
                    name=f"Swing Low",
                    hovertemplate=f"<b>Swing Low</b><br>" +
                                 f"Price: {price:.2f}<br>" +
                                 f"Time: {timestamp}<br>" +
                                 f"<i>Support level</i><extra></extra>",
                    showlegend=False
                ))
                
                # Add clean label
                fig.add_annotation(
                    x=timestamp, 
                    y=marker_y_position,
                    text=f"<b>LOW</b><br>@{price:.2f}", 
                    showarrow=True, 
                    arrowhead=2, 
                    arrowcolor='#26a69a',
                    ax=0, 
                    ay=arrow_y_offset,
                    bgcolor='#26a69a', 
                    font=dict(color='white', size=10), 
                    borderpad=4,
                    bordercolor='white',
                    borderwidth=1
                )
                
                # Connect swing lows with lines
                if i > 0:
                    prev_timestamp, prev_price = swing_lows[i-1]
                    self._add_swing_break_line(
                        fig, 
                        {'timestamp': prev_timestamp, 'price': prev_price},
                        {'timestamp': timestamp, 'price': price},
                        '#26a69a'
                    )
        
        # Add trend background with the detected trend color
        fig.add_hrect(
            y0=df_std['Low'].min() * 0.999,
            y1=df_std['High'].max() * 1.001,
            fillcolor=trend_color,
            opacity=0.08,
            layer="below",
            line_width=0,
        )
        
        # Add trend summary annotation
        trend_emoji = {'uptrend': '📈', 'downtrend': '📉', 'sideways': '↔️'}
        fig.add_annotation(
            x=df_std.index[-1],
            y=df_std['High'].max() * 0.98,
            text=f"<b>{trend_emoji.get(trend, '❓')} {trend.upper()}</b><br>" +
                 f"Highs: {len(swing_highs)}<br>Lows: {len(swing_lows)}",
            showarrow=False,
            xanchor='right',
            yanchor='top',
            bgcolor=trend_color,
            font=dict(color='white', size=12),
            bordercolor='white',
            borderwidth=2,
            borderpad=8
        )
        
        return fig

    def plot_structure_analysis(self, df: pd.DataFrame, symbol: str = "SYMBOL", timeframe: str = "1H") -> go.Figure:
        """
        Chart 2: HH/HL/LL/LH Structure Detection - Matching BOS/CHOCH chart style
        """
        # Standardize DataFrame
        df_std = self.standardize_dataframe(df)
        
        # Create base chart matching level_plotter.py style
        fig = self._create_base_chart(df_std, f"{symbol} - Market Structure Analysis (HH/HL/LH/LL)", symbol)
        
        # Get market structure analysis
        analysis = get_market_analysis(df_std)
        structure = analysis['structure']
        trend = analysis['trend']
        
        # Process structure points with BOS/CHOCH styling
        structure_counts = {'HH': 0, 'HL': 0, 'LH': 0, 'LL': 0}
        
        for i, struct_point in enumerate(structure):
            struct_type = struct_point['type']
            timestamp = struct_point['timestamp']
            price = struct_point['price']
            
            structure_counts[struct_type] += 1
            
            # Get color for this structure type
            color = self.color_scheme.get((struct_type, 'Bullish' if struct_type in ['HH', 'HL'] else 'Bearish'), '#888888')
            
            # Determine symbol and position based on structure type
            if struct_type == 'HH':
                symbol_icon = 'star'
                marker_y_position = price + (abs(price) * 0.002)
                arrow_y_offset = 50
            elif struct_type == 'HL':
                symbol_icon = 'triangle-up'
                marker_y_position = price - (abs(price) * 0.002)
                arrow_y_offset = -50
            elif struct_type == 'LH':
                symbol_icon = 'triangle-down'
                marker_y_position = price + (abs(price) * 0.002)
                arrow_y_offset = 50
            elif struct_type == 'LL':
                symbol_icon = 'star'
                marker_y_position = price - (abs(price) * 0.002)
                arrow_y_offset = -50
            
            # Add structure point marker
            fig.add_trace(go.Scatter(
                x=[timestamp], 
                y=[marker_y_position], 
                mode='markers',
                marker=dict(
                    symbol=symbol_icon, 
                    size=15, 
                    color=color, 
                    line=dict(width=2, color='white')
                ),
                name=f"{struct_type}",
                hovertemplate=f"<b>{struct_type}</b><br>" +
                             f"Price: {price:.2f}<br>" +
                             f"Time: {timestamp}<br>" +
                             f"<i>{self._get_structure_description(struct_type)}</i><extra></extra>",
                showlegend=False
            ))
            
            # Add clean structure label
            fig.add_annotation(
                x=timestamp, 
                y=marker_y_position,
                text=f"<b>{struct_type}</b><br>@{price:.2f}", 
                showarrow=True, 
                arrowhead=2, 
                arrowcolor=color,
                ax=0, 
                ay=arrow_y_offset,
                bgcolor=color, 
                font=dict(color='white', size=10), 
                borderpad=4,
                bordercolor='white',
                borderwidth=1
            )
            
            # Connect structure points of same type with lines
            if i > 0:
                # Find previous structure point of same type
                prev_same_type = None
                for j in range(i-1, -1, -1):
                    if structure[j]['type'] == struct_type:
                        prev_same_type = structure[j]
                        break
                
                if prev_same_type:
                    self._add_swing_break_line(
                        fig,
                        {'timestamp': prev_same_type['timestamp'], 'price': prev_same_type['price']},
                        {'timestamp': timestamp, 'price': price},
                        color
                    )
        
        # Add trend background
        trend_color = self.trend_colors.get(trend, '#9E9E9E')
        fig.add_hrect(
            y0=df_std['Low'].min() * 0.999,
            y1=df_std['High'].max() * 1.001,
            fillcolor=trend_color,
            opacity=0.08,
            layer="below",
            line_width=0,
        )
        
        # Add structure summary annotation
        total_points = sum(structure_counts.values())
        bullish_bias = ((structure_counts['HH'] + structure_counts['HL']) / total_points * 100) if total_points > 0 else 0
        
        fig.add_annotation(
            x=df_std.index[200],
            y=df_std['High'].max(), #if df_std['High'][-1] < 0.8 * df_std['High'].max() else 0.7 * df_std['High'].max(),
            text=f"<b>📊 STRUCTURE</b><br>" +
                 f"Trend: {trend.upper()}<br>" +
                 f"HH: {structure_counts['HH']} | HL: {structure_counts['HL']}<br>" +
                 f"LH: {structure_counts['LH']} | LL: {structure_counts['LL']}<br>" +
                 f"Bullish Bias: {bullish_bias:.1f}%",
            showarrow=False,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(0,0,0,0.8)',
            font=dict(color='white', size=11),
            bordercolor='white',
            borderwidth=2,
            borderpad=8
        )
        
        return fig

    def plot_structure(self, df: pd.DataFrame, structure: pd.DataFrame, events: List[StructurePoint] | None = None, symbol: str = "SYMBOL", timeframe: str = "15Min",
                       show_labels: bool = False) -> go.Figure:
        """
        Chart: Market Structure Detection - Matching BOS/CHOCH chart style
        """

        # Standardize and copy DataFrame
        df_std = self.standardize_dataframe(df)

        # Create base chart matching level_plotter.py style

        fig = go.Figure()
        # Create base chart matching level_plotter.py style
        fig = self._create_base_chart(df_std, f"{symbol} {timeframe} - Market Structure Analysis (HH/HL/LH/LL)", symbol)
        date_to_num = dict(zip(df_std.index, df_std["numeric_index"]))
        num_to_date = dict(zip(df_std["numeric_index"], df_std.index))
        sp_by_timestamp = {sp.timestamp: sp for sp in events } if events is not None else {}


        # Process structure points with BOS/CHOCH styling
        structure_counts = {'HH': 0, 'HL': 0, 'LH': 0, 'LL': 0}

        for i, struct_point in enumerate(structure.itertuples()):
            struct_type = struct_point.Classification
            timestamp = struct_point.Index
            price = float(struct_point.Level)
            retracement = float(struct_point.Retracement)
            
            if struct_type not in structure_counts:
                continue  # Skip unknown structure types
            
            structure_counts[struct_type] += 1
            
            # Get color for this structure type
            color = self.color_scheme.get((str(struct_type), 'Bullish' if struct_type in ['HH', 'HL'] else 'Bearish'), '#888888')
            
            # Determine symbol and position based on structure type
            if struct_type == 'HH':
                symbol_icon = 'diamond-tall'
                marker_y_position = price + (abs(price) * 0.0005)
                arrow_y_offset = -50
            elif struct_type == 'HL':
                symbol_icon = 'diamond-wide'
                marker_y_position = price - (abs(price) * 0.0005)
                arrow_y_offset = 50
            elif struct_type == 'LH':
                symbol_icon = 'diamond-wide'
                marker_y_position = price + (abs(price) * 0.0005)
                arrow_y_offset = -50
            elif struct_type == 'LL':
                symbol_icon = 'diamond-tall'
                marker_y_position = price - (abs(price) * 0.0005)
                arrow_y_offset = +50
            
            is_in_rangebreak = timestamp.day_name().lower() in ["saturday", "sunday"] # type: ignore
            if is_in_rangebreak:
                print(f"Warning: Structure point at {timestamp} falls within a rangebreak period.")

            # Add structure point marker
            fig.add_trace(go.Scatter(
                x=[date_to_num[timestamp]], 
                y=[marker_y_position], 
                mode='markers',
                marker=dict(
                    symbol=symbol_icon, 
                    size=7, 
                    color=color, 
                    line=dict(width=2, color='white')
                ),
                name=f"{struct_type}",
                hovertemplate=f"<b>{struct_type}</b><br>" +
                             f"Price: {price:.5f}<br>" +
                             f"Time: {timestamp}<br>" +
                             f"Retracement: {retracement:.2f}%<br>" +
                             f"Structure Type: {sp_by_timestamp[timestamp].structure_type.value if events is not None else 'Missing events'} <br>" + # type: ignore
                             f"Strong: {sp_by_timestamp[timestamp].strong if events is not None else 'Missing events'} <br>" +
                             f"<i>{self._get_structure_description(struct_type)}</i><extra></extra>",
                showlegend=False
            ))
            
            if show_labels:
                # Add clean structure label
                fig.add_annotation(
                    x=date_to_num[timestamp], 
                    y=marker_y_position,
                    text=f"<b>{struct_type}</b><br>@{price:.5f}", 
                    showarrow=True, 
                    arrowhead=2, 
                    arrowcolor=color,
                    ax=0, 
                    ay=arrow_y_offset,
                    bgcolor=color, 
                    font=dict(color='white', size=10), 
                    borderpad=4,
                    bordercolor='white',
                    borderwidth=1
                )
            
            # Connect structure points of same type with lines
            # if i > 0:
            #     # Find previous structure point of same type
            #     prev_same_type = None
            #     for j in range(i-1, -1, -1):
            #         if structure[j]['type'] == struct_type:
            #             prev_same_type = structure[j]
            #             break
                
            #     if prev_same_type:
            #         self._add_swing_break_line(
            #             fig,
            #             {'timestamp': prev_same_type['timestamp'], 'price': prev_same_type['price']},
            #             {'timestamp': timestamp, 'price': price},
            #             color
            #         )

        # Add trend background
        # trend_color = self.trend_colors.get(trend, '#9E9E9E')
        # fig.add_hrect(
        #     y0=df_std['Low'].min() * 0.999,
        #     y1=df_std['High'].max() * 1.001,
        #     fillcolor=trend_color,
        #     opacity=0.08,
        #     layer="below",
        #     line_width=0,
        # )

        # Filter only rows with valid HighLow
        swings = structure.dropna(subset=['HighLow', 'Level'])

        # Use the datetime index of the swings directly
        swing_dates = swings.index
        swing_levels = swings['Level'].values
        swing_types = swings['HighLow'].values

        # Draw lines between consecutive swings
        for i in range(len(swings) - 1):
            date = swing_dates[i]
            fig.add_trace(
                go.Scatter(
                    x=[date_to_num[swing_dates[i]], date_to_num[swing_dates[i + 1]]],
                    y=[swing_levels[i], swing_levels[i + 1]],
                    mode='lines',
                    line=dict(
                        color='rgba(0, 128, 0, 0.5)' if swing_types[i] == -1 else 'rgba(255, 0, 0, 0.5)',
                        width=2
                    ),
                    showlegend=False,
                    customdata=[swing_dates[i]],
                    hovertemplate="X: %{customdata}, Y: %{y}<extra></extra>",
                )
        )
        
        # Add structure summary annotation
        total_points = sum(structure_counts.values())
        bullish_bias = ((structure_counts['HH'] + structure_counts['HL']) / total_points * 100) if total_points > 0 else 0
        
        len_of_df = len(df_std)
        relative_x_pos = int(len_of_df * 0.1)

        fig.add_annotation(
            x=date_to_num[df_std.index[relative_x_pos]],
            y=df_std['High'].max(), #if df_std['High'][-1] < 0.8 * df_std['High'].max() else 0.7 * df_std['High'].max(),
            text=f"<b>📈 STRUCTURE</b><br>" +
                 #f"Trend: {trend.upper()}<br>" +
                 f"HH: {structure_counts['HH']} | HL: {structure_counts['HL']}<br>" +
                 f"LH: {structure_counts['LH']} | LL: {structure_counts['LL']}<br>" +
                 f"Bullish Bias: {bullish_bias:.1f}%",
            showarrow=False,
            xanchor='right',
            yanchor='top',
            bgcolor='rgba(0,0,0,0.8)',
            font=dict(color='white', size=11),
            bordercolor='white',
            borderwidth=2,
            borderpad=8
        )


        return fig

    def _get_structure_description(self, struct_type: str) -> str:
        """Get description for structure type"""
        descriptions = {
            'HH': 'Higher High - Bullish structure',
            'HL': 'Higher Low - Bullish structure', 
            'LH': 'Lower High - Bearish structure',
            'LL': 'Lower Low - Bearish structure'
        }
        return descriptions.get(struct_type, 'Structure point')