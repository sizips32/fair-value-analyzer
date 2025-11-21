"""
시각화 대시보드 모듈
Plotly를 활용한 인터랙티브 차트 생성
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple

try:
    from ..analysis.monte_carlo import SimulationResult
except ImportError:
    from analysis.monte_carlo import SimulationResult


class FairValueDashboard:
    """공정가치 분석 대시보드"""

    def __init__(self):
        # 색상 팔레트 정의
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }

        # 차트 기본 레이아웃
        self.layout_config = {
            'template': 'plotly_white',
            'font': {'family': 'Arial, sans-serif', 'size': 12},
            'showlegend': True,
            'hovermode': 'x unified'
        }

    def create_comprehensive_chart(self, data: pd.DataFrame) -> go.Figure:
        """종합 가격 차트 (캔들스틱 + 기술지표)"""
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('가격 및 이동평균', 'MACD', 'RSI', '거래량'),
            row_width=[0.2, 0.2, 0.2, 0.4]
        )

        # 1. 캔들스틱 차트
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color=self.colors['success'],
                decreasing_line_color=self.colors['danger']
            ),
            row=1, col=1
        )

        # 이동평균선 추가
        if 'SMA_20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['SMA_20'],
                    line=dict(color=self.colors['primary'], width=1),
                    name='SMA 20'
                ),
                row=1, col=1
            )

        if 'SMA_50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['SMA_50'],
                    line=dict(color=self.colors['secondary'], width=1),
                    name='SMA 50'
                ),
                row=1, col=1
            )

        # 볼린저 밴드
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['BB_Upper'],
                    line=dict(color='rgba(128,128,128,0.3)'),
                    name='BB Upper',
                    showlegend=False
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['BB_Lower'],
                    line=dict(color='rgba(128,128,128,0.3)'),
                    fill='tonexty',
                    name='Bollinger Bands'
                ),
                row=1, col=1
            )

        # 2. MACD
        if 'MACD' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['MACD'],
                    line=dict(color=self.colors['primary']),
                    name='MACD'
                ),
                row=2, col=1
            )

        if 'MACD_Signal' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['MACD_Signal'],
                    line=dict(color=self.colors['secondary']),
                    name='Signal'
                ),
                row=2, col=1
            )

        if 'MACD_Histogram' in data.columns:
            fig.add_trace(
                go.Bar(
                    x=data.index, y=data['MACD_Histogram'],
                    marker_color=self.colors['info'],
                    name='Histogram',
                    opacity=0.6
                ),
                row=2, col=1
            )

        # 3. RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['RSI'],
                    line=dict(color=self.colors['warning']),
                    name='RSI'
                ),
                row=3, col=1
            )

            # RSI 기준선
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                         annotation_text="과매수", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                         annotation_text="과매도", row=3, col=1)

        # 4. 거래량
        fig.add_trace(
            go.Bar(
                x=data.index, y=data['Volume'],
                marker_color=self.colors['info'],
                name='Volume',
                opacity=0.7
            ),
            row=4, col=1
        )

        # 레이아웃 업데이트
        fig.update_layout(
            title="종합 기술적 분석 차트",
            xaxis_rangeslider_visible=False,
            height=800,
            **self.layout_config
        )

        return fig

    def create_simulation_paths_chart(self, mc_result: SimulationResult) -> go.Figure:
        """몬테카를로 시뮬레이션 경로 차트"""
        fig = go.Figure()

        # 경로 샘플 (너무 많으면 느려짐)
        n_paths_to_show = min(100, mc_result.paths.shape[0])
        indices = np.random.choice(mc_result.paths.shape[0], n_paths_to_show, replace=False)

        for i, idx in enumerate(indices):
            fig.add_trace(
                go.Scatter(
                    x=list(range(mc_result.paths.shape[1])),
                    y=mc_result.paths[idx],
                    mode='lines',
                    line=dict(color=self.colors['primary'], width=0.5),
                    opacity=0.3,
                    showlegend=False,
                    hovertemplate='Day: %{x}<br>Price: %{y:,.0f}<extra></extra>'
                )
            )

        # 평균 경로
        mean_path = np.mean(mc_result.paths, axis=0)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(mean_path))),
                y=mean_path,
                mode='lines',
                line=dict(color=self.colors['danger'], width=3),
                name='평균 경로'
            )
        )

        fig.update_layout(
            title="몬테카를로 시뮬레이션 경로",
            xaxis_title="일수",
            yaxis_title="가격",
            **self.layout_config
        )

        return fig

    def create_price_distribution_chart(self, mc_result: SimulationResult) -> go.Figure:
        """최종 가격 분포 히스토그램"""
        fig = go.Figure()

        # 히스토그램
        fig.add_trace(
            go.Histogram(
                x=mc_result.final_prices,
                nbinsx=50,
                name='가격 분포',
                marker_color=self.colors['primary'],
                opacity=0.7
            )
        )

        # 통계선들
        mean_price = mc_result.statistics['mean_price']
        median_price = mc_result.statistics['median_price']

        fig.add_vline(x=mean_price, line_dash="dash", line_color="red",
                     annotation_text=f"평균: {mean_price:,.0f}")
        fig.add_vline(x=median_price, line_dash="dash", line_color="green",
                     annotation_text=f"중앙값: {median_price:,.0f}")

        # 신뢰구간
        for level, ci in mc_result.confidence_intervals.items():
            if level == "95%":
                fig.add_vrect(
                    x0=ci['lower'], x1=ci['upper'],
                    fillcolor="rgba(255,0,0,0.1)",
                    annotation_text=f"{level} 신뢰구간",
                    annotation_position="top left"
                )

        fig.update_layout(
            title="최종 가격 분포",
            xaxis_title="가격",
            yaxis_title="빈도",
            **self.layout_config
        )

        return fig

    def create_confidence_intervals_chart(self, mc_result: SimulationResult) -> go.Figure:
        """신뢰구간 시계열 차트"""
        fig = go.Figure()

        # 백분위수 계산
        percentiles = [5, 25, 50, 75, 95]
        path_percentiles = np.percentile(mc_result.paths, percentiles, axis=0)

        x_axis = list(range(mc_result.paths.shape[1]))

        # 신뢰구간 영역
        fig.add_trace(
            go.Scatter(
                x=x_axis + x_axis[::-1],
                y=list(path_percentiles[0]) + list(path_percentiles[4][::-1]),
                fill='toself',
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='90% 신뢰구간',
                hoverinfo="skip"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_axis + x_axis[::-1],
                y=list(path_percentiles[1]) + list(path_percentiles[3][::-1]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='50% 신뢰구간',
                hoverinfo="skip"
            )
        )

        # 중앙값 라인
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=path_percentiles[2],
                line=dict(color=self.colors['primary'], width=2),
                name='중앙값'
            )
        )

        fig.update_layout(
            title="가격 예측 신뢰구간",
            xaxis_title="일수",
            yaxis_title="가격",
            **self.layout_config
        )

        return fig

    def create_technical_indicators_chart(self, data: pd.DataFrame) -> go.Figure:
        """기술적 지표 종합 차트"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RSI', 'Stochastic', 'MACD', '볼린저 밴드 위치'),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )

        # RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['RSI'],
                    line=dict(color=self.colors['warning']),
                    name='RSI'
                ),
                row=1, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

        # Stochastic
        if 'Stoch_K' in data.columns and 'Stoch_D' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['Stoch_K'],
                    line=dict(color=self.colors['primary']),
                    name='%K'
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['Stoch_D'],
                    line=dict(color=self.colors['secondary']),
                    name='%D'
                ),
                row=1, col=2
            )
            fig.add_hline(y=80, line_dash="dash", line_color="red", row=1, col=2)
            fig.add_hline(y=20, line_dash="dash", line_color="green", row=1, col=2)

        # MACD
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['MACD'],
                    line=dict(color=self.colors['primary']),
                    name='MACD'
                ),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['MACD_Signal'],
                    line=dict(color=self.colors['secondary']),
                    name='Signal'
                ),
                row=2, col=1
            )

        # 볼린저 밴드 위치
        if 'BB_Position' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['BB_Position'],
                    line=dict(color=self.colors['info']),
                    name='BB Position'
                ),
                row=2, col=2
            )
            fig.add_hline(y=0.8, line_dash="dash", line_color="red", row=2, col=2)
            fig.add_hline(y=0.2, line_dash="dash", line_color="green", row=2, col=2)

        fig.update_layout(
            title="기술적 지표 대시보드",
            height=600,
            **self.layout_config
        )

        return fig

    def create_risk_return_chart(self, mc_result: SimulationResult) -> go.Figure:
        """리스크-수익률 산점도"""
        fig = go.Figure()

        returns = mc_result.returns
        risk = np.abs(returns)  # 간단한 리스크 지표

        # 산점도
        fig.add_trace(
            go.Scatter(
                x=risk, y=returns,
                mode='markers',
                marker=dict(
                    color=returns,
                    colorscale='RdYlGn',
                    size=4,
                    opacity=0.6,
                    colorbar=dict(title="수익률")
                ),
                name='시뮬레이션 결과',
                hovertemplate='리스크: %{x:.2%}<br>수익률: %{y:.2%}<extra></extra>'
            )
        )

        # 기준선
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            title="리스크-수익률 분포",
            xaxis_title="리스크 (절대 수익률)",
            yaxis_title="수익률",
            **self.layout_config
        )

        return fig

    def create_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """상관관계 히트맵"""
        # 수치형 컬럼만 선택
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
        ))

        fig.update_layout(
            title="지표 간 상관관계",
            **self.layout_config
        )

        return fig

    def create_performance_metrics_chart(self, metrics: Dict) -> go.Figure:
        """성과 지표 레이더 차트"""
        categories = ['수익률', '변동성', '샤프비율', '최대손실률', '승률']

        # 정규화된 값 (0-100 스케일)
        values = [
            max(0, min(100, metrics.get('return', 0) * 10 + 50)),  # -5% ~ 5% -> 0-100
            max(0, min(100, 100 - metrics.get('volatility', 0.2) * 500)),  # 변동성 역순
            max(0, min(100, (metrics.get('sharpe_ratio', 0) + 2) * 25)),  # -2 ~ 2 -> 0-100
            max(0, min(100, 100 - abs(metrics.get('max_drawdown', 0)) * 500)),  # 손실률 역순
            max(0, min(100, metrics.get('win_rate', 0.5) * 100))  # 승률
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='성과 지표',
            marker_color=self.colors['primary']
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            title="투자 성과 레이더",
            **self.layout_config
        )

        return fig

    def create_drawdown_chart(self, data: pd.DataFrame) -> go.Figure:
        """손실률 차트"""
        if 'Close' not in data.columns:
            return go.Figure()

        prices = data['Close']
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak * 100

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=drawdown,
                fill='tonexty',
                fillcolor='rgba(255,0,0,0.3)',
                line=dict(color=self.colors['danger']),
                name='Drawdown (%)'
            )
        )

        fig.add_hline(y=0, line_color="black", line_width=1)

        fig.update_layout(
            title="누적 손실률 (Drawdown)",
            xaxis_title="날짜",
            yaxis_title="손실률 (%)",
            **self.layout_config
        )

        return fig

    def create_volume_analysis_chart(self, data: pd.DataFrame) -> go.Figure:
        """거래량 분석 차트"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('가격 vs 거래량', '거래량 지표'),
            vertical_spacing=0.1
        )

        # 가격과 거래량
        fig.add_trace(
            go.Scatter(
                x=data.index, y=data['Close'],
                line=dict(color=self.colors['primary']),
                name='가격'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=data.index, y=data['Volume'],
                marker_color=self.colors['info'],
                name='거래량',
                opacity=0.7,
                yaxis='y2'
            ),
            row=1, col=1
        )

        # 거래량 지표
        if 'OBV' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['OBV'],
                    line=dict(color=self.colors['success']),
                    name='OBV'
                ),
                row=2, col=1
            )

        if 'Volume_Ratio' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index, y=data['Volume_Ratio'],
                    line=dict(color=self.colors['warning']),
                    name='거래량 비율'
                ),
                row=2, col=1
            )
            fig.add_hline(y=1, line_dash="dash", line_color="gray", row=2, col=1)

        fig.update_layout(
            title="거래량 분석",
            height=600,
            yaxis2=dict(overlaying='y', side='right'),
            **self.layout_config
        )

        return fig
