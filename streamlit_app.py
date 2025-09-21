"""
Fair Value Analyzer - Streamlit 대시보드
통합 공정가치 분석 웹 애플리케이션
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

# 로컬 모듈 import
from analysis.workflow import UnifiedFairValueWorkflow
from config.settings import config_manager
from visualization.dashboard import FairValueDashboard

# 페이지 설정
st.set_page_config(
    page_title="Fair Value Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .signal-positive {
        color: #22c55e;
        font-weight: bold;
    }
    .signal-negative {
        color: #ef4444;
        font-weight: bold;
    }
    .signal-neutral {
        color: #64748b;
        font-weight: bold;
    }
    .insights-box {
        background-color: #fefce8;
        border: 1px solid #facc15;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FairValueApp:
    """Fair Value Analyzer 메인 애플리케이션"""

    def __init__(self):
        self.dashboard = FairValueDashboard()
        self.init_session_state()

    def init_session_state(self):
        """세션 상태 초기화"""
        if 'analysis_result' not in st.session_state:
            st.session_state.analysis_result = None
        if 'last_analysis_time' not in st.session_state:
            st.session_state.last_analysis_time = None
        if 'analysis_running' not in st.session_state:
            st.session_state.analysis_running = False

    def render_sidebar(self) -> Dict:
        """사이드바 렌더링 및 설정 수집"""
        st.sidebar.markdown("# ⚙️ 분석 설정")

        # 시장 선택
        markets = list(config_manager.markets.keys())
        market_names = [config_manager.markets[m].name for m in markets]

        selected_idx = st.sidebar.selectbox(
            "📈 분석 대상 시장",
            range(len(markets)),
            format_func=lambda x: market_names[x],
            help="분석할 시장을 선택하세요"
        )
        selected_market = markets[selected_idx]

        # 분석 기간
        st.sidebar.markdown("### 📅 분석 기간")

        period_options = {
            "1년": 365,
            "2년": 730,
            "3년": 1095,
            "5년": 1825
        }

        selected_period = st.sidebar.selectbox(
            "기간 선택",
            list(period_options.keys()),
            index=1,  # 기본값: 2년
            help="분석에 사용할 과거 데이터 기간"
        )

        # 사용자 정의 날짜
        use_custom_date = st.sidebar.checkbox("사용자 정의 날짜 사용")

        if use_custom_date:
            end_date = st.sidebar.date_input(
                "종료 날짜",
                datetime.now(),
                help="분석 종료 날짜"
            )
            start_date = st.sidebar.date_input(
                "시작 날짜",
                datetime.now() - timedelta(days=period_options[selected_period]),
                help="분석 시작 날짜"
            )
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_options[selected_period])

        # 분석 옵션
        st.sidebar.markdown("### 🔧 분석 옵션")

        monte_carlo_sims = st.sidebar.slider(
            "몬테카를로 시뮬레이션 횟수",
            1000, 50000, 10000,
            step=1000,
            help="더 많은 시뮬레이션은 정확도를 높이지만 시간이 오래 걸립니다"
        )

        forecast_days = st.sidebar.slider(
            "예측 기간 (일)",
            30, 252, 126,
            help="미래 가격을 예측할 기간"
        )

        confidence_levels = st.sidebar.multiselect(
            "신뢰구간 (%)",
            [90, 95, 99],
            default=[95, 99],
            help="표시할 신뢰구간 수준"
        )

        # 고급 설정
        with st.sidebar.expander("🔬 고급 설정"):
            mc_method = st.sidebar.selectbox(
                "몬테카를로 방법",
                ["geometric_brownian", "jump_diffusion", "heston_stochastic"],
                help="시뮬레이션 방법 선택"
            )

            include_scenarios = st.sidebar.checkbox(
                "시나리오 분석 포함",
                value=True,
                help="불마켓/베어마켓 시나리오 분석"
            )

        # 설정 업데이트
        if st.sidebar.button("⚙️ 설정 저장"):
            config_manager.update_config('monte_carlo',
                simulations=monte_carlo_sims,
                forecast_days=forecast_days,
                confidence_levels=confidence_levels,
                method=mc_method
            )
            st.sidebar.success("설정이 저장되었습니다!")

        return {
            'market': selected_market,
            'start_date': start_date,
            'end_date': end_date,
            'monte_carlo_sims': monte_carlo_sims,
            'forecast_days': forecast_days,
            'confidence_levels': confidence_levels,
            'mc_method': mc_method,
            'include_scenarios': include_scenarios
        }

    def render_main_header(self, market_name: str):
        """메인 헤더 렌더링"""
        market_config = config_manager.get_market_config(market_name)

        st.markdown(f'<h1 class="main-header">📊 {market_config.name} Fair Value Analyzer</h1>',
                   unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align: center; color: #666; margin-bottom: 2rem;">
            실시간 데이터 기반 공정가치 분석 및 예측 시스템<br>
            <small>Ticker: {market_config.ticker} | Currency: {market_config.currency}</small>
        </div>
        """, unsafe_allow_html=True)

    def render_analysis_controls(self, settings: Dict) -> bool:
        """분석 실행 컨트롤"""
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            run_analysis = st.button(
                "🚀 분석 실행",
                disabled=st.session_state.analysis_running,
                help="선택한 설정으로 전체 분석을 실행합니다"
            )

        with col2:
            if st.session_state.last_analysis_time:
                st.info(f"마지막 분석: {st.session_state.last_analysis_time.strftime('%H:%M:%S')}")

        with col3:
            auto_refresh = st.checkbox("자동 새로고침 (5분)", value=False)

        # 자동 새로고침 로직
        if auto_refresh and st.session_state.last_analysis_time:
            time_diff = datetime.now() - st.session_state.last_analysis_time
            if time_diff.total_seconds() > 300:  # 5분
                return True

        return run_analysis

    async def run_analysis(self, settings: Dict):
        """분석 실행"""
        st.session_state.analysis_running = True

        try:
            # 진행률 표시
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(progress: float, step_name: str, status: str):
                progress_bar.progress(progress)
                status_text.text(f"🔄 {step_name}: {status}")

            # 워크플로우 초기화
            workflow = UnifiedFairValueWorkflow(settings['market'])

            # 몬테카를로 설정 업데이트
            config_manager.update_config('monte_carlo',
                simulations=settings['monte_carlo_sims'],
                forecast_days=settings['forecast_days'],
                confidence_levels=settings['confidence_levels'],
                method=settings['mc_method']
            )

            # 분석 실행
            result = await workflow.run_complete_analysis(
                start_date=settings['start_date'],
                end_date=settings['end_date'],
                progress_callback=progress_callback
            )

            # 결과 저장
            st.session_state.analysis_result = result
            st.session_state.last_analysis_time = datetime.now()

            # UI 정리
            progress_bar.empty()
            status_text.empty()

            st.success("✅ 분석이 완료되었습니다!")

        except Exception as e:
            st.error(f"❌ 분석 중 오류 발생: {str(e)}")

        finally:
            st.session_state.analysis_running = False

    def render_summary_metrics(self, result):
        """요약 지표 카드"""
        if not result or not result.summary:
            return

        summary = result.summary

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            price = summary.get('market_info', {}).get('current_price', 0)
            change = summary.get('market_info', {}).get('price_change', 0)
            change_pct = summary.get('market_info', {}).get('price_change_percent', 0)

            st.metric(
                label="현재가",
                value=f"{price:,.0f}",
                delta=f"{change:+.2f} ({change_pct:+.2f}%)"
            )

        with col2:
            expected_return = summary.get('prediction_summary', {}).get('expected_return', 0)
            positive_prob = summary.get('prediction_summary', {}).get('positive_probability', 0)

            st.metric(
                label="예상 수익률",
                value=f"{expected_return:.1f}%",
                delta=f"상승확률: {positive_prob:.1f}%"
            )

        with col3:
            volatility = summary.get('risk_summary', {}).get('annual_volatility', 0)
            sharpe = summary.get('risk_summary', {}).get('sharpe_ratio', 0)

            st.metric(
                label="연간 변동성",
                value=f"{volatility:.1f}%",
                delta=f"샤프비율: {sharpe:.2f}"
            )

        with col4:
            signal = summary.get('technical_summary', {}).get('composite_signal', 0)
            strength = summary.get('technical_summary', {}).get('signal_strength', 0)

            signal_text = "매수" if signal > 0 else "매도" if signal < 0 else "관망"
            signal_color = "signal-positive" if signal > 0 else "signal-negative" if signal < 0 else "signal-neutral"

            st.markdown(f"""
            <div class="metric-container">
                <div style="font-size: 0.875rem; color: #666;">투자 신호</div>
                <div class="{signal_color}" style="font-size: 1.5rem;">{signal_text}</div>
                <div style="font-size: 0.875rem;">강도: {abs(strength):.2f}</div>
            </div>
            """, unsafe_allow_html=True)

    def render_main_dashboard(self, result):
        """메인 대시보드"""
        if not result:
            st.info("🔄 분석을 실행하여 결과를 확인하세요.")
            return

        # 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs(["📈 종합 분석", "🎲 몬테카를로", "📊 기술적 분석", "⚠️ 리스크 분석"])

        with tab1:
            self.render_comprehensive_analysis(result)

        with tab2:
            self.render_monte_carlo_analysis(result)

        with tab3:
            self.render_technical_analysis(result)

        with tab4:
            self.render_risk_analysis(result)

    def render_comprehensive_analysis(self, result):
        """종합 분석 탭"""
        col1, col2 = st.columns([2, 1])

        with col1:
            # 가격 차트 + 기술적 지표
            if not result.market_data.empty:
                fig = self.dashboard.create_comprehensive_chart(result.market_data)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # 인사이트 박스
            st.markdown("### 💡 AI 인사이트")

            if result.insights:
                insights_html = "<div class='insights-box'>"
                for insight in result.insights[-10:]:  # 최근 10개
                    insights_html += f"<p style='margin: 0.5rem 0;'>• {insight}</p>"
                insights_html += "</div>"

                st.markdown(insights_html, unsafe_allow_html=True)
            else:
                st.info("인사이트를 생성 중입니다...")

        # 신뢰구간 차트
        if result.monte_carlo_result:
            st.markdown("### 🎯 가격 예측 신뢰구간")
            confidence_fig = self.dashboard.create_confidence_intervals_chart(result.monte_carlo_result)
            st.plotly_chart(confidence_fig, use_container_width=True)

    def render_monte_carlo_analysis(self, result):
        """몬테카를로 분석 탭"""
        if not result.monte_carlo_result:
            st.warning("몬테카를로 시뮬레이션 결과가 없습니다.")
            return

        mc_result = result.monte_carlo_result

        col1, col2 = st.columns(2)

        with col1:
            # 시뮬레이션 경로
            st.markdown("### 📈 시뮬레이션 경로")
            paths_fig = self.dashboard.create_simulation_paths_chart(mc_result)
            st.plotly_chart(paths_fig, use_container_width=True)

        with col2:
            # 분포 히스토그램
            st.markdown("### 📊 최종 가격 분포")
            dist_fig = self.dashboard.create_price_distribution_chart(mc_result)
            st.plotly_chart(dist_fig, use_container_width=True)

        # 통계 테이블
        st.markdown("### 📋 시뮬레이션 통계")
        stats_df = pd.DataFrame({
            '지표': ['평균 가격', '중앙값', '표준편차', '최저가', '최고가', '상승 확률', '샤프 비율'],
            '값': [
                f"{mc_result.statistics['mean_price']:,.0f}",
                f"{mc_result.statistics['median_price']:,.0f}",
                f"{mc_result.statistics['std_price']:,.0f}",
                f"{mc_result.statistics['min_price']:,.0f}",
                f"{mc_result.statistics['max_price']:,.0f}",
                f"{mc_result.statistics['positive_return_prob']*100:.1f}%",
                f"{mc_result.statistics['sharpe_ratio']:.3f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True)

    def render_technical_analysis(self, result):
        """기술적 분석 탭"""
        if result.market_data.empty:
            st.warning("시장 데이터가 없습니다.")
            return

        # 기술적 지표 차트
        technical_fig = self.dashboard.create_technical_indicators_chart(result.market_data)
        st.plotly_chart(technical_fig, use_container_width=True)

        # 매매 신호 요약
        if not result.signals.empty:
            st.markdown("### 📊 매매 신호 요약")

            latest_signals = result.signals.iloc[-1]
            signal_cols = st.columns(5)

            signal_names = ['RSI', 'MACD', 'Bollinger', 'MA', 'Stochastic']
            signal_keys = ['RSI_Signal', 'MACD_Signal', 'BB_Signal', 'MA_Signal', 'Stoch_Signal']

            for i, (name, key) in enumerate(zip(signal_names, signal_keys)):
                with signal_cols[i]:
                    if key in latest_signals:
                        signal_value = latest_signals[key]
                        signal_text = "매수" if signal_value > 0 else "매도" if signal_value < 0 else "중립"
                        signal_color = "#22c55e" if signal_value > 0 else "#ef4444" if signal_value < 0 else "#64748b"

                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; border: 1px solid {signal_color}; border-radius: 0.5rem;">
                            <div style="font-weight: bold;">{name}</div>
                            <div style="color: {signal_color}; font-size: 1.2rem;">{signal_text}</div>
                        </div>
                        """, unsafe_allow_html=True)

    def render_risk_analysis(self, result):
        """리스크 분석 탭"""
        if not result.risk_metrics:
            st.warning("리스크 지표가 없습니다.")
            return

        risk_metrics = result.risk_metrics

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 기본 리스크 지표")

            basic_metrics = pd.DataFrame({
                '지표': [
                    '일간 변동성',
                    '연간 변동성',
                    '샤프 비율',
                    '소르티노 비율',
                    '최대 손실률'
                ],
                '값': [
                    f"{risk_metrics.get('volatility_daily', 0)*100:.2f}%",
                    f"{risk_metrics.get('volatility_annual', 0)*100:.2f}%",
                    f"{risk_metrics.get('sharpe_ratio', 0):.3f}",
                    f"{risk_metrics.get('sortino_ratio', 0):.3f}",
                    f"{abs(risk_metrics.get('maximum_drawdown', 0))*100:.2f}%"
                ]
            })
            st.dataframe(basic_metrics, use_container_width=True)

        with col2:
            st.markdown("### ⚠️ Value at Risk (VaR)")

            var_metrics = pd.DataFrame({
                '신뢰구간': ['95%', '99%'],
                '과거 데이터 VaR': [
                    f"{risk_metrics.get('var_95_historical', 0)*100:.2f}%",
                    f"{risk_metrics.get('var_99_historical', 0)*100:.2f}%"
                ],
                '시뮬레이션 VaR': [
                    f"{((risk_metrics.get('var_95_simulation', 0) / result.market_data['Close'].iloc[-1]) - 1)*100:.2f}%" if not result.market_data.empty else "N/A",
                    f"{((risk_metrics.get('var_99_simulation', 0) / result.market_data['Close'].iloc[-1]) - 1)*100:.2f}%" if not result.market_data.empty else "N/A"
                ]
            })
            st.dataframe(var_metrics, use_container_width=True)

        # 리스크 시각화
        if result.monte_carlo_result:
            st.markdown("### 📈 리스크-수익률 분포")
            risk_return_fig = self.dashboard.create_risk_return_chart(result.monte_carlo_result)
            st.plotly_chart(risk_return_fig, use_container_width=True)

    def render_footer(self):
        """푸터"""
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>📊 Fair Value Analyzer v1.0 |
            Powered by Monte Carlo Simulation & Technical Analysis</p>
            <p><small>⚠️ 투자 결정은 본인 책임하에 신중히 하시기 바랍니다.</small></p>
        </div>
        """, unsafe_allow_html=True)

    def run(self):
        """애플리케이션 실행"""
        # 사이드바 설정
        settings = self.render_sidebar()

        # 메인 헤더
        self.render_main_header(settings['market'])

        # 분석 컨트롤
        should_run_analysis = self.render_analysis_controls(settings)

        # 분석 실행
        if should_run_analysis:
            asyncio.run(self.run_analysis(settings))
            st.rerun()  # 결과 표시를 위해 리로드

        # 요약 지표
        if st.session_state.analysis_result:
            self.render_summary_metrics(st.session_state.analysis_result)

        # 메인 대시보드
        self.render_main_dashboard(st.session_state.analysis_result)

        # 푸터
        self.render_footer()

# 애플리케이션 실행
if __name__ == "__main__":
    app = FairValueApp()
    app.run()