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

        # 분석 모드 선택
        analysis_mode = st.sidebar.radio(
            "📊 분석 모드",
            ["주요 지수", "개별 종목"],
            help="분석할 대상을 선택하세요"
        )

        if analysis_mode == "주요 지수":
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
            custom_ticker = None
            
        else:  # 개별 종목
            st.sidebar.markdown("### 🏢 개별 종목 분석")
            
            # 도움말
            with st.sidebar.expander("💡 티커 입력 도움말"):
                st.markdown("""
                **주요 티커 예시:**
                - 미국: AAPL, MSFT, TSLA, GOOGL
                - 한국: 005930.KS (삼성전자), 000660.KS (SK하이닉스)
                - 일본: 7203.T (토요타), 6758.T (소니)
                - 유럽: ASML.AS, SAP.DE
                
                **참고사항:**
                - 한국 주식은 .KS 접미사 필요
                - 일본 주식은 .T 접미사 필요
                - 유럽 주식은 .AS, .DE 등 접미사 필요
                """)
            
            # 티커 입력
            custom_ticker = st.sidebar.text_input(
                "📝 종목 티커 입력",
                placeholder="예: AAPL, MSFT, TSLA, 005930.KS",
                help="Yahoo Finance 티커 심볼을 입력하세요"
            )
            
            if custom_ticker:
                # 티커 유효성 검사
                try:
                    import yfinance as yf
                    test_ticker = yf.Ticker(custom_ticker)
                    info = test_ticker.info
                    if info and 'symbol' in info:
                        company_name = info.get('longName', custom_ticker)
                        sector = info.get('sector', '')
                        industry = info.get('industry', '')
                        currency = info.get('currency', 'USD')
                        
                        st.sidebar.success(f"✅ {company_name} 확인됨")
                        
                        # 추가 정보 표시
                        if sector:
                            st.sidebar.info(f"🏢 섹터: {sector}")
                        if industry:
                            st.sidebar.info(f"🏭 업종: {industry}")
                        st.sidebar.info(f"💰 통화: {currency}")
                        
                        selected_market = "custom"
                    else:
                        st.sidebar.error("❌ 유효하지 않은 티커입니다")
                        selected_market = "kospi"  # 기본값
                except Exception as e:
                    st.sidebar.error(f"❌ 티커 검증 오류: {str(e)}")
                    selected_market = "kospi"  # 기본값
            else:
                st.sidebar.warning("⚠️ 티커를 입력해주세요")
                selected_market = "kospi"  # 기본값

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
            'custom_ticker': custom_ticker,
            'start_date': start_date,
            'end_date': end_date,
            'monte_carlo_sims': monte_carlo_sims,
            'forecast_days': forecast_days,
            'confidence_levels': confidence_levels,
            'mc_method': mc_method,
            'include_scenarios': include_scenarios
        }

    def render_main_header(self, settings: Dict):
        """메인 헤더 렌더링"""
        market_name = settings['market']
        custom_ticker = settings.get('custom_ticker')
        
        if market_name == "custom" and custom_ticker:
            # 개별 종목 분석
            try:
                import yfinance as yf
                ticker = yf.Ticker(custom_ticker)
                info = ticker.info
                
                company_name = info.get('longName', custom_ticker)
                currency = info.get('currency', 'USD')
                
                st.markdown(f'<h1 class="main-header">📊 {company_name} Fair Value Analyzer</h1>',
                           unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="text-align: center; color: #666; margin-bottom: 2rem;">
                    실시간 데이터 기반 공정가치 분석 및 예측 시스템<br>
                    <small>Ticker: {custom_ticker} | Currency: {currency}</small>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"종목 정보를 가져올 수 없습니다: {str(e)}")
                st.markdown('<h1 class="main-header">📊 개별 종목 Fair Value Analyzer</h1>',
                           unsafe_allow_html=True)
        else:
            # 주요 지수 분석
            market_config = config_manager.get_market_config(market_name)

            st.markdown(f'<h1 class="main-header">📊 {market_config.name} Fair Value Analyzer</h1>',
                       unsafe_allow_html=True)

            st.markdown(f"""
            <div style="text-align: center; color: #666; margin-bottom: 2rem;">
                실시간 데이터 기반 공정가치 분석 및 예측 시스템<br>
                <small>Ticker: {market_config.ticker} | Currency: {market_config.currency}</small>
            </div>
            """, unsafe_allow_html=True)

    def render_stock_info(self, result):
        """개별 종목 추가 정보 표시"""
        if not result or not result.summary:
            return
            
        summary = result.summary
        market_info = summary.get('market_info', {})
        
        # 실시간 데이터에서 추가 정보 가져오기
        try:
            import yfinance as yf
            from data.collectors import DataProcessor
            
            ticker = yf.Ticker(market_info.get('ticker', ''))
            info = ticker.info
            
            st.markdown("### 🏢 종목 기본 정보")
            
            # 통화 정보 가져오기
            currency = info.get('currency', 'USD')
            financial_currency = info.get('financialCurrency', currency)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                market_cap = info.get('marketCap', 0)
                if market_cap > 0:
                    if market_cap >= 1e12:
                        market_cap_str = f"{market_cap/1e12:.1f}T"
                    elif market_cap >= 1e9:
                        market_cap_str = f"{market_cap/1e9:.1f}B"
                    elif market_cap >= 1e6:
                        market_cap_str = f"{market_cap/1e6:.1f}M"
                    else:
                        market_cap_str = f"{market_cap:,.0f}"
                    
                    st.metric(
                        label="시가총액",
                        value=market_cap_str,
                        help="Market Capitalization"
                    )
            
            with col2:
                pe_ratio = info.get('trailingPE', 0)
                if pe_ratio and pe_ratio > 0:
                    st.metric(
                        label="PER",
                        value=f"{pe_ratio:.1f}",
                        help="Price-to-Earnings Ratio"
                    )
            
            with col3:
                pb_ratio = info.get('priceToBook', 0)
                if pb_ratio and pb_ratio > 0:
                    st.metric(
                        label="PBR",
                        value=f"{pb_ratio:.2f}",
                        help="Price-to-Book Ratio"
                    )
            
            with col4:
                # 개선된 배당수익률 표시
                dividend_info = {
                    'dividend_yield': info.get('dividendYield', 0),
                    'dividend_rate': info.get('dividendRate', 0),
                    'payout_ratio': info.get('payoutRatio', 0)
                }
                
                # 배당 데이터 검증
                is_valid, errors = DataProcessor.validate_dividend_data(dividend_info)
                
                if is_valid and dividend_info['dividend_yield'] > 0:
                    # 배당 지표 계산
                    current_price = info.get('currentPrice', 0)
                    dividend_metrics = DataProcessor.calculate_dividend_metrics(dividend_info, current_price)
                    
                    st.metric(
                        label="배당수익률",
                        value=f"{dividend_metrics['dividend_yield_percent']:.2f}%",
                        help=f"Dividend Yield | 안정성: {dividend_metrics['dividend_sustainability']}"
                    )
                elif dividend_info['dividend_yield'] == 0:
                    st.metric(
                        label="배당수익률",
                        value="N/A",
                        help="배당 정보 없음"
                    )
                else:
                    st.metric(
                        label="배당수익률",
                        value="오류",
                        help=f"데이터 오류: {', '.join(errors)}"
                    )
            
            # 추가 밸류에이션 비율 표시
            st.markdown("#### 📊 밸류에이션 비율")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                ps_ratio = info.get('priceToSalesTrailing12Months', 0)
                if ps_ratio and ps_ratio > 0:
                    st.metric(
                        label="PSR",
                        value=f"{ps_ratio:.2f}",
                        help="Price-to-Sales Ratio"
                    )
            
            with col2:
                peg_ratio = info.get('pegRatio', 0)
                if peg_ratio and peg_ratio > 0:
                    st.metric(
                        label="PEG",
                        value=f"{peg_ratio:.2f}",
                        help="Price/Earnings to Growth Ratio"
                    )
            
            with col3:
                ev_ebitda = info.get('enterpriseToEbitda', 0)
                if ev_ebitda and ev_ebitda > 0:
                    st.metric(
                        label="EV/EBITDA",
                        value=f"{ev_ebitda:.1f}",
                        help="Enterprise Value to EBITDA"
                    )
            
            with col4:
                # 추가 지표가 있으면 여기에 표시
                st.metric(
                    label="통화",
                    value=financial_currency,
                    help="Financial Currency"
                )
            
            # 재무구조 정보 표시
            st.markdown("#### 💼 재무구조")
            
            # 재무 데이터 수집
            financial_data = {
                'total_revenue': info.get('totalRevenue', 0),
                'gross_profit': info.get('grossProfits', 0),
                'operating_income': info.get('operatingIncome', 0),
                'net_income': info.get('netIncomeToCommon', info.get('netIncome', 0)),
                'ebitda': info.get('ebitda', 0),
                'gross_margin': info.get('grossMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'profit_margin': info.get('profitMargins', 0),
                'ebitda_margin': info.get('ebitdaMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', 0)
            }
            
            # 재무 지표 계산
            financial_metrics = DataProcessor.calculate_financial_metrics(financial_data, financial_currency)
            
            # 매출액 및 이익 정보
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="매출액",
                    value=financial_metrics['formatted_revenue'],
                    help="Total Revenue"
                )
            
            with col2:
                st.metric(
                    label="영업이익",
                    value=financial_metrics['formatted_operating_income'],
                    help="Operating Income"
                )
            
            with col3:
                st.metric(
                    label="순이익",
                    value=financial_metrics['formatted_net_income'],
                    help="Net Income"
                )
            
            with col4:
                st.metric(
                    label="EBITDA",
                    value=financial_metrics['formatted_ebitda'],
                    help="Earnings Before Interest, Taxes, Depreciation and Amortization"
                )
            
            # 수익성 지표
            st.markdown("##### 📈 수익성 지표")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="매출액이익률",
                    value=f"{financial_metrics['gross_margin']:.1f}%",
                    help="Gross Margin"
                )
            
            with col2:
                st.metric(
                    label="영업이익률",
                    value=f"{financial_metrics['operating_margin']:.1f}%",
                    help="Operating Margin"
                )
            
            with col3:
                st.metric(
                    label="순이익률",
                    value=f"{financial_metrics['net_margin']:.1f}%",
                    help="Net Profit Margin"
                )
            
            with col4:
                st.metric(
                    label="EBITDA 마진",
                    value=f"{financial_metrics['ebitda_margin']:.1f}%",
                    help="EBITDA Margin"
                )
            
            # 성장률 지표
            st.markdown("##### 📊 성장률 지표")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="매출액 증가율",
                    value=f"{financial_metrics['revenue_growth']:.1f}%",
                    help="Revenue Growth Rate"
                )
            
            with col2:
                st.metric(
                    label="순이익 증가율",
                    value=f"{financial_metrics['earnings_growth']:.1f}%",
                    help="Earnings Growth Rate"
                )
            
            with col3:
                st.metric(
                    label="분기 순이익 증가율",
                    value=f"{financial_metrics['earnings_quarterly_growth']:.1f}%",
                    help="Quarterly Earnings Growth Rate"
                )
            
            # 배당 상세 정보 표시
            if dividend_info['dividend_yield'] > 0:
                st.markdown("#### 💰 배당 상세 정보")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # 통화별 연간 배당금 포맷팅
                    dividend_rate = dividend_info['dividend_rate']
                    formatted_dividend = DataProcessor.format_currency(dividend_rate, financial_currency)
                    st.metric(
                        label="연간 배당금",
                        value=formatted_dividend,
                        help="Annual Dividend per Share"
                    )
                
                with col2:
                    st.metric(
                        label="배당성향",
                        value=f"{dividend_info['payout_ratio']*100:.1f}%",
                        help="Payout Ratio"
                    )
                
                with col3:
                    ex_dividend_date = info.get('exDividendDate', None)
                    if ex_dividend_date:
                        from datetime import datetime
                        if isinstance(ex_dividend_date, (int, float)):
                            ex_date = datetime.fromtimestamp(ex_dividend_date).strftime('%Y-%m-%d')
                        else:
                            ex_date = str(ex_dividend_date)
                        st.metric(
                            label="배당락일",
                            value=ex_date,
                            help="Ex-Dividend Date"
                        )
                    else:
                        st.metric(
                            label="배당락일",
                            value="N/A",
                            help="Ex-Dividend Date"
                        )
            
            # 추가 정보
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 기본 정보")
                basic_info = {
                    "회사명": info.get('longName', 'N/A'),
                    "섹터": info.get('sector', 'N/A'),
                    "업종": info.get('industry', 'N/A'),
                    "거래소": info.get('exchange', 'N/A'),
                    "통화": info.get('currency', 'N/A')
                }
                
                for key, value in basic_info.items():
                    if value != 'N/A':
                        st.text(f"{key}: {value}")
            
            with col2:
                st.markdown("#### 📈 거래 정보")
                trading_info = {
                    "52주 최고가": f"{info.get('fiftyTwoWeekHigh', 0):.2f}",
                    "52주 최저가": f"{info.get('fiftyTwoWeekLow', 0):.2f}",
                    "평균 거래량": f"{info.get('averageVolume', 0):,}",
                    "현재 거래량": f"{info.get('volume', 0):,}",
                    "거래 시간": info.get('exchangeTimezoneName', 'N/A')
                }
                
                for key, value in trading_info.items():
                    if value != 'N/A' and value != '0.00':
                        st.text(f"{key}: {value}")
                        
        except Exception as e:
            st.warning(f"종목 정보를 가져올 수 없습니다: {str(e)}")

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
            if settings['market'] == 'custom' and settings.get('custom_ticker'):
                # 개별 종목 분석을 위한 커스텀 설정
                workflow = UnifiedFairValueWorkflow("custom", custom_ticker=settings['custom_ticker'])
            else:
                # 주요 지수 분석
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
            currency = summary.get('market_info', {}).get('currency', 'USD')

            st.metric(
                label="현재가",
                value=f"{price:,.2f} {currency}",
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
        self.render_main_header(settings)

        # 분석 컨트롤
        should_run_analysis = self.render_analysis_controls(settings)

        # 분석 실행
        if should_run_analysis:
            asyncio.run(self.run_analysis(settings))
            st.rerun()  # 결과 표시를 위해 리로드

        # 요약 지표
        if st.session_state.analysis_result:
            self.render_summary_metrics(st.session_state.analysis_result)
            
            # 개별 종목 추가 정보
            if settings.get('market') == 'custom' and settings.get('custom_ticker'):
                self.render_stock_info(st.session_state.analysis_result)

        # 메인 대시보드
        self.render_main_dashboard(st.session_state.analysis_result)

        # 푸터
        self.render_footer()

# 애플리케이션 실행
if __name__ == "__main__":
    app = FairValueApp()
    app.run()
