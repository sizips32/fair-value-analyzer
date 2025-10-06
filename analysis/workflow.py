"""
통합 워크플로우 모듈
전체 분석 과정을 단일 클래스로 통합하여 원클릭 실행 지원
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
import logging
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.technical import TechnicalAnalyzer
from analysis.monte_carlo import MonteCarloSimulator, SimulationResult
from data.collectors import DataCollector, DataProcessor
from config.settings import config_manager

@dataclass
class AnalysisStep:
    """분석 단계 정의"""
    name: str
    description: str
    function: callable
    required: bool = True
    progress_weight: float = 1.0

@dataclass
class AnalysisResult:
    """통합 분석 결과"""
    market_data: pd.DataFrame
    technical_analysis: Dict
    monte_carlo_result: SimulationResult
    fundamental_metrics: Dict
    signals: pd.DataFrame
    insights: List[str]
    risk_metrics: Dict
    summary: Dict

class UnifiedFairValueWorkflow:
    """통합 공정가치 분석 워크플로우"""

    def __init__(self, market_name: str = "kospi", custom_ticker: str = None):
        self.market_name = market_name
        self.custom_ticker = custom_ticker
        
        # 시장 설정 처리
        if market_name == "custom" and custom_ticker:
            # 개별 종목을 위한 커스텀 설정 생성
            from config.settings import MarketConfig
            self.market_config = MarketConfig(
                ticker=custom_ticker,
                name=custom_ticker,
                timezone="UTC",  # 기본값
                trading_hours="09:30-16:00",  # 기본값
                currency="USD"  # 기본값, 실제로는 yfinance에서 가져옴
            )
        else:
            self.market_config = config_manager.get_market_config(market_name)
            
        self.technical_config = config_manager.technical
        self.monte_carlo_config = config_manager.monte_carlo

        # 컴포넌트 초기화
        self.data_collector = DataCollector(self.market_config)
        self.technical_analyzer = TechnicalAnalyzer(self.technical_config)
        self.monte_carlo_simulator = MonteCarloSimulator(self.monte_carlo_config)
        self.data_processor = DataProcessor()

        # 로거 설정
        self.logger = logging.getLogger(__name__)

        # 분석 단계 정의
        self.analysis_steps = [
            AnalysisStep("데이터 수집", "시장 데이터 수집 및 전처리", self._collect_data, True, 1.5),
            AnalysisStep("데이터 검증", "데이터 품질 검증 및 정리", self._validate_data, True, 0.5),
            AnalysisStep("기술적 분석", "기술적 지표 계산 및 신호 생성", self._technical_analysis, True, 2.0),
            AnalysisStep("몬테카를로 시뮬레이션", "확률적 가격 예측", self._monte_carlo_analysis, True, 3.0),
            AnalysisStep("리스크 분석", "위험 지표 계산", self._risk_analysis, True, 1.0),
            AnalysisStep("인사이트 생성", "AI 기반 분석 해석", self._generate_insights, True, 1.0),
            AnalysisStep("결과 통합", "최종 결과 생성", self._compile_results, True, 1.0)
        ]

    async def run_complete_analysis(self,
                                  start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None,
                                  progress_callback: Optional[callable] = None) -> AnalysisResult:
        """
        전체 분석 실행

        Args:
            start_date: 분석 시작 날짜
            end_date: 분석 종료 날짜
            progress_callback: 진행률 콜백 함수

        Returns:
            AnalysisResult 객체
        """
        # 기본 날짜 설정
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365 * 2)  # 2년

        self.logger.info(f"Starting analysis for {self.market_name} from {start_date} to {end_date}")

        # 분석 컨텍스트 초기화
        context = {
            'start_date': start_date,
            'end_date': end_date,
            'market_data': None,
            'technical_analysis': None,
            'monte_carlo_result': None,
            'risk_metrics': None,
            'insights': [],
            'errors': []
        }

        # 전체 진행률 가중치 계산
        total_weight = sum(step.progress_weight for step in self.analysis_steps if step.required)
        completed_weight = 0

        try:
            for i, step in enumerate(self.analysis_steps):
                if not step.required:
                    continue

                self.logger.info(f"Executing step {i+1}/{len(self.analysis_steps)}: {step.name}")

                try:
                    # 단계 실행
                    result = await step.function(context)

                    # 결과를 컨텍스트에 저장
                    if result is not None:
                        context.update(result)

                    # 진행률 업데이트
                    completed_weight += step.progress_weight
                    progress = completed_weight / total_weight

                    if progress_callback:
                        progress_callback(progress, step.name, "완료")

                    self.logger.info(f"Step {step.name} completed successfully")

                except Exception as e:
                    error_msg = f"Error in step {step.name}: {str(e)}"
                    self.logger.error(error_msg)
                    context['errors'].append(error_msg)

                    if progress_callback:
                        progress_callback(completed_weight / total_weight, step.name, f"오류: {str(e)}")

                    # 필수 단계에서 오류 발생 시 중단
                    if step.required:
                        raise

            # 최종 결과 생성
            return self._create_final_result(context)

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise

    async def _collect_data(self, context: Dict) -> Dict:
        """데이터 수집 단계"""
        try:
            # 비동기적으로 데이터 수집
            loop = asyncio.get_event_loop()
            market_data = await loop.run_in_executor(
                None,
                self.data_collector.fetch_historical_data,
                context['start_date'],
                context['end_date']
            )

            # 실시간 데이터도 함께 수집
            real_time_data = await loop.run_in_executor(
                None,
                self.data_collector.fetch_real_time_data
            )
            
            # 개별 종목인 경우 통화 정보 업데이트
            if self.market_name == "custom" and real_time_data:
                currency = real_time_data.get('currency', 'USD')
                if hasattr(self.market_config, 'currency'):
                    self.market_config.currency = currency

            return {
                'market_data': market_data,
                'real_time_data': real_time_data
            }

        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            raise

    async def _validate_data(self, context: Dict) -> Dict:
        """데이터 검증 단계"""
        market_data = context['market_data']

        if market_data is None or market_data.empty:
            raise ValueError("Market data is empty")

        # 데이터 품질 검증
        is_valid, errors = self.data_processor.validate_data(market_data)

        if not is_valid:
            self.logger.warning(f"Data validation issues: {errors}")
            context['errors'].extend(errors)

        # 파생 변수 생성
        enhanced_data = self.data_processor.create_features(market_data)

        return {'market_data': enhanced_data}

    async def _technical_analysis(self, context: Dict) -> Dict:
        """기술적 분석 단계"""
        market_data = context['market_data']

        # 모든 기술적 지표 계산
        enriched_data = self.technical_analyzer.calculate_all_indicators(market_data)

        # 매매 신호 생성
        signals = self.technical_analyzer.generate_signals(enriched_data)

        # 최신 분석 결과
        latest_analysis = self.technical_analyzer.get_latest_analysis(enriched_data)

        return {
            'market_data': enriched_data,
            'signals': signals,
            'technical_analysis': latest_analysis
        }

    async def _monte_carlo_analysis(self, context: Dict) -> Dict:
        """몬테카를로 시뮬레이션 단계"""
        market_data = context['market_data']

        # 비동기적으로 시뮬레이션 실행
        loop = asyncio.get_event_loop()
        monte_carlo_result = await loop.run_in_executor(
            None,
            self.monte_carlo_simulator.run_simulation,
            market_data
        )

        # 시나리오 분석
        scenarios = {
            'bull_market': {'volatility_multiplier': 0.8},
            'bear_market': {'volatility_multiplier': 1.5},
            'high_volatility': {'volatility_multiplier': 2.0}
        }

        scenario_results = await loop.run_in_executor(
            None,
            self.monte_carlo_simulator.scenario_analysis,
            market_data,
            scenarios
        )

        return {
            'monte_carlo_result': monte_carlo_result,
            'scenario_analysis': scenario_results
        }

    async def _risk_analysis(self, context: Dict) -> Dict:
        """리스크 분석 단계"""
        market_data = context['market_data']
        monte_carlo_result = context['monte_carlo_result']

        # VaR 및 CVaR 계산
        returns = market_data['Returns'].dropna()
        current_price = market_data['Close'].iloc[-1]

        risk_metrics = {
            # 기본 리스크 지표
            'volatility_daily': returns.std(),
            'volatility_annual': returns.std() * np.sqrt(252),
            'var_95_historical': returns.quantile(0.05),
            'var_99_historical': returns.quantile(0.01),
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),
            'cvar_99': returns[returns <= returns.quantile(0.01)].mean(),

            # 몬테카를로 기반 리스크 지표
            'var_95_simulation': monte_carlo_result.statistics['var_95'],
            'var_99_simulation': monte_carlo_result.statistics['var_99'],
            'expected_shortfall_95': monte_carlo_result.statistics['expected_shortfall_95'],
            'expected_shortfall_99': monte_carlo_result.statistics['expected_shortfall_99'],

            # 추가 리스크 지표
            'maximum_drawdown': self._calculate_maximum_drawdown(market_data),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': self._calculate_calmar_ratio(market_data),

            # 상관관계 및 베타 (향후 확장)
            'beta': None,  # 시장 지수와의 베타
            'correlation_spy': None  # S&P500과의 상관관계
        }

        return {'risk_metrics': risk_metrics}

    async def _generate_insights(self, context: Dict) -> Dict:
        """인사이트 생성 단계"""
        insights = []

        # 기술적 분석 인사이트
        tech_analysis = context['technical_analysis']
        if tech_analysis:
            insights.extend(self._technical_insights(tech_analysis))

        # 몬테카를로 인사이트
        mc_result = context['monte_carlo_result']
        if mc_result:
            insights.extend(self._monte_carlo_insights(mc_result))

        # 리스크 인사이트
        risk_metrics = context['risk_metrics']
        if risk_metrics:
            insights.extend(self._risk_insights(risk_metrics))

        # 종합 투자 추천
        recommendation = self._generate_investment_recommendation(context)
        insights.append(f"💡 종합 투자 추천: {recommendation}")

        return {'insights': insights}

    async def _compile_results(self, context: Dict) -> Dict:
        """결과 통합 단계"""
        # 요약 정보 생성
        summary = self._create_summary(context)

        return {'summary': summary}

    def _technical_insights(self, tech_analysis: Dict) -> List[str]:
        """기술적 분석 기반 인사이트"""
        insights = []
        indicators = tech_analysis.get('technical_indicators', {})
        signals = tech_analysis.get('signals', {})

        # RSI 인사이트
        rsi = indicators.get('rsi', 50)
        if rsi > 70:
            insights.append("⚠️ RSI 과매수 구간: 단기 조정 가능성 높음")
        elif rsi < 30:
            insights.append("🟢 RSI 과매도 구간: 반등 가능성 존재")

        # MACD 인사이트
        macd_signal = signals.get('macd_signal', 0)
        if macd_signal == 1:
            insights.append("📈 MACD 상승 신호: 상승 모멘텀 증가")
        elif macd_signal == -1:
            insights.append("📉 MACD 하락 신호: 하락 모멘텀 증가")

        # 볼린저 밴드 인사이트
        bb_position = indicators.get('bb_position', 0.5)
        if bb_position > 0.8:
            insights.append("🔴 볼린저 밴드 상단 근접: 과매수 상태")
        elif bb_position < 0.2:
            insights.append("🟢 볼린저 밴드 하단 근접: 과매도 상태")

        # 종합 신호 강도
        signal_strength = signals.get('signal_strength', 0)
        if abs(signal_strength) > 0.6:
            direction = "상승" if signal_strength > 0 else "하락"
            insights.append(f"🎯 강한 {direction} 신호 감지 (강도: {abs(signal_strength):.2f})")

        return insights

    def _monte_carlo_insights(self, mc_result: SimulationResult) -> List[str]:
        """몬테카를로 시뮬레이션 기반 인사이트"""
        insights = []
        stats = mc_result.statistics

        # 예상 수익률
        expected_return = stats['mean_return'] * 100
        insights.append(f"📊 예상 수익률: {expected_return:.1f}%")

        # 상승 확률
        positive_prob = stats['positive_return_prob'] * 100
        insights.append(f"📈 상승 확률: {positive_prob:.1f}%")

        # 신뢰구간
        for level, ci in mc_result.confidence_intervals.items():
            if level == "95%":
                insights.append(f"🎯 95% 신뢰구간: {ci['lower']:.0f} ~ {ci['upper']:.0f}")

        # 리스크 평가
        var_95 = stats['var_95']
        current_price = stats['mean_price'] / (1 + stats['mean_return'])  # 역산
        risk_level = abs(var_95 - current_price) / current_price

        if risk_level > 0.2:
            insights.append("⚠️ 높은 리스크: 신중한 투자 필요")
        elif risk_level < 0.1:
            insights.append("🟢 낮은 리스크: 상대적으로 안정적")

        return insights

    def _risk_insights(self, risk_metrics: Dict) -> List[str]:
        """리스크 지표 기반 인사이트"""
        insights = []

        # 변동성 평가
        annual_vol = risk_metrics.get('volatility_annual', 0) * 100
        if annual_vol > 30:
            insights.append(f"⚠️ 높은 변동성: 연간 {annual_vol:.1f}%")
        elif annual_vol < 15:
            insights.append(f"🟢 낮은 변동성: 연간 {annual_vol:.1f}%")

        # 샤프 비율
        sharpe = risk_metrics.get('sharpe_ratio', 0)
        if sharpe > 1:
            insights.append(f"✨ 우수한 위험대비수익률 (샤프비율: {sharpe:.2f})")
        elif sharpe < 0:
            insights.append(f"⚠️ 낮은 위험대비수익률 (샤프비율: {sharpe:.2f})")

        # 최대 손실률
        max_dd = risk_metrics.get('maximum_drawdown', 0) * 100
        if abs(max_dd) > 20:
            insights.append(f"🔴 높은 최대손실률: {abs(max_dd):.1f}%")

        return insights

    def _generate_investment_recommendation(self, context: Dict) -> str:
        """종합 투자 추천 생성"""
        # 점수 시스템으로 종합 판단
        score = 0
        factors = []

        # 기술적 분석 점수
        tech_signals = context.get('technical_analysis', {}).get('signals', {})
        composite_signal = tech_signals.get('composite_signal', 0)
        signal_strength = tech_signals.get('signal_strength', 0)

        tech_score = composite_signal * abs(signal_strength)
        score += tech_score * 0.4
        factors.append(f"기술적분석: {tech_score:.2f}")

        # 몬테카를로 점수
        mc_result = context.get('monte_carlo_result')
        if mc_result:
            positive_prob = mc_result.statistics['positive_return_prob']
            expected_return = mc_result.statistics['mean_return']
            mc_score = (positive_prob - 0.5) * 2 + expected_return
            score += mc_score * 0.4
            factors.append(f"확률분석: {mc_score:.2f}")

        # 리스크 점수
        risk_metrics = context.get('risk_metrics', {})
        sharpe = risk_metrics.get('sharpe_ratio', 0)
        risk_score = min(sharpe, 2) / 2  # 정규화
        score += risk_score * 0.2
        factors.append(f"리스크조정: {risk_score:.2f}")

        # 추천 등급 결정
        if score > 0.6:
            recommendation = "적극 매수"
        elif score > 0.3:
            recommendation = "매수"
        elif score > -0.3:
            recommendation = "관망"
        elif score > -0.6:
            recommendation = "매도"
        else:
            recommendation = "적극 매도"

        return f"{recommendation} (종합점수: {score:.2f})"

    def _create_summary(self, context: Dict) -> Dict:
        """요약 정보 생성"""
        market_data = context.get('market_data')
        tech_analysis = context.get('technical_analysis', {})
        mc_result = context.get('monte_carlo_result')
        risk_metrics = context.get('risk_metrics', {})

        if market_data is None or market_data.empty:
            return {}

        current_price = market_data['Close'].iloc[-1]
        price_change = market_data['Close'].iloc[-1] - market_data['Close'].iloc[-2] if len(market_data) > 1 else 0
        price_change_pct = (price_change / market_data['Close'].iloc[-2] * 100) if len(market_data) > 1 else 0

        summary = {
            'market_info': {
                'market_name': self.market_config.name,
                'ticker': self.market_config.ticker,
                'current_price': current_price,
                'price_change': price_change,
                'price_change_percent': price_change_pct,
                'currency': getattr(self.market_config, 'currency', 'USD'),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'technical_summary': {
                'rsi': tech_analysis.get('technical_indicators', {}).get('rsi', 0),
                'composite_signal': tech_analysis.get('signals', {}).get('composite_signal', 0),
                'signal_strength': tech_analysis.get('signals', {}).get('signal_strength', 0)
            },
            'prediction_summary': {
                'expected_return': mc_result.statistics['mean_return'] * 100 if mc_result else 0,
                'positive_probability': mc_result.statistics['positive_return_prob'] * 100 if mc_result else 0,
                'confidence_95_lower': mc_result.confidence_intervals['95%']['lower'] if mc_result else 0,
                'confidence_95_upper': mc_result.confidence_intervals['95%']['upper'] if mc_result else 0
            },
            'risk_summary': {
                'annual_volatility': risk_metrics.get('volatility_annual', 0) * 100,
                'sharpe_ratio': risk_metrics.get('sharpe_ratio', 0),
                'var_95': risk_metrics.get('var_95_simulation', 0),
                'maximum_drawdown': abs(risk_metrics.get('maximum_drawdown', 0)) * 100
            },
            'data_quality': {
                'data_points': len(market_data),
                'date_range': f"{market_data.index[0].strftime('%Y-%m-%d')} ~ {market_data.index[-1].strftime('%Y-%m-%d')}",
                'errors': context.get('errors', [])
            }
        }

        return summary

    def _create_final_result(self, context: Dict) -> AnalysisResult:
        """최종 결과 객체 생성"""
        return AnalysisResult(
            market_data=context.get('market_data', pd.DataFrame()),
            technical_analysis=context.get('technical_analysis', {}),
            monte_carlo_result=context.get('monte_carlo_result'),
            fundamental_metrics={},  # 향후 확장
            signals=context.get('signals', pd.DataFrame()),
            insights=context.get('insights', []),
            risk_metrics=context.get('risk_metrics', {}),
            summary=context.get('summary', {})
        )

    # 헬퍼 함수들
    def _calculate_maximum_drawdown(self, data: pd.DataFrame) -> float:
        """최대 손실률 계산"""
        prices = data['Close']
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return drawdown.min()

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """샤프 비율 계산"""
        excess_returns = returns - risk_free_rate/252
        return excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """소르티노 비율 계산"""
        excess_returns = returns - risk_free_rate/252
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.001

        return excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

    def _calculate_calmar_ratio(self, data: pd.DataFrame) -> float:
        """칼마 비율 계산"""
        returns = data['Close'].pct_change().dropna()
        annual_return = returns.mean() * 252
        max_drawdown = abs(self._calculate_maximum_drawdown(data))

        return annual_return / max_drawdown if max_drawdown > 0 else 0
