"""
몬테카를로 시뮬레이션 모듈
기하 브라운 운동 및 다양한 확률 모델을 통한 가격 예측
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MonteCarloConfig

@dataclass
class SimulationResult:
    """시뮬레이션 결과 데이터 클래스"""
    paths: np.ndarray
    final_prices: np.ndarray
    returns: np.ndarray
    statistics: Dict
    confidence_intervals: Dict

class MonteCarloSimulator:
    """몬테카를로 시뮬레이션 엔진"""

    def __init__(self, config: MonteCarloConfig):
        self.config = config
        self.simulation_methods = {
            'geometric_brownian': self._geometric_brownian_motion,
            'jump_diffusion': self._jump_diffusion,
            'heston_stochastic': self._heston_stochastic_volatility,
            'variance_gamma': self._variance_gamma
        }

    def run_simulation(self,
                      data: pd.DataFrame,
                      method: Optional[str] = None) -> SimulationResult:
        """
        몬테카를로 시뮬레이션 실행

        Args:
            data: 과거 가격 데이터
            method: 시뮬레이션 방법 (기본값: config에서 설정)

        Returns:
            SimulationResult 객체
        """
        method = method or self.config.method

        if method not in self.simulation_methods:
            raise ValueError(f"Unknown simulation method: {method}")

        # 데이터 전처리
        returns = data['Close'].pct_change().dropna()
        current_price = data['Close'].iloc[-1]

        # 통계 파라미터 계산
        params = self._calculate_parameters(returns)

        # 시뮬레이션 실행
        simulation_func = self.simulation_methods[method]
        paths = simulation_func(current_price, params)

        # 결과 분석
        final_prices = paths[:, -1]
        result_returns = (final_prices - current_price) / current_price

        statistics = self._calculate_statistics(final_prices, current_price)
        confidence_intervals = self._calculate_confidence_intervals(final_prices)

        return SimulationResult(
            paths=paths,
            final_prices=final_prices,
            returns=result_returns,
            statistics=statistics,
            confidence_intervals=confidence_intervals
        )

    def _calculate_parameters(self, returns: pd.Series) -> Dict:
        """통계 파라미터 계산"""
        return {
            'mean': returns.mean(),
            'std': returns.std(),
            'skew': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'drift': returns.mean() * 252,  # 연간화
            'volatility': returns.std() * np.sqrt(252),  # 연간화
            'var_95': returns.quantile(0.05),
            'var_99': returns.quantile(0.01)
        }

    def _geometric_brownian_motion(self,
                                 current_price: float,
                                 params: Dict) -> np.ndarray:
        """기하 브라운 운동 시뮬레이션"""
        dt = 1/252  # 일간 데이터
        drift = params['drift']
        vol = params['volatility']

        # 병렬 처리를 위한 시뮬레이션 분할
        num_cores = min(mp.cpu_count(), 4)
        sims_per_core = self.config.simulations // num_cores

        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = []
            for _ in range(num_cores):
                future = executor.submit(
                    self._gbm_single_batch,
                    current_price, drift, vol, dt, sims_per_core
                )
                futures.append(future)

            results = [future.result() for future in futures]

        return np.vstack(results)

    def _gbm_single_batch(self,
                         current_price: float,
                         drift: float,
                         vol: float,
                         dt: float,
                         n_sims: int) -> np.ndarray:
        """단일 배치 GBM 시뮬레이션"""
        np.random.seed()  # 각 프로세스마다 다른 시드

        # 랜덤 워크 생성
        random_shocks = np.random.normal(
            0, 1, (n_sims, self.config.forecast_days)
        )

        # 가격 경로 계산
        paths = np.zeros((n_sims, self.config.forecast_days + 1))
        paths[:, 0] = current_price

        for t in range(1, self.config.forecast_days + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * random_shocks[:, t-1]
            )

        return paths

    def _jump_diffusion(self, current_price: float, params: Dict) -> np.ndarray:
        """점프 확산 모델 (Merton Jump Diffusion)"""
        dt = 1/252
        drift = params['drift']
        vol = params['volatility']

        # 점프 파라미터 (경험적 설정)
        jump_intensity = 0.1  # 연간 평균 점프 횟수
        jump_mean = -0.02     # 점프 크기 평균
        jump_std = 0.05       # 점프 크기 표준편차

        paths = np.zeros((self.config.simulations, self.config.forecast_days + 1))
        paths[:, 0] = current_price

        for t in range(1, self.config.forecast_days + 1):
            # 확산 부분
            dW = np.random.normal(0, np.sqrt(dt), self.config.simulations)
            diffusion = (drift - 0.5 * vol**2) * dt + vol * dW

            # 점프 부분
            jump_occurs = np.random.poisson(jump_intensity * dt, self.config.simulations)
            jump_sizes = np.where(
                jump_occurs > 0,
                np.random.normal(jump_mean, jump_std, self.config.simulations),
                0
            )

            paths[:, t] = paths[:, t-1] * np.exp(diffusion + jump_sizes)

        return paths

    def _heston_stochastic_volatility(self, current_price: float, params: Dict) -> np.ndarray:
        """헤스턴 확률 변동성 모델"""
        dt = 1/252

        # 헤스턴 모델 파라미터 (경험적 설정)
        kappa = 2.0      # 평균 회귀 속도
        theta = 0.04     # 장기 평균 분산
        sigma_v = 0.3    # 분산의 변동성
        rho = -0.7       # 상관계수
        v0 = params['volatility']**2  # 초기 분산

        paths = np.zeros((self.config.simulations, self.config.forecast_days + 1))
        variance = np.zeros((self.config.simulations, self.config.forecast_days + 1))

        paths[:, 0] = current_price
        variance[:, 0] = v0

        for t in range(1, self.config.forecast_days + 1):
            # 상관된 브라운 운동
            dW1 = np.random.normal(0, np.sqrt(dt), self.config.simulations)
            dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), self.config.simulations)

            # 분산 과정 (CIR 과정)
            variance[:, t] = np.maximum(
                variance[:, t-1] + kappa * (theta - variance[:, t-1]) * dt +
                sigma_v * np.sqrt(variance[:, t-1]) * dW2,
                0.001  # 분산이 음수가 되지 않도록
            )

            # 가격 과정
            paths[:, t] = paths[:, t-1] * np.exp(
                (params['drift'] - 0.5 * variance[:, t-1]) * dt +
                np.sqrt(variance[:, t-1]) * dW1
            )

        return paths

    def _variance_gamma(self, current_price: float, params: Dict) -> np.ndarray:
        """분산-감마 모델"""
        # VG 모델 파라미터 추정
        nu = 0.2  # 분산 파라미터

        paths = np.zeros((self.config.simulations, self.config.forecast_days + 1))
        paths[:, 0] = current_price

        for t in range(1, self.config.forecast_days + 1):
            # 감마 분포에서 시간 증분 샘플링
            gamma_time = np.random.gamma(1/nu, nu, self.config.simulations)

            # VG 과정
            vg_increments = np.random.normal(
                params['mean'] * gamma_time,
                params['std'] * np.sqrt(gamma_time),
                self.config.simulations
            )

            paths[:, t] = paths[:, t-1] * np.exp(vg_increments)

        return paths

    def _calculate_statistics(self, final_prices: np.ndarray, current_price: float) -> Dict:
        """시뮬레이션 결과 통계 계산"""
        returns = (final_prices - current_price) / current_price

        return {
            'mean_price': np.mean(final_prices),
            'median_price': np.median(final_prices),
            'std_price': np.std(final_prices),
            'min_price': np.min(final_prices),
            'max_price': np.max(final_prices),
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'positive_return_prob': np.mean(returns > 0),
            'skewness': stats.skew(final_prices),
            'kurtosis': stats.kurtosis(final_prices),
            'var_95': np.percentile(final_prices, 5),
            'var_99': np.percentile(final_prices, 1),
            'expected_shortfall_95': np.mean(final_prices[final_prices <= np.percentile(final_prices, 5)]),
            'expected_shortfall_99': np.mean(final_prices[final_prices <= np.percentile(final_prices, 1)])
        }

    def _calculate_confidence_intervals(self, final_prices: np.ndarray) -> Dict:
        """신뢰구간 계산"""
        confidence_intervals = {}

        for level in self.config.confidence_levels:
            alpha = (100 - level) / 2
            lower = np.percentile(final_prices, alpha)
            upper = np.percentile(final_prices, 100 - alpha)

            confidence_intervals[f'{level}%'] = {
                'lower': lower,
                'upper': upper,
                'range': upper - lower
            }

        return confidence_intervals

    def scenario_analysis(self,
                         data: pd.DataFrame,
                         scenarios: Dict[str, Dict]) -> Dict[str, SimulationResult]:
        """시나리오 분석"""
        results = {}

        for scenario_name, scenario_params in scenarios.items():
            # 임시로 설정 변경
            original_config = self.config

            # 시나리오별 파라미터 적용
            temp_config = MonteCarloConfig(
                simulations=scenario_params.get('simulations', self.config.simulations),
                forecast_days=scenario_params.get('forecast_days', self.config.forecast_days),
                confidence_levels=self.config.confidence_levels,
                method=scenario_params.get('method', self.config.method)
            )

            self.config = temp_config

            # 데이터 조정 (변동성 조정 등)
            adjusted_data = data.copy()
            if 'volatility_multiplier' in scenario_params:
                returns = adjusted_data['Close'].pct_change()
                returns *= scenario_params['volatility_multiplier']

                # 조정된 수익률로 가격 재구성
                adjusted_prices = [adjusted_data['Close'].iloc[0]]
                for ret in returns[1:]:
                    adjusted_prices.append(adjusted_prices[-1] * (1 + ret))

                adjusted_data['Close'] = adjusted_prices

            results[scenario_name] = self.run_simulation(adjusted_data)

            # 원래 설정 복원
            self.config = original_config

        return results

    def backtesting_analysis(self,
                           data: pd.DataFrame,
                           prediction_window: int = 30,
                           step_size: int = 5) -> Dict:
        """백테스팅 분석"""
        results = {
            'accuracy_scores': [],
            'prediction_errors': [],
            'confidence_coverage': {level: [] for level in self.config.confidence_levels}
        }

        for start_idx in range(len(data) - prediction_window - self.config.forecast_days,
                              len(data) - prediction_window, step_size):

            if start_idx < self.config.forecast_days:
                continue

            # 훈련 데이터
            train_data = data.iloc[start_idx - self.config.forecast_days:start_idx]

            # 실제 가격 (예측 대상)
            actual_price = data['Close'].iloc[start_idx + prediction_window]

            # 시뮬레이션 실행
            sim_result = self.run_simulation(train_data)

            # 정확도 평가
            predicted_price = sim_result.statistics['mean_price']
            error = abs(predicted_price - actual_price) / actual_price

            results['accuracy_scores'].append(1 - error)
            results['prediction_errors'].append(error)

            # 신뢰구간 적중률
            for level in self.config.confidence_levels:
                ci = sim_result.confidence_intervals[f'{level}%']
                in_interval = ci['lower'] <= actual_price <= ci['upper']
                results['confidence_coverage'][level].append(in_interval)

        # 종합 성능 지표
        results['overall_accuracy'] = np.mean(results['accuracy_scores'])
        results['mean_absolute_error'] = np.mean(results['prediction_errors'])

        for level in self.config.confidence_levels:
            coverage_rate = np.mean(results['confidence_coverage'][level])
            results[f'coverage_{level}%'] = coverage_rate

        return results

    def sensitivity_analysis(self, data: pd.DataFrame) -> Dict:
        """민감도 분석"""
        base_result = self.run_simulation(data)
        sensitivity_results = {}

        # 변동성 민감도
        volatility_multipliers = [0.5, 0.75, 1.25, 1.5, 2.0]
        sensitivity_results['volatility'] = {}

        for mult in volatility_multipliers:
            scenario = {'volatility_multiplier': mult}
            result = self.scenario_analysis(data, {f'vol_{mult}': scenario})
            sensitivity_results['volatility'][mult] = result[f'vol_{mult}'].statistics

        # 시뮬레이션 횟수 민감도
        simulation_counts = [1000, 5000, 20000, 50000]
        sensitivity_results['simulations'] = {}

        for count in simulation_counts:
            if count <= self.config.simulations:
                scenario = {'simulations': count}
                result = self.scenario_analysis(data, {f'sim_{count}': scenario})
                sensitivity_results['simulations'][count] = result[f'sim_{count}'].statistics

        return sensitivity_results

class MonteCarloPortfolio:
    """포트폴리오 몬테카를로 시뮬레이션"""

    def __init__(self, config: MonteCarloConfig):
        self.config = config

    def simulate_portfolio(self,
                         portfolio_data: Dict[str, pd.DataFrame],
                         weights: Dict[str, float]) -> Dict:
        """포트폴리오 시뮬레이션"""
        if abs(sum(weights.values()) - 1.0) > 0.001:
            raise ValueError("포트폴리오 가중치의 합이 1이 아닙니다")

        # 각 자산별 시뮬레이션
        asset_simulations = {}
        simulators = {}

        for asset, data in portfolio_data.items():
            simulator = MonteCarloSimulator(self.config)
            result = simulator.run_simulation(data)
            asset_simulations[asset] = result
            simulators[asset] = simulator

        # 포트폴리오 가치 계산
        portfolio_paths = np.zeros((self.config.simulations, self.config.forecast_days + 1))

        # 초기 포트폴리오 가치를 1로 정규화
        portfolio_paths[:, 0] = 1.0

        for t in range(1, self.config.forecast_days + 1):
            portfolio_value = 0
            for asset, weight in weights.items():
                if asset in asset_simulations:
                    # 각 자산의 수익률
                    asset_returns = (asset_simulations[asset].paths[:, t] /
                                   asset_simulations[asset].paths[:, 0])
                    portfolio_value += weight * asset_returns

            portfolio_paths[:, t] = portfolio_value

        # 포트폴리오 통계
        final_returns = portfolio_paths[:, -1] - 1

        portfolio_stats = {
            'mean_return': np.mean(final_returns),
            'std_return': np.std(final_returns),
            'sharpe_ratio': np.mean(final_returns) / np.std(final_returns) if np.std(final_returns) > 0 else 0,
            'var_95': np.percentile(final_returns, 5),
            'var_99': np.percentile(final_returns, 1),
            'max_drawdown': self._calculate_max_drawdown(portfolio_paths),
            'positive_return_prob': np.mean(final_returns > 0)
        }

        return {
            'portfolio_paths': portfolio_paths,
            'final_returns': final_returns,
            'statistics': portfolio_stats,
            'asset_contributions': self._calculate_asset_contributions(asset_simulations, weights)
        }

    def _calculate_max_drawdown(self, paths: np.ndarray) -> float:
        """최대 손실률 계산"""
        max_drawdowns = []

        for i in range(paths.shape[0]):
            path = paths[i, :]
            peak = np.maximum.accumulate(path)
            drawdown = (path - peak) / peak
            max_drawdowns.append(np.min(drawdown))

        return np.mean(max_drawdowns)

    def _calculate_asset_contributions(self,
                                     asset_simulations: Dict,
                                     weights: Dict[str, float]) -> Dict:
        """자산별 기여도 계산"""
        contributions = {}

        for asset, weight in weights.items():
            if asset in asset_simulations:
                asset_returns = asset_simulations[asset].returns
                contribution = weight * np.mean(asset_returns)
                contributions[asset] = {
                    'weight': weight,
                    'expected_return': np.mean(asset_returns),
                    'contribution': contribution,
                    'risk_contribution': weight * np.std(asset_returns)
                }

        return contributions