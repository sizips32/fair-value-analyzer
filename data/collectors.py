"""
데이터 수집 모듈
다양한 데이터 소스로부터 금융 데이터를 수집하고 전처리
"""

import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from utils.logging_config import get_logger
try:
    import streamlit as st
except ImportError:
    # Streamlit이 없는 환경에서도 작동하도록
    st = None

try:
    from ..config.settings import MarketConfig
except ImportError:
    from config.settings import MarketConfig


class DataCollector:
    """통합 데이터 수집기"""

    def __init__(self, market_config: MarketConfig, use_cache: bool = True):
        self.config = market_config
        self.logger = get_logger(__name__)
        self.cache = DataCache() if use_cache else None

    def fetch_historical_data(self,
                            start_date: datetime,
                            end_date: datetime,
                            interval: str = "1d") -> pd.DataFrame:
        """
        과거 데이터 수집

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            interval: 데이터 간격 (1d, 1h, 5m 등)

        Returns:
            OHLCV 데이터프레임
        """
        # 캐시 확인
        if self.cache:
            cache_key = self.cache.get_cache_key(
                f"{self.config.ticker}_{interval}", start_date, end_date
            )
            cached_data = self.cache.load_from_cache(cache_key)
            if cached_data is not None:
                self.logger.info(f"Using cached data for {self.config.ticker}")
                return cached_data

        try:
            ticker = yf.Ticker(self.config.ticker)
            data = ticker.history(start=start_date, end=end_date, interval=interval)

            if data.empty:
                raise ValueError(f"No data found for {self.config.ticker}")

            # 데이터 정리
            data = self._clean_data(data)

            # 캐시 저장
            if self.cache:
                self.cache.save_to_cache(cache_key, data)

            self.logger.info(f"Successfully fetched {len(data)} rows for {self.config.ticker}")
            return data

        except ValueError as e:
            self.logger.error(f"No data available for {self.config.ticker}: {e}")
            raise
        except ConnectionError as e:
            self.logger.error(f"Network error fetching data for {self.config.ticker}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error fetching data for {self.config.ticker}: {e}", exc_info=True)
            raise

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """데이터 정리 및 전처리"""
        # 열 이름 정리 - yfinance에서 반환되는 실제 열 이름 확인
        self.logger.debug(f"Original columns: {list(data.columns)}")

        # yfinance에서 반환되는 표준 열 이름에 맞춰 조정
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if len(data.columns) != len(expected_columns):
            # 실제 데이터의 열 이름을 그대로 사용하되, 필요한 열만 선택
            available_cols = [col for col in expected_columns if col in data.columns]
            if len(available_cols) >= 4:  # 최소한 OHLC는 있어야 함
                data = data[available_cols]
            else:
                raise ValueError(f"Required columns not found. Available: {list(data.columns)}")
        else:
            data.columns = expected_columns

        # 결측값 처리
        data = data.dropna()

        # 이상치 제거 (3 시그마 규칙)
        for col in ['Open', 'High', 'Low', 'Close']:
            mean = data[col].mean()
            std = data[col].std()
            data = data[abs(data[col] - mean) <= 3 * std]

        # 수익률 계산
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))

        # 변동성 계산 (20일 롤링)
        data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)

        return data.dropna()

    def fetch_real_time_data(self) -> Dict:
        """실시간 데이터 수집"""
        try:
            ticker = yf.Ticker(self.config.ticker)
            info = ticker.info

            # 현재 가격 정보
            # Note: 지수(indices)의 경우 currentPrice가 없으므로 regularMarketPrice 사용
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            previous_close = info.get('previousClose', 0)

            current_data = {
                'current_price': current_price,
                'previous_close': previous_close,
                'open': info.get('open') or info.get('regularMarketOpen', 0),
                'day_high': info.get('dayHigh') or info.get('regularMarketDayHigh', 0),
                'day_low': info.get('dayLow') or info.get('regularMarketDayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'financial_currency': info.get('financialCurrency', info.get('currency', 'USD')),
                'long_name': info.get('longName', self.config.ticker),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'dividend_yield': info.get('dividendYield', 0),
                'dividend_rate': info.get('dividendRate', 0),
                'ex_dividend_date': info.get('exDividendDate', None),
                'payout_ratio': info.get('payoutRatio', 0),
                # 밸류에이션 비율 추가
                'pe_ratio': info.get('trailingPE', 0),
                'pb_ratio': info.get('priceToBook', 0),
                'ps_ratio': info.get('priceToSalesTrailing12Months', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'ev_ebitda': info.get('enterpriseToEbitda', 0),
                # 재무구조 정보 추가
                'total_revenue': info.get('totalRevenue', 0),
                'gross_profit': info.get('grossProfits', 0),
                'operating_income': info.get('operatingIncome', 0),
                'net_income': info.get('netIncomeToCommon', info.get('netIncome', 0)),
                'ebitda': info.get('ebitda', 0),
                # 수익성 지표
                'gross_margin': info.get('grossMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
                'profit_margin': info.get('profitMargins', 0),
                'ebitda_margin': info.get('ebitdaMargins', 0),
                # 성장률 지표
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth', 0),
                'revenue_per_share': info.get('revenuePerShare', 0),
                'timestamp': datetime.now()
            }

            # 변화량 계산
            if previous_close > 0:
                change = current_price - previous_close
                change_percent = (change / previous_close) * 100
                current_data['change'] = change
                current_data['change_percent'] = change_percent
            else:
                current_data['change'] = 0
                current_data['change_percent'] = 0

            self.logger.debug(f"Real-time data fetched: {self.config.ticker} = {current_price} (prev: {previous_close})")
            return current_data

        except ValueError as e:
            self.logger.error(f"Invalid data in real-time fetch: {e}")
            return {}
        except ConnectionError as e:
            self.logger.error(f"Network error fetching real-time data: {e}")
            return {}
        except KeyError as e:
            self.logger.error(f"Missing data field in real-time response: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Error fetching real-time data: {e}", exc_info=True)
            return {}

class DataProcessor:
    """데이터 전처리 및 검증"""

    @staticmethod
    def validate_dividend_data(dividend_info: Dict) -> Tuple[bool, List[str]]:
        """
        배당 관련 데이터 검증
        
        Args:
            dividend_info: 배당 정보 딕셔너리
            
        Returns:
            (유효성 여부, 오류 메시지 리스트)
        """
        errors = []
        
        dividend_yield = dividend_info.get('dividend_yield', 0)
        dividend_rate = dividend_info.get('dividend_rate', 0)
        payout_ratio = dividend_info.get('payout_ratio', 0)
        
        # 배당수익률 검증 (0-100% 범위)
        # yfinance의 dividendYield는 이미 백분율로 제공됨 (0-100 범위)
        if dividend_yield < 0 or dividend_yield > 100:
            errors.append(f"배당수익률이 비정상적입니다: {dividend_yield:.2f}%")
        
        # 배당률 검증 (양수여야 함)
        if dividend_rate < 0:
            errors.append(f"배당률이 음수입니다: {dividend_rate}")
            
        # 배당성향 검증 (0-100% 범위)
        if payout_ratio < 0 or payout_ratio > 1:
            errors.append(f"배당성향이 비정상적입니다: {payout_ratio*100:.2f}%")
            
        # 배당수익률과 배당률 일관성 검증
        if dividend_yield > 0 and dividend_rate > 0:
            # 대략적인 일관성 검사 (정확한 계산은 주가에 따라 달라짐)
            if dividend_yield > 20:  # 20% 이상은 비정상적으로 높음
                errors.append(f"배당수익률이 비정상적으로 높습니다: {dividend_yield:.2f}%")
        
        return len(errors) == 0, errors

    @staticmethod
    def format_large_number(value: float, currency: str) -> str:
        """
        큰 숫자 포맷팅 (억, 조 단위)
        
        Args:
            value: 포맷팅할 값
            currency: 통화 코드
            
        Returns:
            포맷팅된 문자열
        """
        if currency == 'KRW':
            # 한국: 억, 조 단위
            if abs(value) >= 1e12:  # 조
                return f"{value/1e12:.1f}조"
            elif abs(value) >= 1e8:  # 억
                return f"{value/1e8:.1f}억"
            else:
                return f"{value:,.0f}"
        else:
            # 기타 통화: B, T 단위
            if abs(value) >= 1e12:  # T
                return f"{value/1e12:.1f}T"
            elif abs(value) >= 1e9:  # B
                return f"{value/1e9:.1f}B"
            elif abs(value) >= 1e6:  # M
                return f"{value/1e6:.1f}M"
            else:
                return f"{value:,.0f}"

    @staticmethod
    def calculate_financial_metrics(financial_data: Dict, currency: str) -> Dict:
        """
        재무구조 지표 계산
        
        Args:
            financial_data: 재무 데이터 딕셔너리
            currency: 통화 코드
            
        Returns:
            계산된 재무 지표 딕셔너리
        """
        total_revenue = financial_data.get('total_revenue', 0)
        gross_profit = financial_data.get('gross_profit', 0)
        operating_income = financial_data.get('operating_income', 0)
        net_income = financial_data.get('net_income', 0)
        ebitda = financial_data.get('ebitda', 0)
        
        # 수익성 지표 계산
        gross_margin = (gross_profit / total_revenue * 100) if total_revenue > 0 else 0
        operating_margin = (operating_income / total_revenue * 100) if total_revenue > 0 else 0
        net_margin = (net_income / total_revenue * 100) if total_revenue > 0 else 0
        ebitda_margin = (ebitda / total_revenue * 100) if total_revenue > 0 else 0
        
        # 포맷팅된 값들
        formatted_revenue = DataProcessor.format_large_number(total_revenue, currency)
        formatted_gross_profit = DataProcessor.format_large_number(gross_profit, currency)
        formatted_operating_income = DataProcessor.format_large_number(operating_income, currency)
        formatted_net_income = DataProcessor.format_large_number(net_income, currency)
        formatted_ebitda = DataProcessor.format_large_number(ebitda, currency)
        
        return {
            'total_revenue': total_revenue,
            'gross_profit': gross_profit,
            'operating_income': operating_income,
            'net_income': net_income,
            'ebitda': ebitda,
            'formatted_revenue': formatted_revenue,
            'formatted_gross_profit': formatted_gross_profit,
            'formatted_operating_income': formatted_operating_income,
            'formatted_net_income': formatted_net_income,
            'formatted_ebitda': formatted_ebitda,
            'gross_margin': gross_margin,
            'operating_margin': operating_margin,
            'net_margin': net_margin,
            'ebitda_margin': ebitda_margin,
            'revenue_growth': financial_data.get('revenue_growth', 0) * 100,
            'earnings_growth': financial_data.get('earnings_growth', 0) * 100,
            'earnings_quarterly_growth': financial_data.get('earnings_quarterly_growth', 0) * 100
        }

    @staticmethod
    def format_currency(value: float, currency: str) -> str:
        """
        통화별 포맷팅
        
        Args:
            value: 포맷팅할 값
            currency: 통화 코드
            
        Returns:
            포맷팅된 문자열
        """
        if currency == 'USD':
            return f"${value:.2f}"
        elif currency == 'KRW':
            return f"₩{value:,.0f}"
        elif currency == 'JPY':
            return f"¥{value:,.0f}"
        elif currency == 'EUR':
            return f"€{value:.2f}"
        elif currency == 'GBP':
            return f"£{value:.2f}"
        else:
            return f"{value:.2f} {currency}"

    @staticmethod
    def calculate_dividend_metrics(dividend_info: Dict, current_price: float) -> Dict:
        """
        배당 관련 지표 계산
        
        Args:
            dividend_info: 배당 정보 딕셔너리
            current_price: 현재 주가
            
        Returns:
            계산된 배당 지표 딕셔너리
        """
        dividend_yield = dividend_info.get('dividend_yield', 0)
        dividend_rate = dividend_info.get('dividend_rate', 0)
        payout_ratio = dividend_info.get('payout_ratio', 0)
        
        metrics = {
            'dividend_yield_percent': dividend_yield,  # 이미 백분율로 제공됨
            'dividend_rate': dividend_rate,
            'payout_ratio_percent': payout_ratio * 100,
            'annual_dividend_per_share': dividend_rate,
            'dividend_per_share_yield': (dividend_rate / current_price * 100) if current_price > 0 else 0
        }
        
        # 배당 안정성 지표
        if payout_ratio > 0:
            if payout_ratio < 0.3:
                metrics['dividend_sustainability'] = '높음'
            elif payout_ratio < 0.6:
                metrics['dividend_sustainability'] = '보통'
            else:
                metrics['dividend_sustainability'] = '낮음'
        else:
            metrics['dividend_sustainability'] = 'N/A'
            
        return metrics

    @staticmethod
    def validate_data(data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        데이터 품질 검증

        Returns:
            (유효성 여부, 오류 메시지 리스트)
        """
        errors = []

        # 기본 검증
        if data.empty:
            errors.append("데이터가 비어있습니다")
            return False, errors

        # 필수 컬럼 확인
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            errors.append(f"필수 컬럼 누락: {missing_columns}")

        # 가격 논리 검증
        for idx, row in data.iterrows():
            if row['High'] < row['Low']:
                errors.append(f"고가가 저가보다 낮음: {idx}")
            if row['Close'] > row['High'] or row['Close'] < row['Low']:
                errors.append(f"종가가 고저가 범위를 벗어남: {idx}")
            if row['Open'] > row['High'] or row['Open'] < row['Low']:
                errors.append(f"시가가 고저가 범위를 벗어남: {idx}")

        # 결측값 확인
        missing_data = data.isnull().sum()
        if missing_data.any():
            errors.append(f"결측값 발견: {missing_data.to_dict()}")

        # 중복 인덱스 확인
        if data.index.duplicated().any():
            errors.append("중복된 날짜 인덱스 발견")

        return len(errors) == 0, errors

    @staticmethod
    def normalize_data(data: pd.DataFrame, method: str = 'minmax') -> pd.DataFrame:
        """
        데이터 정규화

        Args:
            data: 정규화할 데이터
            method: 정규화 방법 ('minmax', 'zscore', 'robust')
        """
        normalized_data = data.copy()

        if method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            normalized_data[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(
                data[['Open', 'High', 'Low', 'Close']]
            )
        elif method == 'zscore':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            normalized_data[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(
                data[['Open', 'High', 'Low', 'Close']]
            )
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            normalized_data[['Open', 'High', 'Low', 'Close']] = scaler.fit_transform(
                data[['Open', 'High', 'Low', 'Close']]
            )

        return normalized_data

    @staticmethod
    def create_features(data: pd.DataFrame) -> pd.DataFrame:
        """파생 변수 생성"""
        featured_data = data.copy()

        # 가격 기반 피처
        featured_data['HL_Ratio'] = (data['High'] - data['Low']) / data['Close']
        featured_data['OC_Ratio'] = (data['Open'] - data['Close']) / data['Close']

        # 변동성 피처
        featured_data['Price_Range'] = data['High'] - data['Low']
        featured_data['True_Range'] = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data['Close'].shift()),
                abs(data['Low'] - data['Close'].shift())
            )
        )

        # 거래량 피처
        featured_data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        featured_data['Volume_Ratio'] = data['Volume'] / featured_data['Volume_MA']

        # 모멘텀 피처
        for period in [5, 10, 20]:
            featured_data[f'Return_{period}d'] = data['Close'].pct_change(period)
            featured_data[f'Volatility_{period}d'] = (
                data['Returns'].rolling(window=period).std() * np.sqrt(252)
            )

        return featured_data.dropna()

class MultiMarketCollector:
    """다중 시장 데이터 수집기"""

    def __init__(self, market_configs: Dict[str, MarketConfig]):
        self.collectors = {
            name: DataCollector(config)
            for name, config in market_configs.items()
        }

    async def fetch_multiple_markets(self,
                                   start_date: datetime,
                                   end_date: datetime) -> Dict[str, pd.DataFrame]:
        """여러 시장 데이터 동시 수집"""
        tasks = []

        for market_name, collector in self.collectors.items():
            task = asyncio.create_task(
                self._fetch_single_market(collector, start_date, end_date, market_name)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 정리
        market_data = {}
        for i, (market_name, result) in enumerate(zip(self.collectors.keys(), results)):
            if isinstance(result, Exception):
                logging.error(f"Error fetching {market_name}: {result}")
            else:
                market_data[market_name] = result

        return market_data

    async def _fetch_single_market(self,
                                 collector: DataCollector,
                                 start_date: datetime,
                                 end_date: datetime,
                                 market_name: str) -> pd.DataFrame:
        """단일 시장 데이터 수집 (비동기)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            collector.fetch_historical_data,
            start_date,
            end_date
        )

class DataCache:
    """데이터 캐싱 시스템 (개선된 버전)"""

    def __init__(
        self, 
        cache_dir: str = "./cache",
        max_cache_size_mb: int = 500,
        default_ttl_hours: int = 24
    ):
        """
        Args:
            cache_dir: 캐시 디렉토리 경로
            max_cache_size_mb: 최대 캐시 크기 (MB)
            default_ttl_hours: 기본 캐시 유효 시간 (시간)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.max_cache_size_mb = max_cache_size_mb
        self.default_ttl_hours = default_ttl_hours
        self.logger = get_logger(__name__)

    def get_cache_key(self, ticker: str, start_date: datetime, end_date: datetime) -> str:
        """캐시 키 생성"""
        return f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

    def save_to_cache(self, key: str, data: pd.DataFrame):
        """캐시에 데이터 저장"""
        try:
            # 캐시 크기 확인 및 정리
            self.cleanup_old_cache()
            
            cache_file = self.cache_dir / f"{key}.parquet"
            data.to_parquet(cache_file, compression='snappy')
            self.logger.debug(f"Saved to cache: {key}")
        except Exception as e:
            self.logger.error(f"Failed to save cache {key}: {e}")

    def load_from_cache(self, key: str, ttl_hours: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        캐시에서 데이터 로드
        
        Args:
            key: 캐시 키
            ttl_hours: 캐시 유효 시간 (None이면 기본값 사용)
        
        Returns:
            캐시된 데이터 또는 None
        """
        cache_file = self.cache_dir / f"{key}.parquet"
        
        if not cache_file.exists():
            return None
        
        try:
            # 캐시 파일 유효성 확인
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            time_diff = datetime.now() - file_time
            ttl = ttl_hours if ttl_hours is not None else self.default_ttl_hours
            
            if time_diff.total_seconds() < ttl * 3600:
                data = pd.read_parquet(cache_file)
                self.logger.debug(f"Loaded from cache: {key}")
                return data
            else:
                # 만료된 캐시 파일 삭제
                self.logger.debug(f"Cache expired: {key}")
                cache_file.unlink()
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to load cache file {cache_file}: {e}")
            # 손상된 캐시 파일 삭제
            try:
                cache_file.unlink()
            except:
                pass
            return None

    def cleanup_old_cache(self):
        """오래된 캐시 파일 정리"""
        try:
            total_size = 0
            cache_files = []
            
            # 모든 캐시 파일 정보 수집
            for cache_file in self.cache_dir.glob("*.parquet"):
                try:
                    stat = cache_file.stat()
                    size = stat.st_size
                    mtime = stat.st_mtime
                    cache_files.append((cache_file, size, mtime))
                    total_size += size
                except Exception as e:
                    self.logger.warning(f"Error reading cache file {cache_file}: {e}")
            
            # 크기 제한 초과 시 오래된 파일부터 삭제
            max_size_bytes = self.max_cache_size_mb * 1024 * 1024
            
            if total_size > max_size_bytes:
                # 수정 시간순 정렬 (오래된 것부터)
                cache_files.sort(key=lambda x: x[2])
                
                deleted_count = 0
                for cache_file, size, _ in cache_files:
                    if total_size <= max_size_bytes:
                        break
                    
                    try:
                        cache_file.unlink()
                        total_size -= size
                        deleted_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to delete cache file {cache_file}: {e}")
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} old cache files")
        
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {e}")

    def clear_all_cache(self):
        """모든 캐시 삭제"""
        try:
            deleted_count = 0
            for cache_file in self.cache_dir.glob("*.parquet"):
                try:
                    cache_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete {cache_file}: {e}")
            
            self.logger.info(f"Cleared {deleted_count} cache files")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    def get_cache_stats(self) -> Dict[str, any]:
        """캐시 통계 정보 반환"""
        try:
            total_size = 0
            file_count = 0
            oldest_file_time = None
            newest_file_time = None
            
            for cache_file in self.cache_dir.glob("*.parquet"):
                try:
                    stat = cache_file.stat()
                    total_size += stat.st_size
                    file_count += 1
                    
                    mtime = datetime.fromtimestamp(stat.st_mtime)
                    if oldest_file_time is None or mtime < oldest_file_time:
                        oldest_file_time = mtime
                    if newest_file_time is None or mtime > newest_file_time:
                        newest_file_time = mtime
                except:
                    pass
            
            return {
                'total_size_mb': total_size / (1024 * 1024),
                'file_count': file_count,
                'oldest_file': oldest_file_time,
                'newest_file': newest_file_time,
                'max_size_mb': self.max_cache_size_mb,
                'usage_percent': (total_size / (self.max_cache_size_mb * 1024 * 1024)) * 100
            }
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}
