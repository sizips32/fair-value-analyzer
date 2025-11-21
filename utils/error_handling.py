"""
공통 에러 처리 유틸리티
재사용 가능한 에러 핸들러 및 데코레이터 제공
"""

import functools
import logging
from typing import Callable, Optional, TypeVar, Any
from functools import wraps

T = TypeVar('T')

logger = logging.getLogger(__name__)


def handle_data_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    데이터 관련 에러를 처리하는 데코레이터
    
    ValueError, KeyError를 처리하고 로깅합니다.
    
    사용 예:
        @handle_data_errors
        def fetch_data():
            # ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Invalid data in {func.__name__}: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing key in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise
    
    return wrapper


def handle_network_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    네트워크 관련 에러를 처리하는 데코레이터
    
    ConnectionError, TimeoutError를 처리하고 로깅합니다.
    
    사용 예:
        @handle_network_errors
        async def fetch_data():
            # ...
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ConnectionError as e:
            logger.error(f"Network error in {func.__name__}: {e}")
            raise
        except TimeoutError as e:
            logger.error(f"Timeout in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ConnectionError as e:
            logger.error(f"Network error in {func.__name__}: {e}")
            raise
        except TimeoutError as e:
            logger.error(f"Timeout in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def handle_analysis_errors(func: Callable[..., T]) -> Callable[..., T]:
    """
    분석 관련 에러를 처리하는 데코레이터
    
    ValueError, KeyError, ConnectionError를 종합적으로 처리합니다.
    
    사용 예:
        @handle_analysis_errors
        async def run_analysis():
            # ...
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (ValueError, KeyError) as e:
            logger.error(f"Configuration/Data error in {func.__name__}: {e}")
            raise
        except ConnectionError as e:
            logger.error(f"Network error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, KeyError) as e:
            logger.error(f"Configuration/Data error in {func.__name__}: {e}")
            raise
        except ConnectionError as e:
            logger.error(f"Network error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            raise
    
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def safe_execute(
    func: Callable[..., T],
    default: Optional[T] = None,
    log_errors: bool = True,
    reraise: bool = False
) -> Optional[T]:
    """
    안전하게 함수를 실행하고 에러를 처리합니다.
    
    Args:
        func: 실행할 함수
        default: 에러 발생 시 반환할 기본값
        log_errors: 에러를 로깅할지 여부
        reraise: 에러를 다시 발생시킬지 여부
    
    Returns:
        함수 실행 결과 또는 default 값
    
    사용 예:
        result = safe_execute(
            lambda: risky_operation(),
            default={},
            log_errors=True
        )
    """
    try:
        return func()
    except (ValueError, KeyError) as e:
        if log_errors:
            logger.warning(f"Data error in safe_execute: {e}")
        if reraise:
            raise
        return default
    except ConnectionError as e:
        if log_errors:
            logger.warning(f"Network error in safe_execute: {e}")
        if reraise:
            raise
        return default
    except Exception as e:
        if log_errors:
            logger.error(f"Unexpected error in safe_execute: {e}", exc_info=True)
        if reraise:
            raise
        return default


class ErrorContext:
    """
    에러 컨텍스트 매니저
    
    사용 예:
        with ErrorContext("데이터 수집"):
            # 작업 수행
    """
    
    def __init__(self, operation_name: str, log_errors: bool = True):
        self.operation_name = operation_name
        self.log_errors = log_errors
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False
        
        if self.log_errors:
            if exc_type in (ValueError, KeyError):
                logger.error(f"{self.operation_name} failed: Invalid data - {exc_val}")
            elif exc_type == ConnectionError:
                logger.error(f"{self.operation_name} failed: Network error - {exc_val}")
            else:
                logger.error(f"{self.operation_name} failed: {exc_val}", exc_info=True)
        
        # 에러를 다시 발생시킴
        return False

