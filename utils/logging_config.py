"""
통합 로깅 설정 모듈
애플리케이션 전체에서 일관된 로깅을 제공
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    통합 로깅 설정
    
    Args:
        log_level: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: 로그 파일 저장 디렉토리
        log_to_file: 파일 로깅 활성화 여부
        log_to_console: 콘솔 로깅 활성화 여부
    
    Returns:
        설정된 루트 로거
    """
    # 로그 포맷
    detailed_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    simple_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 루트 로거 설정
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 기존 핸들러 제거 (중복 방지)
    logger.handlers.clear()
    
    # 콘솔 핸들러
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter(simple_format, date_format))
        logger.addHandler(console_handler)
    
    # 파일 핸들러 (로테이션)
    if log_to_file:
        if log_dir is None:
            log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True, parents=True)
        
        # 일반 로그 파일 (INFO 이상)
        info_handler = logging.handlers.RotatingFileHandler(
            log_dir / "app.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(logging.Formatter(detailed_format, date_format))
        logger.addHandler(info_handler)
        
        # 에러 로그 파일 (ERROR 이상만)
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "error.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(logging.Formatter(detailed_format, date_format))
        logger.addHandler(error_handler)
        
        # 디버그 로그 파일 (DEBUG 레벨일 때만)
        if log_level.upper() == "DEBUG":
            debug_handler = logging.handlers.RotatingFileHandler(
                log_dir / "debug.log",
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=3,
                encoding='utf-8'
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(logging.Formatter(detailed_format, date_format))
            logger.addHandler(debug_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    모듈별 로거 가져오기
    
    Args:
        name: 로거 이름 (일반적으로 __name__ 사용)
    
    Returns:
        로거 인스턴스
    """
    return logging.getLogger(name)


class LoggerMixin:
    """
    클래스에 로거를 추가하기 위한 Mixin
    
    사용 예:
        class MyClass(LoggerMixin):
            def __init__(self):
                super().__init__()
                self.logger.info("MyClass initialized")
    """
    
    @property
    def logger(self) -> logging.Logger:
        """클래스 이름을 기반으로 로거 반환"""
        name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return logging.getLogger(name)


def log_execution_time(func):
    """
    함수 실행 시간을 로깅하는 데코레이터
    
    사용 예:
        @log_execution_time
        def my_function():
            # ...
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    
    return wrapper


def log_async_execution_time(func):
    """
    비동기 함수 실행 시간을 로깅하는 데코레이터
    
    사용 예:
        @log_async_execution_time
        async def my_async_function():
            # ...
    """
    import functools
    import time
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    
    return wrapper
