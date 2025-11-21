"""
환경 변수 기반 설정 모듈
.env 파일을 통한 설정 관리

⚠️ DEPRECATED: 이 모듈은 더 이상 사용되지 않습니다.
환경 변수 설정은 config/settings.py의 ConfigManager에 통합되었습니다.
향후 버전에서 제거될 예정입니다.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    # .env 파일 로드 (프로젝트 루트에서)
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class EnvConfig:
    """환경 변수 기반 설정"""
    
    # Streamlit 설정
    STREAMLIT_PORT = int(os.getenv('STREAMLIT_SERVER_PORT', '8543'))
    STREAMLIT_HEADLESS = os.getenv('STREAMLIT_SERVER_HEADLESS', 'true').lower() == 'true'
    
    # 캐시 설정
    CACHE_DIR = Path(os.getenv('CACHE_DIR', './cache'))
    CACHE_MAX_SIZE_MB = int(os.getenv('CACHE_MAX_SIZE_MB', '500'))
    CACHE_TTL_HOURS = int(os.getenv('CACHE_TTL_HOURS', '24'))
    
    # 로깅 설정
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
    LOG_DIR = Path(os.getenv('LOG_DIR', './logs'))
    LOG_TO_FILE = os.getenv('LOG_TO_FILE', 'true').lower() == 'true'
    LOG_TO_CONSOLE = os.getenv('LOG_TO_CONSOLE', 'true').lower() == 'true'
    
    # MCP 서버 설정
    MCP_SERVER_PORT = int(os.getenv('MCP_SERVER_PORT', '8000'))
    
    # 데이터 수집 설정
    DATA_FETCH_TIMEOUT = int(os.getenv('DATA_FETCH_TIMEOUT', '30'))  # 초
    DATA_FETCH_RETRIES = int(os.getenv('DATA_FETCH_RETRIES', '3'))
    
    # 분석 설정
    DEFAULT_ANALYSIS_PERIOD_YEARS = int(os.getenv('DEFAULT_ANALYSIS_PERIOD_YEARS', '2'))
    MONTE_CARLO_SIMULATIONS = int(os.getenv('MONTE_CARLO_SIMULATIONS', '10000'))
    MONTE_CARLO_DAYS = int(os.getenv('MONTE_CARLO_DAYS', '252'))
    
    # API 설정 (필요시)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    
    @classmethod
    def validate(cls) -> tuple[bool, list[str]]:
        """
        설정 검증
        
        Returns:
            (유효성 여부, 오류 메시지 리스트)
        """
        errors = []
        
        # 포트 범위 검증
        if not (1024 <= cls.STREAMLIT_PORT <= 65535):
            errors.append(f"Invalid STREAMLIT_PORT: {cls.STREAMLIT_PORT} (must be 1024-65535)")
        
        if not (1024 <= cls.MCP_SERVER_PORT <= 65535):
            errors.append(f"Invalid MCP_SERVER_PORT: {cls.MCP_SERVER_PORT} (must be 1024-65535)")
        
        # 캐시 크기 검증
        if cls.CACHE_MAX_SIZE_MB < 0:
            errors.append(f"Invalid CACHE_MAX_SIZE_MB: {cls.CACHE_MAX_SIZE_MB} (must be >= 0)")
        
        # 로그 레벨 검증
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if cls.LOG_LEVEL not in valid_log_levels:
            errors.append(f"Invalid LOG_LEVEL: {cls.LOG_LEVEL} (must be one of {valid_log_levels})")
        
        # 시뮬레이션 파라미터 검증
        if cls.MONTE_CARLO_SIMULATIONS < 100:
            errors.append(f"MONTE_CARLO_SIMULATIONS too low: {cls.MONTE_CARLO_SIMULATIONS} (minimum 100)")
        
        if cls.MONTE_CARLO_DAYS < 1:
            errors.append(f"Invalid MONTE_CARLO_DAYS: {cls.MONTE_CARLO_DAYS} (must be >= 1)")
        
        return len(errors) == 0, errors
    
    @classmethod
    def print_config(cls):
        """현재 설정 출력 (디버깅용)"""
        print("=" * 50)
        print("Environment Configuration")
        print("=" * 50)
        print(f"Streamlit Port: {cls.STREAMLIT_PORT}")
        print(f"Streamlit Headless: {cls.STREAMLIT_HEADLESS}")
        print(f"Cache Directory: {cls.CACHE_DIR}")
        print(f"Cache Max Size: {cls.CACHE_MAX_SIZE_MB} MB")
        print(f"Cache TTL: {cls.CACHE_TTL_HOURS} hours")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print(f"Log Directory: {cls.LOG_DIR}")
        print(f"Log to File: {cls.LOG_TO_FILE}")
        print(f"Log to Console: {cls.LOG_TO_CONSOLE}")
        print(f"MCP Server Port: {cls.MCP_SERVER_PORT}")
        print(f"Monte Carlo Simulations: {cls.MONTE_CARLO_SIMULATIONS}")
        print(f"Monte Carlo Days: {cls.MONTE_CARLO_DAYS}")
        print(f"Dotenv Available: {DOTENV_AVAILABLE}")
        print("=" * 50)


# 싱글톤 인스턴스
env_config = EnvConfig()

# 설정 검증 (모듈 로드 시)
is_valid, validation_errors = env_config.validate()
if not is_valid:
    import warnings
    for error in validation_errors:
        warnings.warn(f"Configuration error: {error}")
