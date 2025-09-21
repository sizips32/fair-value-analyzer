# Fair Value Analyzer

## 🚀 개요

Fair Value Analyzer는 기존의 복잡한 Jupyter 노트북 워크플로우를 직관적이고 효율적인 웹 애플리케이션으로 변환한 통합 공정가치 분석 도구입니다.

### 주요 특징

- **📊 원클릭 분석**: 복잡한 분석 과정을 단일 버튼으로 실행
- **🎯 실시간 대시보드**: 모든 지표를 한 화면에 통합 표시
- **🎲 고도화된 몬테카를로**: 다중 확률 모델 지원
- **🤖 AI 인사이트**: 자동 분석 결과 해석
- **⚙️ 설정 기반**: YAML 파일을 통한 유연한 설정 관리

## 🏗️ 프로젝트 구조

```
fair_value_analyzer/
├── analysis/                   # 분석 엔진
│   ├── technical.py           # 기술적 분석
│   ├── monte_carlo.py         # 몬테카를로 시뮬레이션
│   └── workflow.py            # 통합 워크플로우
├── data/                      # 데이터 관리
│   └── collectors.py          # 데이터 수집 및 전처리
├── visualization/             # 시각화
│   └── dashboard.py           # 대시보드 차트
├── config/                    # 설정 관리
│   ├── settings.py            # 설정 클래스
│   └── config.yaml            # 기본 설정
├── utils/                     # 유틸리티
├── streamlit_app.py           # 메인 애플리케이션
└── requirements.txt           # 의존성 패키지
```

## 🚀 설치 및 실행

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 애플리케이션 실행

```bash
cd fair_value_analyzer
streamlit run streamlit_app.py
```

### 3. 브라우저 접속

```
http://localhost:8501
```

## 📋 사용법

### 기본 워크플로우

1. **시장 선택**: 사이드바에서 분석할 시장 선택 (KOSPI, S&P500, NASDAQ 등)
2. **기간 설정**: 분석 기간 설정 (1~5년 또는 사용자 정의)
3. **옵션 설정**: 몬테카를로 시뮬레이션 횟수, 예측 기간 등
4. **분석 실행**: "🚀 분석 실행" 버튼 클릭
5. **결과 확인**: 4개 탭에서 상세 결과 확인

### 고급 설정

#### 몬테카를로 시뮬레이션
- **시뮬레이션 횟수**: 1,000 ~ 50,000 (기본값: 10,000)
- **예측 기간**: 30 ~ 252일 (기본값: 126일, 약 6개월)
- **신뢰구간**: 90%, 95%, 99%
- **시뮬레이션 방법**:
  - Geometric Brownian Motion
  - Jump Diffusion
  - Heston Stochastic Volatility

#### 기술적 분석
- RSI 기간: 14일 (기본값)
- 이동평균: 20일, 50일, 200일
- 볼린저 밴드: 20일, 2 표준편차
- MACD: 12일, 26일, 9일

## 📊 분석 결과 해석

### 종합 분석 탭
- **가격 차트**: 캔들스틱 + 기술적 지표
- **AI 인사이트**: 자동 생성된 분석 해석
- **신뢰구간**: 예측 가격 범위

### 몬테카를로 탭
- **시뮬레이션 경로**: 가능한 가격 움직임
- **가격 분포**: 최종 가격의 확률 분포
- **통계 요약**: 주요 통계 지표

### 기술적 분석 탭
- **종합 차트**: RSI, MACD, 볼린저 밴드 등
- **매매 신호**: 5개 지표 기반 종합 신호

### 리스크 분석 탭
- **기본 지표**: 변동성, 샤프비율, 최대손실률
- **VaR**: 95%, 99% 신뢰구간 손실 위험
- **분포 차트**: 리스크-수익률 관계

## ⚙️ 설정 커스터마이징

### config.yaml 수정

```yaml
# 새로운 시장 추가
markets:
  custom_market:
    ticker: "^FTSE"
    name: "FTSE 100"
    timezone: "Europe/London"
    trading_hours: "08:00-16:30"
    currency: "GBP"

# 기술적 분석 파라미터 조정
technical:
  rsi_period: 21
  ma_periods: [10, 30, 60]
  bollinger_period: 25
```

### 프로그래밍 방식 설정

```python
from config.settings import config_manager

# 몬테카를로 설정 변경
config_manager.update_config('monte_carlo',
    simulations=20000,
    forecast_days=180,
    method='heston_stochastic'
)
```

## 🔧 확장 가능성

### 새로운 분석 모듈 추가

```python
# analysis/custom_analysis.py
class CustomAnalyzer:
    def __init__(self, config):
        self.config = config

    def analyze(self, data):
        # 사용자 정의 분석 로직
        pass
```

### 새로운 시각화 추가

```python
# visualization/custom_charts.py
def create_custom_chart(data):
    # 사용자 정의 차트
    return fig
```

## 🐛 트러블슈팅

### 일반적인 문제

1. **데이터 수집 오류**
   - 인터넷 연결 확인
   - Yahoo Finance API 상태 확인
   - 티커 심볼 정확성 확인

2. **느린 성능**
   - 몬테카를로 시뮬레이션 횟수 감소
   - 분석 기간 단축
   - 메모리 사용량 확인

3. **차트 표시 오류**
   - 브라우저 캐시 클리어
   - Plotly 라이브러리 버전 확인

### 로그 확인

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 📈 성능 최적화

### 캐싱 활용
- 데이터 수집: 1시간 캐시
- 분석 결과: 세션 단위 캐시
- 설정: 파일 기반 영구 저장

### 병렬 처리
- 몬테카를로 시뮬레이션: 멀티프로세싱
- 데이터 수집: 비동기 처리
- 차트 생성: 지연 로딩

## 🤝 기여 방법

1. Fork 및 Clone
2. 새로운 기능 브랜치 생성
3. 코드 작성 및 테스트
4. Pull Request 생성

## 📄 라이선스

MIT License

## 📞 지원

- 이슈 트래커: GitHub Issues
- 문서: 프로젝트 Wiki
- 이메일: 프로젝트 관리자

---

**⚠️ 주의사항**: 본 도구는 교육 및 연구 목적으로 제작되었습니다. 실제 투자 결정은 신중히 하시기 바랍니다.# fair-value-analyzer
