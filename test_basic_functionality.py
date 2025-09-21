#!/usr/bin/env python3
"""
Fair Value Analyzer 기본 기능 테스트 스크립트
"""

import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """필수 모듈 import 테스트"""
    print("=" * 50)
    print("📦 Import 테스트")
    print("=" * 50)

    try:
        import streamlit
        print("✅ Streamlit")

        from config.settings import config_manager
        print("✅ Config Manager")

        from data.collectors import DataCollector
        print("✅ Data Collector")

        from analysis.technical import TechnicalAnalyzer
        print("✅ Technical Analyzer")

        from analysis.monte_carlo import MonteCarloSimulator
        print("✅ Monte Carlo Simulator")

        from analysis.workflow import UnifiedFairValueWorkflow
        print("✅ Unified Workflow")

        from visualization.dashboard import FairValueDashboard
        print("✅ Dashboard")

        return True

    except Exception as e:
        print(f"❌ Import 오류: {e}")
        return False

def test_configuration():
    """설정 시스템 테스트"""
    print("\n" + "=" * 50)
    print("⚙️ Configuration 테스트")
    print("=" * 50)

    try:
        from config.settings import config_manager

        # 시장 설정 확인
        markets = list(config_manager.markets.keys())
        print(f"✅ 지원 시장: {markets}")

        # KOSPI 설정 확인
        kospi_config = config_manager.get_market_config('kospi')
        print(f"✅ KOSPI 설정: {kospi_config.name} ({kospi_config.ticker})")

        # 기술적 분석 설정 확인
        tech_config = config_manager.technical
        print(f"✅ 기술적 분석 설정: RSI({tech_config.rsi_period}), MA({tech_config.ma_periods})")

        # 몬테카를로 설정 확인
        mc_config = config_manager.monte_carlo
        print(f"✅ 몬테카를로 설정: {mc_config.simulations}회, {mc_config.forecast_days}일")

        return True

    except Exception as e:
        print(f"❌ Configuration 오류: {e}")
        return False

def test_data_collection():
    """데이터 수집 테스트"""
    print("\n" + "=" * 50)
    print("📊 Data Collection 테스트")
    print("=" * 50)

    try:
        from data.collectors import DataCollector
        from config.settings import config_manager
        from datetime import datetime, timedelta

        market_config = config_manager.get_market_config('kospi')
        collector = DataCollector(market_config)

        # 최근 3개월 데이터 수집
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        print(f"📅 데이터 수집 기간: {start_date.date()} ~ {end_date.date()}")

        data = collector.fetch_historical_data(start_date, end_date)
        print(f"✅ 데이터 수집 성공: {len(data)}행")
        print(f"✅ 컬럼: {list(data.columns)}")

        if len(data) > 0:
            latest = data.iloc[-1]
            print(f"✅ 최신 종가: {latest['Close']:,.0f}")

        return True, data

    except Exception as e:
        print(f"❌ Data Collection 오류: {e}")
        return False, None

def test_technical_analysis(data):
    """기술적 분석 테스트"""
    print("\n" + "=" * 50)
    print("📈 Technical Analysis 테스트")
    print("=" * 50)

    try:
        if data is None or len(data) < 20:
            print("⚠️ 기술적 분석에 충분한 데이터가 없습니다")
            return False

        from analysis.technical import TechnicalAnalyzer
        from config.settings import config_manager

        tech_config = config_manager.technical
        analyzer = TechnicalAnalyzer(tech_config)

        # 기술적 지표 계산
        enriched_data = analyzer.calculate_all_indicators(data)
        print(f"✅ 기술적 지표 계산: {len(enriched_data.columns)}개 컬럼")

        # 매매 신호 생성
        signals = analyzer.generate_signals(enriched_data)
        print(f"✅ 매매 신호 생성: {len(signals.columns)}개 신호")

        # 최신 분석 결과
        latest_analysis = analyzer.get_latest_analysis(enriched_data)
        if latest_analysis:
            rsi = latest_analysis.get('technical_indicators', {}).get('rsi', 0)
            composite_signal = latest_analysis.get('signals', {}).get('composite_signal', 0)
            print(f"✅ 최신 RSI: {rsi:.1f}")
            print(f"✅ 종합 신호: {composite_signal}")

        return True

    except Exception as e:
        print(f"❌ Technical Analysis 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_monte_carlo(data):
    """몬테카를로 시뮬레이션 테스트"""
    print("\n" + "=" * 50)
    print("🎲 Monte Carlo Simulation 테스트")
    print("=" * 50)

    try:
        if data is None or len(data) < 30:
            print("⚠️ 몬테카를로 시뮬레이션에 충분한 데이터가 없습니다")
            return False

        from analysis.monte_carlo import MonteCarloSimulator
        from config.settings import config_manager

        # 테스트용 축소 설정
        mc_config = config_manager.monte_carlo
        mc_config.simulations = 100  # 테스트용으로 축소
        mc_config.forecast_days = 30

        simulator = MonteCarloSimulator(mc_config)

        print(f"🎯 시뮬레이션 설정: {mc_config.simulations}회, {mc_config.forecast_days}일")

        result = simulator.run_simulation(data)

        print(f"✅ 시뮬레이션 성공: {result.paths.shape}")
        print(f"✅ 예상 수익률: {result.statistics['mean_return']*100:.2f}%")
        print(f"✅ 상승 확률: {result.statistics['positive_return_prob']*100:.1f}%")

        # 신뢰구간
        ci_95 = result.confidence_intervals.get('95%', {})
        if ci_95:
            print(f"✅ 95% 신뢰구간: {ci_95['lower']:,.0f} ~ {ci_95['upper']:,.0f}")

        return True

    except Exception as e:
        print(f"❌ Monte Carlo Simulation 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard():
    """대시보드 테스트"""
    print("\n" + "=" * 50)
    print("📊 Dashboard 테스트")
    print("=" * 50)

    try:
        from visualization.dashboard import FairValueDashboard

        dashboard = FairValueDashboard()
        print("✅ 대시보드 객체 생성 성공")

        # 색상 팔레트 확인
        colors = dashboard.colors
        print(f"✅ 색상 팔레트: {len(colors)}개")

        return True

    except Exception as e:
        print(f"❌ Dashboard 오류: {e}")
        return False

def test_workflow():
    """통합 워크플로우 테스트"""
    print("\n" + "=" * 50)
    print("🔄 Unified Workflow 테스트")
    print("=" * 50)

    try:
        from analysis.workflow import UnifiedFairValueWorkflow

        workflow = UnifiedFairValueWorkflow('kospi')
        print("✅ 워크플로우 객체 생성 성공")

        # 분석 단계 확인
        steps = workflow.analysis_steps
        print(f"✅ 분석 단계: {len(steps)}개")
        for i, step in enumerate(steps, 1):
            print(f"   {i}. {step.name}")

        return True

    except Exception as e:
        print(f"❌ Workflow 오류: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🚀 Fair Value Analyzer 기능 테스트 시작")

    results = {}

    # 1. Import 테스트
    results['imports'] = test_imports()

    # 2. Configuration 테스트
    results['config'] = test_configuration()

    # 3. Data Collection 테스트
    data_result, data = test_data_collection()
    results['data_collection'] = data_result

    # 4. Technical Analysis 테스트
    results['technical_analysis'] = test_technical_analysis(data)

    # 5. Monte Carlo 테스트
    results['monte_carlo'] = test_monte_carlo(data)

    # 6. Dashboard 테스트
    results['dashboard'] = test_dashboard()

    # 7. Workflow 테스트
    results['workflow'] = test_workflow()

    # 결과 요약
    print("\n" + "=" * 50)
    print("📋 테스트 결과 요약")
    print("=" * 50)

    total_tests = len(results)
    passed_tests = sum(results.values())

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")

    print(f"\n총 테스트: {total_tests}개")
    print(f"성공: {passed_tests}개")
    print(f"실패: {total_tests - passed_tests}개")
    print(f"성공률: {passed_tests/total_tests*100:.1f}%")

    if passed_tests == total_tests:
        print("\n🎉 모든 테스트 통과! 애플리케이션이 정상적으로 작동합니다.")
        print("\n실행 방법:")
        print("streamlit run streamlit_app.py")
    else:
        print(f"\n⚠️ {total_tests - passed_tests}개 테스트 실패. 문제를 해결해주세요.")

if __name__ == "__main__":
    main()