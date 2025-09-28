# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running the Application
```bash
# Navigate to the project directory
cd fair_value_analyzer

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run streamlit_app.py
```

### Running Tests
```bash
# Run basic functionality tests
python test_basic_functionality.py
```

### Development Tools
```bash
# Code formatting (if black is installed)
black .

# Linting (if flake8 is installed)
flake8 .
```

## Architecture Overview

### Core Workflow Pattern
The application follows a unified workflow pattern where all analysis operations are orchestrated through `UnifiedFairValueWorkflow` class in `analysis/workflow.py`. This single entry point coordinates:

1. **Data Collection** (`data/collectors.py`): Fetches market data from Yahoo Finance using `DataCollector` and processes it with `DataProcessor`
2. **Technical Analysis** (`analysis/technical.py`): Calculates indicators (RSI, MACD, Bollinger Bands, Moving Averages) via `TechnicalAnalyzer`
3. **Monte Carlo Simulation** (`analysis/monte_carlo.py`): Runs price predictions using `MonteCarloSimulator` with configurable methods (Geometric Brownian Motion, Jump Diffusion, Heston)
4. **Visualization** (`visualization/dashboard.py`): Creates interactive Plotly charts through `FairValueDashboard`

### Configuration Management
The application uses YAML-based configuration (`config/config.yaml`) managed by `ConfigManager` in `config/settings.py`. Configuration includes:
- **Markets**: Pre-configured markets (KOSPI, S&P500, NASDAQ, Nikkei) with tickers, timezones, trading hours
- **Technical Parameters**: RSI period, MA periods, Bollinger bands settings
- **Monte Carlo Settings**: Simulation count, forecast days, confidence levels
- **ML Models**: LSTM, GRU, Prophet configurations (defined but not yet implemented)

### Streamlit Application Structure
The main entry point (`streamlit_app.py`) provides:
- **Sidebar**: Market selection, time period, analysis options
- **Main Area**: 4-tab interface (종합 분석, 몬테카를로, 기술적 분석, 리스크 분석)
- **Session State Management**: Caches analysis results to avoid re-computation
- **Progress Tracking**: Real-time progress bar during analysis execution

### Key Design Patterns

1. **Async Processing**: Uses `asyncio` for parallel data collection and processing
2. **Caching Strategy**: Leverages Streamlit's `@st.cache_data` for performance optimization
3. **Dataclass Results**: Structured results using `@dataclass` for type safety and clarity
4. **Modular Analysis**: Each analysis component (technical, Monte Carlo) is independent and testable

## Important Context

### External Dependencies
- **Yahoo Finance API** (`yfinance`): Primary data source, no authentication required but subject to rate limits
- **TA-Lib**: Technical analysis library requiring binary installation
- **Plotly**: Interactive charting library for all visualizations

### Performance Considerations
- Monte Carlo simulations are CPU-intensive; default is 10,000 simulations
- Large forecast periods (>252 days) significantly increase computation time
- Streamlit reruns entire script on interaction; use session state to preserve results

### Current Limitations
- ML models (LSTM, GRU, Prophet) are configured but not yet implemented
- No real-time data streaming; uses batch data collection
- Limited to equity markets supported by Yahoo Finance