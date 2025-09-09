# Algorithmic Trading Engine

A high-performance algorithmic trading engine built in Python with C++ extensions for optimal performance. This engine supports multiple trading strategies, real-time market data processing, comprehensive risk management, and live trading integration.

## 🚀 Features

- **Multi-strategy trading engine** with support for cointegration, mean reversion, and momentum strategies
- **Real-time market data processing** with low latency and high throughput
- **Advanced risk management system** with position sizing, stop-loss, and drawdown controls
- **Comprehensive backtesting framework** for strategy validation and optimization
- **Live trading integration** with Interactive Brokers (IBKR)
- **Performance analytics** and detailed reporting
- **C++ extensions** for performance-critical operations
- **Bayesian optimization** for hyperparameter tuning
- **Cointegration-based strategies** for pairs trading

## 📁 Project Structure

```
algorithmic_trading_engine/
├── src/
│   ├── cpp/                    # C++ extensions for performance-critical operations
│   │   └── fast_operations.cpp
│   ├── python/                 # Python trading engine implementation
│   │   ├── trading_engine.py   # Main trading engine
│   │   ├── cointegration_strategy.py  # Cointegration trading strategy
│   │   ├── bayesian_optimizer.py      # Bayesian optimization
│   │   ├── data_handler.py     # Market data handling
│   │   └── ibkr_client.py      # Interactive Brokers client
│   └── utils/                  # Utility functions and helpers
│       └── data_loader.py
├── config/                     # Configuration files
│   └── config.yaml            # Main configuration
├── data/                      # Market data storage
├── logs/                      # Trading logs and performance data
├── tests/                     # Test suite
├── examples/                  # Usage examples
├── docs/                      # Documentation
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── Makefile                   # Build automation
└── README.md                  # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- C++ compiler (for C++ extensions)
- Git

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mitashshah/algorithmic_trading_engine.git
   cd algorithmic_trading_engine
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   make install
   # or manually:
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Build C++ extensions (optional but recommended for performance):**
   ```bash
   make build-cpp
   ```

5. **Create necessary directories:**
   ```bash
   make setup-dirs
   ```

### Development Setup

For development work, install additional development dependencies:

```bash
make install-dev
make dev-setup
```

## 🚀 Usage

### Basic Usage

The trading engine supports multiple modes:

```bash
# Paper trading mode (default - no real money)
python main.py --paper

# Live trading mode (requires IBKR connection)
python main.py --live

# Optimization mode (hyperparameter tuning)
python main.py --optimize

# Custom configuration
python main.py --config custom_config.yaml
```

### Using Make Commands

```bash
# Show all available commands
make help

# Run in paper trading mode
make run-paper

# Run in live trading mode (WARNING: uses real money!)
make run-live

# Run optimization
make run-opt

# Run tests
make test

# Format code
make format

# Lint code
make lint
```

### Configuration

Edit `config/config.yaml` to customize:

- **Trading Parameters:** Initial capital, position sizing, leverage
- **Strategy Parameters:** Cointegration settings, risk thresholds
- **Risk Management:** Drawdown limits, stop-loss, take-profit
- **Data Sources:** Symbols, update frequency, historical data
- **Performance Metrics:** Benchmark, risk-free rate, targets

Example configuration:
```yaml
trading:
  initial_capital: 1000000  # $1M initial capital
  max_position_size: 0.1    # Max 10% in single position
  max_leverage: 2.0         # Max 2x leverage

cointegration:
  lookback_period: 252      # 1 year of trading days
  z_score_threshold: 2.0    # Entry threshold
  exit_threshold: 0.5       # Exit threshold

risk:
  max_drawdown: 0.15        # Max 15% drawdown
  stop_loss: 0.05           # 5% stop loss
  take_profit: 0.15         # 15% take profit
```

### Interactive Brokers Setup

For live trading, you'll need to set up Interactive Brokers:

1. **Install TWS or IB Gateway**
2. **Configure API settings:**
   - Enable API connections
   - Set port (default: 7497 for paper, 7496 for live)
   - Set client ID (default: 1)

3. **Update configuration:**
   ```yaml
   ibkr:
     host: "127.0.0.1"
     port: 7497
     client_id: 1
   ```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
make test

# Run specific test file
python -m pytest tests/test_heston_model.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## 📊 Examples

### Basic Trading Example

```python
from src.python.trading_engine import TradingEngine

# Initialize engine
engine = TradingEngine("config/config.yaml")

# Start in paper trading mode
engine.start(trading_enabled=False, optimization_mode=False)

# Get performance summary
performance = engine.get_performance_summary()
print(f"Current Capital: ${performance['current_capital']:,.2f}")
print(f"Total Return: {performance['total_return']:.2%}")
```

### Custom Strategy Example

```python
from src.python.cointegration_strategy import CointegrationStrategy

# Create custom strategy
strategy = CointegrationStrategy(
    symbols=["AAPL", "MSFT"],
    lookback_period=252,
    z_score_threshold=2.0
)

# Run backtest
results = strategy.backtest(start_date="2023-01-01", end_date="2023-12-31")
print(f"Strategy Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

## 🔧 Development

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Run all checks
make check
```

### Building C++ Extensions

```bash
# Build C++ extensions
make build-cpp

# Clean build artifacts
make clean
```

### Creating Distribution Package

```bash
# Create package
make package

# Install from package
pip install dist/algorithmic_trading_engine-1.0.0.tar.gz
```

## 📈 Performance

The engine is optimized for high-performance trading:

- **C++ extensions** for computationally intensive operations
- **Parallel processing** for data analysis and optimization
- **Efficient data structures** for real-time market data
- **Memory optimization** for large datasets
- **Low-latency** order execution

## ⚠️ Risk Disclaimer

**This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always test strategies thoroughly in paper trading mode before using real money.**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues:** [GitHub Issues](https://github.com/mitashshah/algorithmic_trading_engine/issues)
- **Documentation:** [Read the Docs](https://algorithmic-trading-engine.readthedocs.io/)
- **Email:** mitash.shah@example.com

## 🙏 Acknowledgments

- Interactive Brokers for market data and trading API
- QuantLib for financial mathematics
- NumPy, Pandas, and SciPy for numerical computing
- The open-source quantitative finance community

---

**Happy Trading! 📈**
