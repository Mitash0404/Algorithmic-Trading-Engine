# Algorithmic Trading Engine

A Python-based algorithmic trading engine with C++ extensions for performance-critical operations.

## Features

- Cointegration-based mean reversion strategies
- Bayesian hyperparameter optimization
- Real-time market data processing
- Interactive Brokers integration
- C++ extensions for low-latency operations
- Risk management and position sizing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mitash0404/Algorithmic-Trading-Engine.git
cd algorithmic_trading_engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Build C++ extensions (optional):
```bash
make build-cpp
```

## Usage

Run the trading engine:
```bash
python main.py --paper  # Paper trading mode
python main.py --live   # Live trading (requires IBKR setup)
```

## Configuration

Edit `config/config.yaml` to customize trading parameters, risk settings, and strategy configurations.

## Project Structure

```
src/
├── python/           # Python trading engine
│   ├── trading_engine.py
│   ├── cointegration_strategy.py
│   ├── bayesian_optimizer.py
│   ├── data_handler.py
│   └── ibkr_client.py
├── cpp/              # C++ performance extensions
│   └── fast_operations.cpp
└── utils/            # Utility functions
    └── data_loader.py
```

## License

MIT License - see LICENSE file for details.