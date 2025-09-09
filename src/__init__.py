"""
Algorithmic Trading Engine

A high-performance algorithmic trading engine with C++ extensions for optimal performance.
"""

__version__ = "1.0.0"
__author__ = "Trading Team"
__email__ = "team@trading.com"

# Import core trading engine components
from .python.trading_engine import TradingEngine
from .python.ibkr_client import IBKRClient
from .python.cointegration_strategy import CointegrationStrategy
from .python.bayesian_optimizer import BayesianOptimizer

__all__ = [
    "TradingEngine",
    "IBKRClient", 
    "CointegrationStrategy",
    "BayesianOptimizer"
]
