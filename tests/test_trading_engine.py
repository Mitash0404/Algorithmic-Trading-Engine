"""
Basic tests for the trading engine core functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python.trading_engine import TradingEngine
from python.cointegration_strategy import CointegrationStrategy
from python.data_handler import DataHandler


class TestTradingEngine:
    """Test core trading engine functionality."""
    
    def test_engine_initialization(self):
        """Test that engine initializes properly."""
        engine = TradingEngine("config/config.yaml")
        assert engine.config_path == "config/config.yaml"
        assert engine.trading_enabled is False
        assert engine.optimization_mode is False
    
    def test_paper_trading_mode(self):
        """Test paper trading mode setup."""
        engine = TradingEngine()
        engine.start(trading_enabled=False, optimization_mode=False)
        assert engine.trading_enabled is False
        assert engine.optimization_mode is False
    
    def test_config_loading(self):
        """Test configuration loading."""
        engine = TradingEngine()
        # Should load default config without errors
        assert engine.config is not None
        assert 'trading' in engine.config
        assert 'cointegration' in engine.config
    
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        engine = TradingEngine()
        engine._initialize_strategies()
        assert engine.strategies is not None
        assert len(engine.strategies) > 0
    
    def test_data_handler_setup(self):
        """Test data handler initialization."""
        engine = TradingEngine()
        engine._initialize_data_handler()
        assert engine.data_handler is not None


class TestCointegrationStrategy:
    """Test cointegration strategy functionality."""
    
    def test_strategy_initialization(self):
        """Test strategy initializes with config."""
        config = {
            'cointegration': {
                'lookback_period': 252,
                'z_score_threshold': 2.0,
                'exit_threshold': 0.5
            }
        }
        strategy = CointegrationStrategy(config)
        assert strategy.lookback_period == 252
        assert strategy.z_score_threshold == 2.0
    
    def test_signal_generation(self):
        """Test signal generation with mock data."""
        config = {'cointegration': {'lookback_period': 50}}
        strategy = CointegrationStrategy(config)
        
        # Mock price data
        prices = pd.DataFrame({
            'AAPL': np.random.randn(100).cumsum() + 100,
            'MSFT': np.random.randn(100).cumsum() + 100
        })
        
        signals = strategy.generate_signals(prices)
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) == len(prices)
    
    def test_cointegration_test(self):
        """Test cointegration testing functionality."""
        config = {'cointegration': {'lookback_period': 50}}
        strategy = CointegrationStrategy(config)
        
        # Create cointegrated series
        x = np.random.randn(100)
        y = 2 * x + np.random.randn(100) * 0.1
        
        is_cointegrated, p_value = strategy._test_cointegration(x, y)
        assert isinstance(is_cointegrated, bool)
        assert 0 <= p_value <= 1


class TestDataHandler:
    """Test data handling functionality."""
    
    def test_data_handler_initialization(self):
        """Test data handler initializes properly."""
        handler = DataHandler()
        assert handler is not None
    
    def test_symbol_validation(self):
        """Test symbol validation."""
        handler = DataHandler()
        valid_symbols = ['AAPL', 'MSFT', 'GOOGL']
        invalid_symbols = ['INVALID', '', None]
        
        for symbol in valid_symbols:
            assert handler._validate_symbol(symbol) is True
        
        for symbol in invalid_symbols:
            assert handler._validate_symbol(symbol) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
