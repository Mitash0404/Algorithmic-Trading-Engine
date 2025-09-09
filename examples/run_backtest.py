"""
Example: Run a backtest on historical data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python.trading_engine import TradingEngine
from python.cointegration_strategy import CointegrationStrategy


def generate_sample_data(symbols, days=252):
    """Generate sample historical data for backtesting."""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    data = {}
    for symbol in symbols:
        # Generate realistic price data with some correlation
        np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
        returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))  # Price series
        data[symbol] = prices
    
    return pd.DataFrame(data, index=dates)


def run_backtest_example():
    """Run a complete backtest example."""
    print("=== Algorithmic Trading Engine - Backtest Example ===\n")
    
    # Initialize trading engine
    print("1. Initializing trading engine...")
    engine = TradingEngine("config/config.yaml")
    
    # Generate sample data
    print("2. Loading historical data...")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    historical_data = generate_sample_data(symbols, days=252)
    print(f"   Loaded {len(historical_data)} days of data for {len(symbols)} symbols")
    
    # Initialize strategy
    print("3. Setting up cointegration strategy...")
    strategy_config = {
        'cointegration': {
            'lookback_period': 60,
            'z_score_threshold': 2.0,
            'exit_threshold': 0.5,
            'min_half_life': 5,
            'max_half_life': 50
        }
    }
    strategy = CointegrationStrategy(strategy_config)
    
    # Run backtest
    print("4. Running backtest...")
    results = strategy.backtest(
        historical_data, 
        start_date='2023-01-01', 
        end_date='2023-12-31'
    )
    
    # Display results
    print("\n5. Backtest Results:")
    print("=" * 50)
    print(f"Total Return:        {results['total_return']:.2%}")
    print(f"Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:        {results['max_drawdown']:.2%}")
    print(f"Win Rate:            {results['win_rate']:.2%}")
    print(f"Total Trades:        {results['total_trades']}")
    print(f"Average Trade:       {results['avg_trade_return']:.2%}")
    
    # Performance by month
    print("\n6. Monthly Performance:")
    print("-" * 30)
    monthly_returns = results.get('monthly_returns', {})
    for month, ret in monthly_returns.items():
        print(f"{month}: {ret:.2%}")
    
    # Strategy statistics
    print("\n7. Strategy Statistics:")
    print("-" * 30)
    print(f"Cointegration Pairs Found: {results.get('cointegrated_pairs', 0)}")
    print(f"Average Position Days:     {results.get('avg_position_days', 0):.1f}")
    print(f"Profit Factor:             {results.get('profit_factor', 0):.2f}")
    
    print("\n=== Backtest Complete ===")
    return results


if __name__ == "__main__":
    results = run_backtest_example()
