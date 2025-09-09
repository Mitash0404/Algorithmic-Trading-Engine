"""
Cointegration Strategy Module for Algorithmic Trading Engine
Implements mean-reversion trading based on cointegrated pairs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, NamedTuple
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
from loguru import logger

warnings.filterwarnings('ignore')

class CointegrationResult(NamedTuple):
    """Result of cointegration test"""
    is_cointegrated: bool
    p_value: float
    cointegration_vector: np.ndarray
    half_life: float
    z_score: float
    spread: np.ndarray

class CointegrationStrategy:
    """Cointegration-based mean reversion trading strategy"""
    
    def __init__(self, config: dict):
        """Initialize the cointegration strategy"""
        self.config = config
        self.lookback_period = config['cointegration']['lookback_period']
        self.significance_level = config['cointegration']['significance_level']
        self.min_half_life = config['cointegration']['min_half_life']
        self.max_half_life = config['cointegration']['max_half_life']
        self.z_score_threshold = config['cointegration']['z_score_threshold']
        self.exit_threshold = config['cointegration']['exit_threshold']
        
        # Strategy state
        self.cointegrated_pairs = []
        self.current_positions = {}
        self.position_history = []
        
        logger.info("CointegrationStrategy initialized successfully")
    
    def find_cointegrated_pairs(self, price_data: Dict[str, pd.DataFrame]) -> List[Tuple[str, str, CointegrationResult]]:
        """Find cointegrated pairs from price data"""
        symbols = list(price_data.keys())
        n_symbols = len(symbols)
        cointegrated_pairs = []
        
        logger.info(f"Searching for cointegrated pairs among {n_symbols} symbols")
        
        for i in range(n_symbols):
            for j in range(i + 1, n_symbols):
                symbol1, symbol2 = symbols[i], symbols[j]
                
                try:
                    # Get price data for both symbols
                    prices1 = price_data[symbol1]['Close'].dropna()
                    prices2 = price_data[symbol2]['Close'].dropna()
                    
                    # Align data
                    common_dates = prices1.index.intersection(prices2.index)
                    if len(common_dates) < self.lookback_period:
                        continue
                    
                    prices1_aligned = prices1.loc[common_dates]
                    prices2_aligned = prices2.loc[common_dates]
                    
                    # Test for cointegration
                    result = self._test_cointegration(prices1_aligned, prices2_aligned)
                    
                    if result.is_cointegrated:
                        cointegrated_pairs.append((symbol1, symbol2, result))
                        logger.info(f"Found cointegrated pair: {symbol1}-{symbol2} (p={result.p_value:.4f})")
                
                except Exception as e:
                    logger.warning(f"Error testing {symbol1}-{symbol2}: {e}")
                    continue
        
        # Sort by p-value (most significant first)
        cointegrated_pairs.sort(key=lambda x: x[2].p_value)
        
        self.cointegrated_pairs = cointegrated_pairs
        logger.info(f"Found {len(cointegrated_pairs)} cointegrated pairs")
        
        return cointegrated_pairs
    
    def _test_cointegration(self, prices1: pd.Series, prices2: pd.Series) -> CointegrationResult:
        """Test for cointegration between two price series"""
        try:
            # Perform Engle-Granger cointegration test
            score, p_value, _ = coint(prices1, prices2)
            
            if p_value > self.significance_level:
                return CointegrationResult(False, p_value, np.array([]), 0.0, 0.0, np.array([]))
            
            # Estimate cointegration vector using OLS
            model = OLS(prices1, prices2).fit()
            cointegration_vector = np.array([1.0, -model.params[0]])
            
            # Calculate spread
            spread = prices1 - model.params[0] * prices2
            
            # Calculate half-life
            half_life = self._calculate_half_life(spread)
            
            # Check half-life constraints
            if not (self.min_half_life <= half_life <= self.max_half_life):
                return CointegrationResult(False, p_value, cointegration_vector, half_life, 0.0, spread)
            
            # Calculate current z-score
            z_score = self._calculate_z_score(spread)
            
            return CointegrationResult(
                is_cointegrated=True,
                p_value=p_value,
                cointegration_vector=cointegration_vector,
                half_life=half_life,
                z_score=z_score,
                spread=spread
            )
            
        except Exception as e:
            logger.error(f"Error in cointegration test: {e}")
            return CointegrationResult(False, 1.0, np.array([]), 0.0, 0.0, np.array([]))
    
    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """Calculate half-life of mean reversion"""
        try:
            # Calculate spread changes
            spread_changes = np.diff(spread)
            spread_lagged = spread[:-1]
            
            # Regression: Δspread = α + β * spread_lagged
            if len(spread_changes) < 2:
                return 0.0
            
            model = OLS(spread_changes, spread_lagged).fit()
            beta = model.params[0]
            
            if beta >= 0:
                return 0.0  # No mean reversion
            
            # Half-life = ln(2) / |β|
            half_life = np.log(2) / abs(beta)
            
            return half_life
            
        except Exception as e:
            logger.error(f"Error calculating half-life: {e}")
            return 0.0
    
    def _calculate_z_score(self, spread: np.ndarray, window: int = 20) -> float:
        """Calculate z-score of current spread"""
        try:
            if len(spread) < window:
                return 0.0
            
            # Use rolling mean and std for stability
            rolling_mean = pd.Series(spread).rolling(window=window).mean()
            rolling_std = pd.Series(spread).rolling(window=window).std()
            
            current_spread = spread[-1]
            current_mean = rolling_mean.iloc[-1]
            current_std = rolling_std.iloc[-1]
            
            if pd.isna(current_mean) or pd.isna(current_std) or current_std == 0:
                return 0.0
            
            z_score = (current_spread - current_mean) / current_std
            return z_score
            
        except Exception as e:
            logger.error(f"Error calculating z-score: {e}")
            return 0.0
    
    def generate_signals(self, price_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Generate trading signals based on cointegrated pairs"""
        signals = []
        
        if not self.cointegrated_pairs:
            logger.warning("No cointegrated pairs available for signal generation")
            return signals
        
        for symbol1, symbol2, result in self.cointegrated_pairs:
            try:
                # Get current prices
                current_price1 = price_data[symbol1]['Close'].iloc[-1]
                current_price2 = price_data[symbol2]['Close'].iloc[-1]
                
                # Calculate current spread
                current_spread = current_price1 - result.cointegration_vector[1] * current_price2
                
                # Calculate current z-score
                current_z_score = self._calculate_z_score(result.spread)
                
                # Generate signal based on z-score
                signal = self._generate_pair_signal(
                    symbol1, symbol2, current_z_score, 
                    result.cointegration_vector, current_price1, current_price2
                )
                
                if signal:
                    signals.append(signal)
                    logger.info(f"Generated signal: {signal}")
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol1}-{symbol2}: {e}")
                continue
        
        return signals
    
    def _generate_pair_signal(self, symbol1: str, symbol2: str, z_score: float, 
                             cointegration_vector: np.ndarray, price1: float, price2: float) -> Optional[Dict]:
        """Generate trading signal for a specific pair"""
        try:
            signal = {
                'pair': f"{symbol1}-{symbol2}",
                'timestamp': pd.Timestamp.now(),
                'z_score': z_score,
                'action': None,
                'symbol1': symbol1,
                'symbol2': symbol2,
                'price1': price1,
                'price2': price2,
                'confidence': 0.0
            }
            
            # Entry signals
            if z_score > self.z_score_threshold:
                # Spread is high, expect mean reversion down
                signal['action'] = 'SHORT_SPREAD'
                signal['confidence'] = min(abs(z_score) / 3.0, 1.0)  # Normalize confidence
                
            elif z_score < -self.z_score_threshold:
                # Spread is low, expect mean reversion up
                signal['action'] = 'LONG_SPREAD'
                signal['confidence'] = min(abs(z_score) / 3.0, 1.0)
            
            # Exit signals for existing positions
            elif abs(z_score) < self.exit_threshold:
                signal['action'] = 'EXIT_SPREAD'
                signal['confidence'] = 1.0 - abs(z_score) / self.exit_threshold
            
            return signal if signal['action'] else None
            
        except Exception as e:
            logger.error(f"Error generating pair signal: {e}")
            return None
    
    def calculate_position_sizes(self, signals: List[Dict], capital: float, 
                                risk_per_trade: float = 0.02) -> List[Dict]:
        """Calculate position sizes for signals"""
        positioned_signals = []
        
        for signal in signals:
            try:
                # Calculate position size based on Kelly criterion and risk
                confidence = signal.get('confidence', 0.0)
                position_size = capital * risk_per_trade * confidence
                
                # Apply position size constraints
                max_position = capital * self.config['trading']['max_position_size']
                position_size = min(position_size, max_position)
                
                # Calculate shares for each symbol
                if signal['action'] in ['LONG_SPREAD', 'SHORT_SPREAD']:
                    # Equal dollar allocation to both sides
                    symbol1_size = position_size / 2
                    symbol2_size = position_size / 2
                    
                    shares1 = int(symbol1_size / signal['price1'])
                    shares2 = int(symbol2_size / signal['price2'])
                    
                    signal['shares1'] = shares1
                    signal['shares2'] = shares2
                    signal['position_size'] = position_size
                    
                    positioned_signals.append(signal)
                
            except Exception as e:
                logger.error(f"Error calculating position size for {signal}: {e}")
                continue
        
        return positioned_signals
    
    def backtest_strategy(self, price_data: Dict[str, pd.DataFrame], 
                         initial_capital: float = 1000000) -> Dict:
        """Backtest the cointegration strategy"""
        logger.info("Starting strategy backtest")
        
        try:
            # Find cointegrated pairs
            cointegrated_pairs = self.find_cointegrated_pairs(price_data)
            
            if not cointegrated_pairs:
                logger.warning("No cointegrated pairs found for backtesting")
                return {}
            
            # Initialize backtest variables
            capital = initial_capital
            positions = {}
            trades = []
            equity_curve = []
            
            # Get common date range
            common_dates = self._get_common_dates(price_data)
            
            for date in common_dates:
                try:
                    # Get current prices
                    current_prices = {}
                    for symbol in price_data:
                        if date in price_data[symbol].index:
                            current_prices[symbol] = price_data[symbol].loc[date, 'Close']
                    
                    if len(current_prices) < 2:
                        continue
                    
                    # Generate signals
                    signals = self.generate_signals({symbol: price_data[symbol].loc[:date] 
                                                   for symbol in current_prices})
                    
                    # Execute trades
                    for signal in signals:
                        trade = self._execute_signal(signal, current_prices, capital)
                        if trade:
                            trades.append(trade)
                            capital += trade['pnl']
                    
                    # Update positions and calculate P&L
                    daily_pnl = self._calculate_daily_pnl(positions, current_prices)
                    capital += daily_pnl
                    
                    # Record equity
                    equity_curve.append({
                        'date': date,
                        'capital': capital,
                        'positions': len(positions),
                        'daily_pnl': daily_pnl
                    })
                    
                except Exception as e:
                    logger.error(f"Error in backtest iteration for {date}: {e}")
                    continue
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(equity_curve, initial_capital)
            
            backtest_results = {
                'initial_capital': initial_capital,
                'final_capital': capital,
                'total_return': (capital - initial_capital) / initial_capital,
                'performance': performance,
                'trades': trades,
                'equity_curve': equity_curve,
                'cointegrated_pairs': len(cointegrated_pairs)
            }
            
            logger.info(f"Backtest completed. Final capital: ${capital:,.2f}")
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return {}
    
    def _get_common_dates(self, price_data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Get common dates across all symbols"""
        common_dates = None
        for symbol, data in price_data.items():
            if common_dates is None:
                common_dates = data.index
            else:
                common_dates = common_dates.intersection(data.index)
        return common_dates
    
    def _execute_signal(self, signal: Dict, current_prices: Dict, capital: float) -> Optional[Dict]:
        """Execute a trading signal"""
        try:
            # Simplified trade execution for backtesting
            trade = {
                'timestamp': signal['timestamp'],
                'pair': signal['pair'],
                'action': signal['action'],
                'symbol1': signal['symbol1'],
                'symbol2': signal['symbol2'],
                'shares1': signal.get('shares1', 0),
                'shares2': signal.get('shares2', 0),
                'price1': current_prices[signal['symbol1']],
                'price2': current_prices[signal['symbol2']],
                'pnl': 0.0
            }
            
            return trade
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return None
    
    def _calculate_daily_pnl(self, positions: Dict, current_prices: Dict) -> float:
        """Calculate daily P&L for current positions"""
        # Simplified P&L calculation for backtesting
        return 0.0
    
    def _calculate_performance_metrics(self, equity_curve: List[Dict], 
                                     initial_capital: float) -> Dict:
        """Calculate performance metrics from equity curve"""
        try:
            if not equity_curve:
                return {}
            
            # Extract capital values
            capitals = [point['capital'] for point in equity_curve]
            returns = pd.Series(capitals).pct_change().dropna()
            
            # Calculate metrics
            total_return = (capitals[-1] - initial_capital) / initial_capital
            annualized_return = total_return * 252 / len(equity_curve)
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Calculate max drawdown
            peak = capitals[0]
            max_drawdown = 0
            for capital in capitals:
                if capital > peak:
                    peak = capital
                drawdown = (peak - capital) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': len(equity_curve)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Test the cointegration strategy
    config = {
        'cointegration': {
            'lookback_period': 252,
            'significance_level': 0.05,
            'min_half_life': 5,
            'max_half_life': 252,
            'z_score_threshold': 2.0,
            'exit_threshold': 0.5
        },
        'trading': {
            'max_position_size': 0.1
        }
    }
    
    strategy = CointegrationStrategy(config)
    
    # Create sample data for testing
    dates = pd.date_range('2022-01-01', periods=252, freq='D')
    np.random.seed(42)
    
    # Create cointegrated series
    random_walk = np.cumsum(np.random.randn(252))
    series1 = 100 + random_walk
    series2 = 50 + 0.5 * random_walk + np.random.randn(252) * 0.1
    
    price_data = {
        'STOCK1': pd.DataFrame({
            'Close': series1,
            'Open': series1 * 0.99,
            'High': series1 * 1.01,
            'Low': series1 * 0.98,
            'Volume': np.random.randint(1000000, 10000000, 252)
        }, index=dates),
        'STOCK2': pd.DataFrame({
            'Close': series2,
            'Open': series2 * 0.99,
            'High': series2 * 1.01,
            'Low': series2 * 0.98,
            'Volume': np.random.randint(1000000, 10000000, 252)
        }, index=dates)
    }
    
    # Test cointegration detection
    pairs = strategy.find_cointegrated_pairs(price_data)
    print(f"Found {len(pairs)} cointegrated pairs")
    
    if pairs:
        for pair in pairs:
            print(f"Pair: {pair[0]}-{pair[1]}, p-value: {pair[2].p_value:.4f}")
    
    # Test signal generation
    signals = strategy.generate_signals(price_data)
    print(f"\nGenerated {len(signals)} signals")
    
    # Test backtesting
    results = strategy.backtest_strategy(price_data, initial_capital=1000000)
    if results:
        print(f"\nBacktest Results:")
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['performance'].get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {results['performance'].get('max_drawdown', 0):.2%}")
