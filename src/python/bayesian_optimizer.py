"""
Bayesian Optimization Module for Algorithmic Trading Engine
Tunes strategy hyperparameters using Bayesian optimization to maximize Sharpe ratio
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
from loguru import logger
import json
import pickle
from pathlib import Path
import scipy.stats as stats

warnings.filterwarnings('ignore')

class BayesianOptimizer:
    """Bayesian optimizer for trading strategy hyperparameters"""
    
    def __init__(self, config: dict, strategy_class: Callable, data_handler: Any):
        """Initialize the Bayesian optimizer"""
        self.config = config
        self.strategy_class = strategy_class
        self.data_handler = data_handler
        
        # Optimization parameters
        self.n_iterations = 100
        self.n_random_starts = 10
        self.acquisition_function = 'ei'  # Expected Improvement
        
        # Hyperparameter bounds
        self.param_bounds = self._define_param_bounds()
        
        # Optimization history
        self.optimization_history = []
        self.best_params = None
        self.best_score = -np.inf
        
        # Gaussian Process
        self.gp = None
        self.scaler = StandardScaler()
        
        logger.info("BayesianOptimizer initialized successfully")
    
    def _define_param_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Define bounds for hyperparameters"""
        return {
            'z_score_threshold': (1.0, 4.0),
            'exit_threshold': (0.1, 1.0),
            'lookback_period': (50, 500),
            'min_half_life': (1, 20),
            'max_half_life': (100, 500),
            'significance_level': (0.01, 0.1),
            'max_position_size': (0.05, 0.2),
            'risk_per_trade': (0.01, 0.05),
            'volatility_window': (10, 50),
            'correlation_threshold': (0.3, 0.8),
            'momentum_window': (5, 30),
            'mean_reversion_strength': (0.1, 2.0),
            'stop_loss': (0.02, 0.1),
            'take_profit': (0.05, 0.3),
            'max_correlation': (0.7, 0.95)
        }
    
    def objective_function(self, params: np.ndarray) -> float:
        """Objective function to maximize (Sharpe ratio)"""
        try:
            # Convert numpy array to parameter dict
            param_dict = self._array_to_params(params)
            
            # Create strategy with new parameters
            strategy_config = self._create_strategy_config(param_dict)
            strategy = self.strategy_class(strategy_config)
            
            # Run backtest
            results = self._run_backtest(strategy)
            
            if not results:
                return -np.inf
            
            # Extract performance metrics
            sharpe_ratio = results['performance'].get('sharpe_ratio', -np.inf)
            max_drawdown = results['performance'].get('max_drawdown', 1.0)
            total_return = results['performance'].get('total_return', -np.inf)
            
            # Penalize excessive drawdown
            if max_drawdown > self.config['risk']['max_drawdown']:
                sharpe_ratio *= 0.5
            
            # Penalize negative returns
            if total_return < 0:
                sharpe_ratio *= 0.7
            
            # Store result
            self.optimization_history.append({
                'params': param_dict,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_return': total_return,
                'timestamp': pd.Timestamp.now()
            })
            
            # Update best result
            if sharpe_ratio > self.best_score:
                self.best_score = sharpe_ratio
                self.best_params = param_dict
                logger.info(f"New best Sharpe ratio: {sharpe_ratio:.3f}")
            
            return sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return -np.inf
    
    def _array_to_params(self, params: np.ndarray) -> Dict[str, float]:
        """Convert numpy array to parameter dictionary"""
        param_names = list(self.param_bounds.keys())
        param_dict = {}
        
        for i, name in enumerate(param_names):
            if i < len(params):
                # Ensure parameters are within bounds
                min_val, max_val = self.param_bounds[name]
                param_dict[name] = np.clip(params[i], min_val, max_val)
        
        return param_dict
    
    def _create_strategy_config(self, params: Dict[str, float]) -> dict:
        """Create strategy configuration from parameters"""
        config = self.config.copy()
        
        # Update cointegration parameters
        if 'z_score_threshold' in params:
            config['cointegration']['z_score_threshold'] = params['z_score_threshold']
        if 'exit_threshold' in params:
            config['cointegration']['exit_threshold'] = params['exit_threshold']
        if 'lookback_period' in params:
            config['cointegration']['lookback_period'] = int(params['lookback_period'])
        if 'min_half_life' in params:
            config['cointegration']['min_half_life'] = int(params['min_half_life'])
        if 'max_half_life' in params:
            config['cointegration']['max_half_life'] = int(params['max_half_life'])
        if 'significance_level' in params:
            config['cointegration']['significance_level'] = params['significance_level']
        
        # Update trading parameters
        if 'max_position_size' in params:
            config['trading']['max_position_size'] = params['max_position_size']
        if 'stop_loss' in params:
            config['risk']['stop_loss'] = params['stop_loss']
        if 'take_profit' in params:
            config['risk']['take_profit'] = params['take_profit']
        
        return config
    
    def _run_backtest(self, strategy: Any) -> Dict:
        """Run backtest for the strategy"""
        try:
            # Get historical data
            symbols = self.config['data']['symbols'][:5]  # Limit for faster testing
            price_data = {}
            
            for symbol in symbols:
                data = self.data_handler.get_price_history(symbol, days=252)
                if not data.empty:
                    price_data[symbol] = data
            
            if len(price_data) < 2:
                logger.warning("Insufficient data for backtesting")
                return {}
            
            # Run backtest
            results = strategy.backtest_strategy(price_data, initial_capital=1000000)
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return {}
    
    def optimize(self, n_iterations: int = None) -> Dict[str, Any]:
        """Run Bayesian optimization"""
        if n_iterations:
            self.n_iterations = n_iterations
        
        logger.info(f"Starting Bayesian optimization with {self.n_iterations} iterations")
        
        try:
            # Initial random points
            initial_points = self._generate_initial_points()
            
            # Evaluate initial points
            initial_scores = []
            for params in initial_points:
                score = self.objective_function(params)
                initial_scores.append(score)
            
            # Initialize Gaussian Process
            X_train = np.array(initial_points)
            y_train = np.array(initial_scores)
            
            # Scale inputs
            X_scaled = self.scaler.fit_transform(X_train)
            
            # Fit GP
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            self.gp = GaussianProcessRegressor(kernel=kernel, random_state=42)
            self.gp.fit(X_scaled, y_train)
            
            # Main optimization loop
            for i in range(self.n_iterations - self.n_random_starts):
                logger.info(f"Optimization iteration {i+1}/{self.n_iterations}")
                
                # Find next point to evaluate
                next_point = self._acquire_next_point()
                
                # Evaluate objective function
                score = self.objective_function(next_point)
                
                # Update training data
                X_train = np.vstack([X_train, next_point.reshape(1, -1)])
                y_train = np.append(y_train, score)
                
                # Retrain GP
                X_scaled = self.scaler.transform(X_train)
                self.gp.fit(X_scaled, y_train)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Iteration {i+1}: Best Sharpe = {self.best_score:.3f}")
            
            # Final results
            optimization_results = {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'optimization_history': self.optimization_history,
                'total_iterations': len(self.optimization_history)
            }
            
            logger.info(f"Optimization completed. Best Sharpe ratio: {self.best_score:.3f}")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return {}
    
    def _generate_initial_points(self) -> List[np.ndarray]:
        """Generate initial random points for optimization"""
        initial_points = []
        
        for _ in range(self.n_random_starts):
            point = []
            for param_name, (min_val, max_val) in self.param_bounds.items():
                if param_name in ['lookback_period', 'min_half_life', 'max_half_life', 'volatility_window', 'momentum_window']:
                    # Integer parameters
                    point.append(np.random.randint(min_val, max_val + 1))
                else:
                    # Float parameters
                    point.append(np.random.uniform(min_val, max_val))
            
            initial_points.append(np.array(point))
        
        return initial_points
    
    def _acquire_next_point(self) -> np.ndarray:
        """Acquire next point using acquisition function"""
        try:
            # Generate candidate points
            n_candidates = 1000
            candidates = []
            
            for _ in range(n_candidates):
                point = []
                for param_name, (min_val, max_val) in self.param_bounds.items():
                    if param_name in ['lookback_period', 'min_half_life', 'max_half_life', 'volatility_window', 'momentum_window']:
                        point.append(np.random.randint(min_val, max_val + 1))
                    else:
                        point.append(np.random.uniform(min_val, max_val))
                
                candidates.append(np.array(point))
            
            candidates = np.array(candidates)
            
            # Scale candidates
            candidates_scaled = self.scaler.transform(candidates)
            
            # Predict mean and std
            mean_pred, std_pred = self.gp.predict(candidates_scaled, return_std=True)
            
            # Calculate acquisition function values
            if self.acquisition_function == 'ei':
                # Expected Improvement
                best_f = np.max(self.gp.y_train_)
                improvement = mean_pred - best_f
                z = improvement / (std_pred + 1e-8)
                ei = improvement * stats.norm.cdf(z) + std_pred * stats.norm.pdf(z)
                ei[std_pred == 0] = 0
                acquisition_values = ei
            else:
                # Upper Confidence Bound
                acquisition_values = mean_pred + 2 * std_pred
            
            # Select best candidate
            best_idx = np.argmax(acquisition_values)
            return candidates[best_idx]
            
        except Exception as e:
            logger.error(f"Error acquiring next point: {e}")
            # Fallback to random point
            return self._generate_initial_points()[0]
    
    def save_results(self, filepath: str):
        """Save optimization results to file"""
        try:
            results = {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'optimization_history': self.optimization_history,
                'param_bounds': self.param_bounds,
                'config': self.config
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(results, f)
            
            logger.info(f"Optimization results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def load_results(self, filepath: str):
        """Load optimization results from file"""
        try:
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
            
            self.best_params = results['best_params']
            self.best_score = results['best_score']
            self.optimization_history = results['optimization_history']
            
            logger.info(f"Optimization results loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")
    
    def plot_optimization_history(self):
        """Plot optimization history"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.optimization_history:
                logger.warning("No optimization history to plot")
                return
            
            # Extract data
            iterations = range(1, len(self.optimization_history) + 1)
            scores = [h['sharpe_ratio'] for h in self.optimization_history]
            drawdowns = [h['max_drawdown'] for h in self.optimization_history]
            
            # Create subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot Sharpe ratio progression
            ax1.plot(iterations, scores, 'b-', alpha=0.7)
            ax1.axhline(y=self.best_score, color='r', linestyle='--', label=f'Best: {self.best_score:.3f}')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Sharpe Ratio')
            ax1.set_title('Optimization Progress - Sharpe Ratio')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot max drawdown progression
            ax2.plot(iterations, drawdowns, 'g-', alpha=0.7)
            ax2.axhline(y=self.config['risk']['max_drawdown'], color='r', linestyle='--', 
                        label=f'Target: {self.config["risk"]["max_drawdown"]:.1%}')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Max Drawdown')
            ax2.set_title('Optimization Progress - Max Drawdown')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting optimization history: {e}")
    
    def get_parameter_importance(self) -> Dict[str, float]:
        """Estimate parameter importance using GP feature analysis"""
        try:
            if self.gp is None:
                logger.warning("GP not fitted yet")
                return {}
            
            # Get kernel parameters
            kernel_params = self.gp.kernel_.get_params()
            
            # Extract length scales (inverse of importance)
            if 'k2__length_scale' in kernel_params:
                length_scales = kernel_params['k2__length_scale']
                if hasattr(length_scales, '__len__'):
                    # Convert to importance scores
                    importance = 1.0 / (length_scales + 1e-8)
                    importance = importance / importance.sum()
                    
                    param_names = list(self.param_bounds.keys())
                    param_importance = dict(zip(param_names, importance))
                    
                    # Sort by importance
                    param_importance = dict(sorted(param_importance.items(), 
                                                 key=lambda x: x[1], reverse=True))
                    
                    return param_importance
            
            return {}
            
        except Exception as e:
            logger.error(f"Error calculating parameter importance: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    # Test the Bayesian optimizer
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
        },
        'risk': {
            'max_drawdown': 0.15,
            'stop_loss': 0.05,
            'take_profit': 0.15
        },
        'data': {
            'symbols': ['AAPL', 'MSFT', 'GOOGL']
        }
    }
    
    # Mock strategy class and data handler for testing
    class MockStrategy:
        def __init__(self, config):
            self.config = config
        
        def backtest_strategy(self, price_data, initial_capital):
            # Mock backtest results
            return {
                'performance': {
                    'sharpe_ratio': np.random.normal(1.0, 0.3),
                    'max_drawdown': np.random.uniform(0.05, 0.25),
                    'total_return': np.random.uniform(-0.1, 0.3)
                }
            }
    
    class MockDataHandler:
        def get_price_history(self, symbol, days):
            # Mock price data
            dates = pd.date_range('2022-01-01', periods=days, freq='D')
            return pd.DataFrame({
                'Close': np.random.randn(days).cumsum() + 100,
                'Open': np.random.randn(days).cumsum() + 100,
                'High': np.random.randn(days).cumsum() + 100,
                'Low': np.random.randn(days).cumsum() + 100,
                'Volume': np.random.randint(1000000, 10000000, days)
            }, index=dates)
    
    # Create optimizer
    optimizer = BayesianOptimizer(config, MockStrategy, MockDataHandler())
    
    # Run optimization (with fewer iterations for testing)
    results = optimizer.optimize(n_iterations=20)
    
    if results:
        print(f"Optimization completed!")
        print(f"Best Sharpe ratio: {results['best_score']:.3f}")
        print(f"Best parameters: {results['best_params']}")
        
        # Get parameter importance
        importance = optimizer.get_parameter_importance()
        if importance:
            print("\nParameter Importance:")
            for param, imp in list(importance.items())[:5]:
                print(f"  {param}: {imp:.3f}")
