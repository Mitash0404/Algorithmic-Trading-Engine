"""
Main Trading Engine for Algorithmic Trading System
Orchestrates data handling, strategy execution, and live trading
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger
import yaml
from pathlib import Path
import signal
import sys

# Import our modules
from .data_handler import DataHandler
from .cointegration_strategy import CointegrationStrategy
from .bayesian_optimizer import BayesianOptimizer
from .ibkr_client import IBKRClient, TradeSignal, OrderType, OrderAction

class TradingEngine:
    """Main trading engine orchestrating all components"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the trading engine"""
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.data_handler = DataHandler(config_path)
        self.strategy = CointegrationStrategy(self.config)
        self.ibkr_client = IBKRClient(self.config)
        
        # Engine state
        self.running = False
        self.trading_enabled = False
        self.optimization_mode = False
        
        # Performance tracking
        self.performance_metrics = {}
        self.trade_history = []
        self.daily_pnl = []
        
        # Risk management
        self.max_drawdown = self.config['risk']['max_drawdown']
        self.current_drawdown = 0.0
        self.initial_capital = self.config['trading']['initial_capital']
        self.current_capital = self.initial_capital
        
        # Threading
        self.main_thread = None
        self.data_thread = None
        self.trading_thread = None
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("TradingEngine initialized successfully")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
    
    def start(self, trading_enabled: bool = False, optimization_mode: bool = False):
        """Start the trading engine"""
        try:
            logger.info("Starting Trading Engine...")
            
            self.trading_enabled = trading_enabled
            self.optimization_mode = optimization_mode
            
            if optimization_mode:
                self._start_optimization_mode()
            else:
                self._start_live_mode()
            
            self.running = True
            logger.info("Trading Engine started successfully")
            
        except Exception as e:
            logger.error(f"Error starting Trading Engine: {e}")
            self.shutdown()
    
    def _start_optimization_mode(self):
        """Start optimization mode"""
        logger.info("Starting optimization mode...")
        
        # Create Bayesian optimizer
        optimizer = BayesianOptimizer(self.config, CointegrationStrategy, self.data_handler)
        
        # Run optimization
        results = optimizer.optimize(n_iterations=50)  # Reduced for testing
        
        if results:
            logger.info(f"Optimization completed with best Sharpe ratio: {results['best_score']:.3f}")
            
            # Update configuration with best parameters
            self._update_config_with_optimized_params(results['best_params'])
            
            # Save optimization results
            optimizer.save_results("optimization_results.pkl")
            
            # Plot results
            optimizer.plot_optimization_history()
            
            # Switch to live mode with optimized parameters
            self.trading_enabled = True
            self._start_live_mode()
        else:
            logger.error("Optimization failed")
            self.shutdown()
    
    def _start_live_mode(self):
        """Start live trading mode"""
        logger.info("Starting live trading mode...")
        
        # Connect to IBKR if trading is enabled
        if self.trading_enabled:
            if not self.ibkr_client.connect():
                logger.error("Failed to connect to IBKR, switching to paper trading mode")
                self.trading_enabled = False
        
        # Start data collection thread
        self.data_thread = threading.Thread(target=self._data_collection_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # Start trading thread if enabled
        if self.trading_enabled:
            self.trading_thread = threading.Thread(target=self._trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
        
        # Start main engine loop
        self.main_thread = threading.Thread(target=self._main_loop)
        self.main_thread.daemon = True
        self.main_thread.start()
    
    def _data_collection_loop(self):
        """Data collection loop running in separate thread"""
        logger.info("Data collection loop started")
        
        while self.running:
            try:
                # Update market data
                asyncio.run(self.data_handler.update_all_data())
                
                # Wait for next update
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                time.sleep(10)
    
    def _trading_loop(self):
        """Trading loop running in separate thread"""
        logger.info("Trading loop started")
        
        while self.running and self.trading_enabled:
            try:
                # Generate trading signals
                signals = self._generate_trading_signals()
                
                # Execute signals
                for signal in signals:
                    if self._should_execute_signal(signal):
                        self._execute_trading_signal(signal)
                
                # Update positions and P&L
                self._update_positions_and_pnl()
                
                # Risk management checks
                self._risk_management_checks()
                
                # Wait for next iteration
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(10)
    
    def _main_loop(self):
        """Main engine loop"""
        logger.info("Main engine loop started")
        
        while self.running:
            try:
                # Update performance metrics
                self._update_performance_metrics()
                
                # Log status
                self._log_engine_status()
                
                # Wait for next iteration
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(10)
    
    def _generate_trading_signals(self) -> List[Dict]:
        """Generate trading signals using the strategy"""
        try:
            # Get current price data
            price_data = {}
            for symbol in self.config['data']['symbols']:
                data = self.data_handler.get_price_history(symbol, days=252)
                if not data.empty:
                    price_data[symbol] = data
            
            if len(price_data) < 2:
                return []
            
            # Generate signals
            signals = self.strategy.generate_signals(price_data)
            
            # Calculate position sizes
            if signals:
                signals = self.strategy.calculate_position_sizes(
                    signals, self.current_capital
                )
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return []
    
    def _should_execute_signal(self, signal: Dict) -> bool:
        """Determine if a signal should be executed"""
        try:
            # Check if we have enough capital
            if signal.get('position_size', 0) > self.current_capital * 0.1:
                return False
            
            # Check risk limits
            if self.current_drawdown > self.max_drawdown:
                return False
            
            # Check if we already have a position in this pair
            pair_key = signal.get('pair', '')
            if any(trade['pair'] == pair_key for trade in self.trade_history[-10:]):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking signal execution: {e}")
            return False
    
    def _execute_trading_signal(self, signal: Dict):
        """Execute a trading signal"""
        try:
            if not self.trading_enabled:
                logger.info(f"Paper trading signal: {signal}")
                return
            
            # Convert signal to TradeSignal objects
            trade_signals = self._convert_signal_to_trades(signal)
            
            # Execute trades
            for trade_signal in trade_signals:
                order_id = self.ibkr_client.execute_trade_signal(trade_signal)
                
                if order_id:
                    # Record trade
                    trade_record = {
                        'timestamp': datetime.now(),
                        'order_id': order_id,
                        'signal': signal,
                        'trade_signal': trade_signal,
                        'status': 'EXECUTED'
                    }
                    self.trade_history.append(trade_record)
                    
                    logger.info(f"Trade executed: {trade_signal.symbol} {trade_signal.action} {trade_signal.quantity}")
                else:
                    logger.error(f"Failed to execute trade: {trade_signal}")
            
        except Exception as e:
            logger.error(f"Error executing trading signal: {e}")
    
    def _convert_signal_to_trades(self, signal: Dict) -> List[TradeSignal]:
        """Convert strategy signal to TradeSignal objects"""
        try:
            trades = []
            
            if signal['action'] == 'LONG_SPREAD':
                # Buy symbol1, sell symbol2
                trades.append(TradeSignal(
                    symbol=signal['symbol1'],
                    action=OrderAction.BUY.value,
                    quantity=signal['shares1'],
                    order_type=OrderType.MARKET
                ))
                trades.append(TradeSignal(
                    symbol=signal['symbol2'],
                    action=OrderAction.SELL.value,
                    quantity=signal['shares2'],
                    order_type=OrderType.MARKET
                ))
            
            elif signal['action'] == 'SHORT_SPREAD':
                # Sell symbol1, buy symbol2
                trades.append(TradeSignal(
                    symbol=signal['symbol1'],
                    action=OrderAction.SELL.value,
                    quantity=signal['shares1'],
                    order_type=OrderType.MARKET
                ))
                trades.append(TradeSignal(
                    symbol=signal['symbol2'],
                    action=OrderAction.BUY.value,
                    quantity=signal['shares2'],
                    order_type=OrderType.MARKET
                ))
            
            return trades
            
        except Exception as e:
            logger.error(f"Error converting signal to trades: {e}")
            return []
    
    def _update_positions_and_pnl(self):
        """Update current positions and calculate P&L"""
        try:
            if self.trading_enabled:
                # Get current positions from IBKR
                positions = self.ibkr_client.get_positions()
                
                # Calculate P&L
                total_pnl = 0.0
                for symbol, pos_info in positions.items():
                    # Simplified P&L calculation
                    # In practice, you'd track entry prices and calculate unrealized P&L
                    pass
                
                # Update current capital
                self.current_capital = self.initial_capital + total_pnl
                
                # Calculate drawdown
                if self.current_capital > self.initial_capital:
                    self.current_drawdown = 0.0
                else:
                    self.current_drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
            
        except Exception as e:
            logger.error(f"Error updating positions and P&L: {e}")
    
    def _risk_management_checks(self):
        """Perform risk management checks"""
        try:
            # Check drawdown limits
            if self.current_drawdown > self.max_drawdown:
                logger.warning(f"Maximum drawdown exceeded: {self.current_drawdown:.2%}")
                self._emergency_stop_trading()
            
            # Check capital limits
            if self.current_capital < self.initial_capital * 0.8:
                logger.warning(f"Capital below 80% threshold: ${self.current_capital:,.2f}")
                self._reduce_position_sizes()
            
        except Exception as e:
            logger.error(f"Error in risk management checks: {e}")
    
    def _emergency_stop_trading(self):
        """Emergency stop trading"""
        logger.error("EMERGENCY STOP: Stopping all trading activities")
        self.trading_enabled = False
        
        # Cancel all open orders
        if self.trading_enabled:
            for order_id in self.ibkr_client.orders:
                self.ibkr_client.cancel_order(order_id)
    
    def _reduce_position_sizes(self):
        """Reduce position sizes due to capital constraints"""
        logger.info("Reducing position sizes due to capital constraints")
        # Implementation would reduce the max_position_size parameter
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate basic metrics
            total_return = (self.current_capital - self.initial_capital) / self.initial_capital
            
            # Get IBKR performance metrics if available
            ibkr_metrics = {}
            if self.trading_enabled:
                ibkr_metrics = self.ibkr_client.get_performance_metrics()
            
            self.performance_metrics = {
                'timestamp': datetime.now(),
                'current_capital': self.current_capital,
                'total_return': total_return,
                'current_drawdown': self.current_drawdown,
                'fill_rate': ibkr_metrics.get('fill_rate', 0.0),
                'avg_order_latency': ibkr_metrics.get('avg_order_latency', 0.0),
                'total_trades': len(self.trade_history),
                'running': self.running,
                'trading_enabled': self.trading_enabled
            }
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _log_engine_status(self):
        """Log current engine status"""
        try:
            logger.info(f"Engine Status - Capital: ${self.current_capital:,.2f}, "
                       f"Return: {self.performance_metrics.get('total_return', 0):.2%}, "
                       f"Drawdown: {self.current_drawdown:.2%}, "
                       f"Trades: {len(self.trade_history)}")
            
        except Exception as e:
            logger.error(f"Error logging engine status: {e}")
    
    def _update_config_with_optimized_params(self, optimized_params: Dict):
        """Update configuration with optimized parameters"""
        try:
            # Update config with optimized parameters
            for param, value in optimized_params.items():
                if param in self.config.get('cointegration', {}):
                    self.config['cointegration'][param] = value
                elif param in self.config.get('trading', {}):
                    self.config['trading'][param] = value
                elif param in self.config.get('risk', {}):
                    self.config['risk'][param] = value
            
            # Update strategy with new config
            self.strategy = CointegrationStrategy(self.config)
            
            logger.info("Configuration updated with optimized parameters")
            
        except Exception as e:
            logger.error(f"Error updating config with optimized parameters: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        return self.performance_metrics.copy()
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history"""
        return self.trade_history.copy()
    
    def shutdown(self):
        """Shutdown the trading engine"""
        logger.info("Shutting down Trading Engine...")
        
        try:
            # Stop all threads
            self.running = False
            
            # Disconnect from IBKR
            if self.trading_enabled:
                self.ibkr_client.disconnect()
            
            # Close data handler connections
            self.data_handler.close_connections()
            
            # Wait for threads to finish
            if self.main_thread and self.main_thread.is_alive():
                self.main_thread.join(timeout=5)
            if self.data_thread and self.data_thread.is_alive():
                self.data_thread.join(timeout=5)
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=5)
            
            logger.info("Trading Engine shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Test the trading engine
    engine = TradingEngine()
    
    try:
        # Start in optimization mode first
        print("Starting Trading Engine in optimization mode...")
        engine.start(trading_enabled=False, optimization_mode=True)
        
        # Keep running for a while to see results
        time.sleep(30)
        
        # Get performance summary
        performance = engine.get_performance_summary()
        print(f"\nPerformance Summary:")
        for key, value in performance.items():
            print(f"  {key}: {value}")
        
        # Get trade history
        trades = engine.get_trade_history()
        print(f"\nTrade History ({len(trades)} trades):")
        for trade in trades[-5:]:  # Show last 5 trades
            print(f"  {trade}")
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        engine.shutdown()
        print("Engine shutdown complete")
