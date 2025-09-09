#!/usr/bin/env python3
"""
Main Entry Point for Algorithmic Trading Engine
"""

import argparse
import sys
import os
import time
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from python.trading_engine import TradingEngine

def setup_logging():
    """Setup logging configuration"""
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "trading_engine.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days"
    )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Algorithmic Trading Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in optimization mode
  python main.py --optimize
  
  # Run in live trading mode
  python main.py --live
  
  # Run in paper trading mode
  python main.py --paper
  
  # Run with custom config
  python main.py --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run in optimization mode to tune hyperparameters"
    )
    
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run in live trading mode (requires IBKR connection)"
    )
    
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Run in paper trading mode (no real money)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="Override symbols from config"
    )
    
    parser.add_argument(
        "--capital",
        type=float,
        help="Override initial capital from config"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    if args.verbose:
        logger.add(sys.stdout, level="DEBUG")
    
    logger.info("Starting Algorithmic Trading Engine")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    try:
        # Check if config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        
        # Create trading engine
        engine = TradingEngine(str(config_path))
        
        # Determine mode
        if args.optimize:
            logger.info("Starting in optimization mode")
            engine.start(trading_enabled=False, optimization_mode=True)
        elif args.live:
            logger.info("Starting in live trading mode")
            engine.start(trading_enabled=True, optimization_mode=False)
        elif args.paper:
            logger.info("Starting in paper trading mode")
            engine.start(trading_enabled=False, optimization_mode=False)
        else:
            logger.info("Starting in default mode (paper trading)")
            engine.start(trading_enabled=False, optimization_mode=False)
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
                
                # Get performance summary every minute
                if int(time.time()) % 60 == 0:
                    performance = engine.get_performance_summary()
                    if performance:
                        logger.info(f"Performance: Capital=${performance.get('current_capital', 0):,.2f}, "
                                  f"Return={performance.get('total_return', 0):.2%}, "
                                  f"Drawdown={performance.get('current_drawdown', 0):.2%}")
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        
        finally:
            engine.shutdown()
            logger.info("Engine shutdown complete")
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
