"""
Data Handler Module for Algorithmic Trading Engine
Handles data fetching, processing, and storage from multiple sources
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData
import redis
import pymongo
from sqlalchemy import create_engine, text
import yaml
from loguru import logger

class DataHandler:
    """Main data handler for fetching and processing market data"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize data handler with configuration"""
        self.config = self._load_config(config_path)
        self.symbols = self.config['data']['symbols']
        self.data_source = self.config['data']['data_source']
        self.update_frequency = self.config['data']['update_frequency']
        self.history_days = self.config['data']['history_days']
        
        # Initialize connections
        self.redis_client = None
        self.mongo_client = None
        self.postgres_engine = None
        self._init_connections()
        
        # Data storage
        self.price_data = {}
        self.volume_data = {}
        self.last_update = {}
        
        logger.info("DataHandler initialized successfully")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration if file not found"""
        return {
            'data': {
                'symbols': ['AAPL', 'MSFT', 'GOOGL'],
                'data_source': 'yfinance',
                'update_frequency': '1min',
                'history_days': 1000
            }
        }
    
    def _init_connections(self):
        """Initialize database connections"""
        try:
            # Redis connection
            redis_uri = self.config.get('database', {}).get('redis_uri', 'redis://localhost:6379/')
            self.redis_client = redis.from_url(redis_uri)
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        try:
            # MongoDB connection
            mongo_uri = self.config.get('database', {}).get('mongodb_uri', 'mongodb://localhost:27017/')
            self.mongo_client = pymongo.MongoClient(mongo_uri)
            self.mongo_client.admin.command('ping')
            logger.info("MongoDB connection established")
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}")
            self.mongo_client = None
        
        try:
            # PostgreSQL connection
            postgres_uri = self.config.get('database', {}).get('postgres_uri', 'postgresql://localhost/trading_db')
            self.postgres_engine = create_engine(postgres_uri)
            self.postgres_engine.connect().close()
            logger.info("PostgreSQL connection established")
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}")
            self.postgres_engine = None
    
    async def fetch_historical_data(self, symbol: str, days: int = None) -> pd.DataFrame:
        """Fetch historical data for a symbol"""
        if days is None:
            days = self.history_days
        
        try:
            if self.data_source == 'yfinance':
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=f"{days}d", interval="1d")
                
                if data.empty:
                    logger.warning(f"No data received for {symbol}")
                    return pd.DataFrame()
                
                # Clean and standardize data
                data = self._clean_data(data)
                
                # Store in memory and cache
                self.price_data[symbol] = data
                self._cache_data(symbol, data)
                
                logger.info(f"Fetched {len(data)} days of data for {symbol}")
                return data
                
            else:
                logger.error(f"Unsupported data source: {self.data_source}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data format"""
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if not all(col in data.columns for col in required_columns):
            logger.warning("Data missing required columns")
            return pd.DataFrame()
        
        # Remove rows with NaN values
        data = data.dropna()
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Sort by date
        data = data.sort_index()
        
        # Calculate additional metrics
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Volatility'] = data['Returns'].rolling(window=20).std()
        
        return data
    
    def _cache_data(self, symbol: str, data: pd.DataFrame):
        """Cache data in Redis and MongoDB"""
        try:
            # Cache in Redis for fast access
            if self.redis_client:
                cache_key = f"price_data:{symbol}"
                self.redis_client.setex(cache_key, 3600, data.to_json())  # 1 hour TTL
            
            # Store in MongoDB for persistence
            if self.mongo_client:
                db = self.mongo_client['trading_db']
                collection = db['price_data']
                
                # Convert to dict for MongoDB storage
                data_dict = data.reset_index().to_dict('records')
                
                # Update or insert
                collection.update_one(
                    {'symbol': symbol},
                    {'$set': {
                        'symbol': symbol,
                        'data': data_dict,
                        'last_updated': datetime.now()
                    }},
                    upsert=True
                )
                
        except Exception as e:
            logger.warning(f"Error caching data for {symbol}: {e}")
    
    async def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol"""
        try:
            if symbol in self.price_data and not self.price_data[symbol].empty:
                return self.price_data[symbol]['Close'].iloc[-1]
            
            # Try to fetch from cache
            if self.redis_client:
                cache_key = f"price_data:{symbol}"
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    data = pd.read_json(cached_data)
                    return data['Close'].iloc[-1]
            
            # Fetch fresh data
            data = await self.fetch_historical_data(symbol, days=1)
            if not data.empty:
                return data['Close'].iloc[-1]
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting latest price for {symbol}: {e}")
            return None
    
    async def get_price_history(self, symbol: str, days: int = None) -> pd.DataFrame:
        """Get price history for a symbol"""
        if days is None:
            days = self.history_days
        
        try:
            if symbol in self.price_data:
                data = self.price_data[symbol]
                if len(data) >= days:
                    return data.tail(days)
            
            # Fetch fresh data
            return await self.fetch_historical_data(symbol, days)
            
        except Exception as e:
            logger.error(f"Error getting price history for {symbol}: {e}")
            return pd.DataFrame()
    
    async def update_all_data(self):
        """Update data for all symbols"""
        logger.info("Starting data update for all symbols")
        
        tasks = []
        for symbol in self.symbols:
            task = asyncio.create_task(self.fetch_historical_data(symbol))
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error updating {self.symbols[i]}: {result}")
            else:
                logger.info(f"Successfully updated {self.symbols[i]}")
        
        logger.info("Data update completed")
    
    def get_correlation_matrix(self, symbols: List[str] = None) -> pd.DataFrame:
        """Calculate correlation matrix for symbols"""
        if symbols is None:
            symbols = self.symbols
        
        try:
            # Get returns data for all symbols
            returns_data = {}
            for symbol in symbols:
                if symbol in self.price_data and not self.price_data[symbol].empty:
                    returns_data[symbol] = self.price_data[symbol]['Returns'].dropna()
            
            if not returns_data:
                logger.warning("No returns data available for correlation calculation")
                return pd.DataFrame()
            
            # Create DataFrame and calculate correlations
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def get_volatility_data(self, symbol: str, window: int = 20) -> pd.Series:
        """Get volatility data for a symbol"""
        try:
            if symbol in self.price_data and not self.price_data[symbol].empty:
                data = self.price_data[symbol]
                if 'Volatility' in data.columns:
                    return data['Volatility']
                else:
                    # Calculate volatility if not present
                    returns = data['Returns'].dropna()
                    volatility = returns.rolling(window=window).std()
                    return volatility
            
            return pd.Series()
            
        except Exception as e:
            logger.error(f"Error getting volatility data for {symbol}: {e}")
            return pd.Series()
    
    def close_connections(self):
        """Close all database connections"""
        try:
            if self.redis_client:
                self.redis_client.close()
            
            if self.mongo_client:
                self.mongo_client.close()
            
            if self.postgres_engine:
                self.postgres_engine.dispose()
                
            logger.info("All database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing connections: {e}")


# Example usage and testing
if __name__ == "__main__":
    async def test_data_handler():
        """Test the data handler"""
        handler = DataHandler()
        
        try:
            # Test fetching data for a single symbol
            data = await handler.fetch_historical_data("AAPL", days=30)
            print(f"AAPL data shape: {data.shape}")
            print(f"Latest AAPL price: ${data['Close'].iloc[-1]:.2f}")
            
            # Test correlation matrix
            corr_matrix = handler.get_correlation_matrix(["AAPL", "MSFT", "GOOGL"])
            print("\nCorrelation Matrix:")
            print(corr_matrix)
            
            # Test volatility
            volatility = handler.get_volatility_data("AAPL")
            print(f"\nAAPL volatility (last 5 days):")
            print(volatility.tail())
            
        except Exception as e:
            print(f"Test failed: {e}")
        
        finally:
            handler.close_connections()
    
    # Run test
    asyncio.run(test_data_handler())
