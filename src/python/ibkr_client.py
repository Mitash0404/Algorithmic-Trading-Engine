"""
Interactive Brokers API Client Module for Algorithmic Trading Engine
Handles live trading execution through IBKR API
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import Order
from ibapi.common import OrderId, TickerId
from ibapi.order_state import OrderState
from ibapi.execution import Execution
from ibapi.account_summary_tags import AccountSummaryTags
import pandas as pd
import numpy as np
from loguru import logger
import yaml
from dataclasses import dataclass
from enum import Enum

class OrderType(Enum):
    """Order types"""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STP"
    STOP_LIMIT = "STP LMT"
    TRAILING_STOP = "TRAIL"

class OrderAction(Enum):
    """Order actions"""
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class TradeSignal:
    """Trade signal structure"""
    symbol: str
    action: str
    quantity: int
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    order_id: Optional[int] = None
    status: str = "PENDING"

# Simplified position class since the original is not available
class Position:
    def __init__(self, symbol: str, position: float, avg_cost: float):
        self.symbol = symbol
        self.position = position
        self.avg_cost = avg_cost

class IBKRClient(EWrapper, EClient):
    """Interactive Brokers API client for live trading"""
    
    def __init__(self, config: dict):
        """Initialize IBKR client"""
        EClient.__init__(self, self)
        
        # Configuration
        self.config = config
        self.host = config['ibkr']['host']
        self.port = config['ibkr']['port']
        self.client_id = config['ibkr']['client_id']
        self.timeout = config['ibkr']['timeout']
        self.retry_attempts = config['ibkr']['retry_attempts']
        
        # Connection state
        self.connected = False
        self.next_order_id = None
        self.connection_thread = None
        
        # Data storage
        self.positions = {}
        self.orders = {}
        self.executions = {}
        self.account_info = {}
        self.market_data = {}
        
        # Callbacks
        self.on_order_status = None
        self.on_execution = None
        self.on_position = None
        self.on_account_summary = None
        
        # Performance tracking
        self.order_latency = []
        self.fill_rate = 0.0
        self.total_orders = 0
        self.filled_orders = 0
        
        logger.info("IBKRClient initialized successfully")
    
    def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway"""
        try:
            logger.info(f"Connecting to IBKR at {self.host}:{self.port}")
            
            # Start connection in separate thread
            self.connection_thread = threading.Thread(target=self._run_connection)
            self.connection_thread.daemon = True
            self.connection_thread.start()
            
            # Wait for connection
            timeout_counter = 0
            while not self.connected and timeout_counter < self.timeout:
                time.sleep(0.1)
                timeout_counter += 1
            
            if self.connected:
                logger.info("Successfully connected to IBKR")
                return True
            else:
                logger.error("Failed to connect to IBKR")
                return False
                
        except Exception as e:
            logger.error(f"Error connecting to IBKR: {e}")
            return False
    
    def _run_connection(self):
        """Run connection in separate thread"""
        try:
            self.connect(self.host, self.port, self.client_id)
            self.run()
        except Exception as e:
            logger.error(f"Connection error: {e}")
    
    def disconnect(self):
        """Disconnect from IBKR"""
        try:
            if self.connected:
                self.disconnect()
                self.connected = False
                logger.info("Disconnected from IBKR")
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to IBKR"""
        return self.connected
    
    # EWrapper callbacks
    def connectAck(self):
        """Called when connection is acknowledged"""
        logger.info("IBKR connection acknowledged")
    
    def nextValidId(self, orderId: OrderId):
        """Called when next valid order ID is received"""
        self.next_order_id = orderId
        logger.info(f"Next valid order ID: {orderId}")
    
    def connectionClosed(self):
        """Called when connection is closed"""
        self.connected = False
        logger.warning("IBKR connection closed")
    
    def error(self, reqId: TickerId, errorCode: int, errorString: str):
        """Called when an error occurs"""
        logger.error(f"IBKR Error {errorCode}: {errorString} (reqId: {reqId})")
    
    def orderStatus(self, orderId: OrderId, status: str, filled: float,
                   remaining: float, avgFillPrice: float, permId: int,
                   parentId: int, lastFillPrice: float, clientId: int,
                   whyHeld: str, mktCapPrice: float):
        """Called when order status changes"""
        try:
            order_status = {
                'order_id': orderId,
                'status': status,
                'filled': filled,
                'remaining': remaining,
                'avg_fill_price': avgFillPrice,
                'last_fill_price': lastFillPrice,
                'timestamp': datetime.now()
            }
            
            self.orders[orderId] = order_status
            
            # Update fill rate
            if status in ['Filled', 'PartiallyFilled']:
                self.filled_orders += 1
            self.total_orders += 1
            self.fill_rate = self.filled_orders / max(self.total_orders, 1)
            
            # Call callback if set
            if self.on_order_status:
                self.on_order_status(order_status)
            
            logger.info(f"Order {orderId} status: {status}")
            
        except Exception as e:
            logger.error(f"Error processing order status: {e}")
    
    def execDetails(self, reqId: TickerId, contract: Contract, execution: Execution):
        """Called when execution details are received"""
        try:
            exec_details = {
                'execution_id': execution.execId,
                'order_id': execution.orderId,
                'symbol': contract.symbol,
                'exchange': contract.exchange,
                'quantity': execution.shares,
                'price': execution.price,
                'timestamp': datetime.now()
            }
            
            self.executions[execution.execId] = exec_details
            
            # Call callback if set
            if self.on_execution:
                self.on_execution(exec_details)
            
            logger.info(f"Execution: {exec_details}")
            
        except Exception as e:
            logger.error(f"Error processing execution: {e}")
    
    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """Called when position information is received"""
        try:
            position_info = {
                'account': account,
                'symbol': contract.symbol,
                'contract_id': contract.conId,
                'position': position,
                'avg_cost': avgCost,
                'timestamp': datetime.now()
            }
            
            self.positions[contract.symbol] = position_info
            
            # Call callback if set
            if self.on_position:
                self.on_position(position_info)
            
        except Exception as e:
            logger.error(f"Error processing position: {e}")
    
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        """Called when account summary information is received"""
        try:
            if account not in self.account_info:
                self.account_info[account] = {}
            
            self.account_info[account][tag] = {
                'value': value,
                'currency': currency,
                'timestamp': datetime.now()
            }
            
            # Call callback if set
            if self.on_account_summary:
                self.on_account_summary(account, tag, value, currency)
            
        except Exception as e:
            logger.error(f"Error processing account summary: {e}")
    
    def tickPrice(self, reqId: TickerId, tickType: int, price: float, attrib):
        """Called when tick price is received"""
        try:
            if reqId not in self.market_data:
                self.market_data[reqId] = {}
            
            tick_types = {1: 'bid', 2: 'ask', 4: 'last', 6: 'high', 7: 'low', 9: 'close'}
            tick_name = tick_types.get(tickType, f'type_{tickType}')
            
            self.market_data[reqId][tick_name] = {
                'price': price,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error processing tick price: {e}")
    
    # Trading methods
    def place_order(self, signal: TradeSignal) -> Optional[int]:
        """Place a trading order"""
        try:
            if not self.connected:
                logger.error("Not connected to IBKR")
                return None
            
            if self.next_order_id is None:
                logger.error("No valid order ID available")
                return None
            
            # Create contract
            contract = self._create_contract(signal.symbol)
            if not contract:
                return None
            
            # Create order
            order = self._create_order(signal)
            if not order:
                return None
            
            # Place order
            order_id = self.next_order_id
            self.placeOrder(order_id, contract, order)
            
            # Update order ID
            self.next_order_id += 1
            
            # Track order
            self.orders[order_id] = {
                'order_id': order_id,
                'symbol': signal.symbol,
                'action': signal.action,
                'quantity': signal.quantity,
                'status': 'SUBMITTED',
                'timestamp': datetime.now()
            }
            
            logger.info(f"Order placed: {signal.symbol} {signal.action} {signal.quantity} shares")
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def _create_contract(self, symbol: str) -> Optional[Contract]:
        """Create IBKR contract object"""
        try:
            contract = Contract()
            contract.symbol = symbol
            contract.secType = "STK"
            contract.exchange = "SMART"
            contract.currency = "USD"
            return contract
        except Exception as e:
            logger.error(f"Error creating contract for {symbol}: {e}")
            return None
    
    def _create_order(self, signal: TradeSignal) -> Optional[Order]:
        """Create IBKR order object"""
        try:
            order = Order()
            order.action = signal.action
            order.totalQuantity = signal.quantity
            order.orderType = signal.order_type.value
            order.tif = signal.time_in_force
            
            # Set limit price if specified
            if signal.limit_price and signal.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
                order.lmtPrice = signal.limit_price
            
            # Set stop price if specified
            if signal.stop_price and signal.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                order.auxPrice = signal.stop_price
            
            return order
            
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return None
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancel an existing order"""
        try:
            if not self.connected:
                logger.error("Not connected to IBKR")
                return False
            
            self.cancelOrder(order_id)
            logger.info(f"Order {order_id} cancellation requested")
            return True
            
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return False
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions"""
        try:
            if not self.connected:
                logger.error("Not connected to IBKR")
                return {}
            
            # Request positions
            self.reqPositions()
            
            # Wait for positions to be received
            time.sleep(0.5)
            
            return self.positions.copy()
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_account_summary(self, account: str = "DU123456") -> Dict[str, Dict]:
        """Get account summary information"""
        try:
            if not self.connected:
                logger.error("Not connected to IBKR")
                return {}
            
            # Request account summary
            self.reqAccountSummary(1, account, AccountSummaryTags.AllTags)
            
            # Wait for account info to be received
            time.sleep(0.5)
            
            return self.account_info.get(account, {})
            
        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {}
    
    def get_market_data(self, symbol: str, req_id: int = None) -> Dict[str, Any]:
        """Get market data for a symbol"""
        try:
            if not self.connected:
                logger.error("Not connected to IBKR")
                return {}
            
            if req_id is None:
                req_id = hash(symbol) % 10000
            
            # Create contract
            contract = self._create_contract(symbol)
            if not contract:
                return {}
            
            # Request market data
            self.reqMktData(req_id, contract, "", False, False, [])
            
            # Wait for data to be received
            time.sleep(0.1)
            
            return self.market_data.get(req_id, {})
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return {}
    
    def get_order_status(self, order_id: int) -> Optional[Dict]:
        """Get status of a specific order"""
        return self.orders.get(order_id)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get trading performance metrics"""
        try:
            metrics = {
                'fill_rate': self.fill_rate,
                'total_orders': self.total_orders,
                'filled_orders': self.filled_orders,
                'avg_order_latency': np.mean(self.order_latency) if self.order_latency else 0,
                'connected': self.connected
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def set_callbacks(self, on_order_status: Callable = None, on_execution: Callable = None,
                     on_position: Callable = None, on_account_summary: Callable = None):
        """Set callback functions for various events"""
        self.on_order_status = on_order_status
        self.on_execution = on_execution
        self.on_position = on_position
        self.on_account_summary = on_account_summary
    
    def execute_trade_signal(self, signal: TradeSignal) -> Optional[int]:
        """Execute a trade signal with latency measurement"""
        try:
            start_time = time.time()
            
            # Place order
            order_id = self.place_order(signal)
            
            if order_id:
                # Measure latency
                latency = (time.time() - start_time) * 1000  # Convert to milliseconds
                self.order_latency.append(latency)
                
                # Keep only recent latency measurements
                if len(self.order_latency) > 1000:
                    self.order_latency = self.order_latency[-1000:]
                
                logger.info(f"Trade signal executed in {latency:.2f} ms")
                return order_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing trade signal: {e}")
            return None


# Example usage and testing
if __name__ == "__main__":
    # Test the IBKR client
    config = {
        'ibkr': {
            'host': '127.0.0.1',
            'port': 7497,
            'client_id': 1,
            'timeout': 20,
            'retry_attempts': 3
        }
    }
    
    # Create client
    client = IBKRClient(config)
    
    # Set callbacks
    def on_order_status(status):
        print(f"Order status: {status}")
    
    def on_execution(execution):
        print(f"Execution: {execution}")
    
    client.set_callbacks(on_order_status=on_order_status, on_execution=on_execution)
    
    try:
        # Test TradeSignal creation
        signal = TradeSignal(
            symbol="AAPL",
            action="BUY",
            quantity=100,
            order_type=OrderType.MARKET
        )
        print(f"Created trade signal: {signal}")
        
        # Test enums
        print(f"Order types: {[ot.value for ot in OrderType]}")
        print(f"Order actions: {[oa.value for oa in OrderAction]}")
        
        print("IBKR client test completed successfully")
        
    except Exception as e:
        print(f"Test failed: {e}")
    
    print("IBKR client ready for use")
