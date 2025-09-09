"""
Data loader utility for market data and option specifications.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path


class DataLoader:
    """
    Utility class for loading and managing market data and option specifications.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.data_dir.mkdir(exist_ok=True)
    
    def create_sample_market_data(self, 
                                spot_price: float = 100.0,
                                n_options: int = 20) -> pd.DataFrame:
        """
        Create sample market data for testing and demonstration.
        
        Args:
            spot_price: Current spot price
            n_options: Number of options to generate
            
        Returns:
            DataFrame with option market data
        """
        # Generate option specifications
        strikes = np.linspace(spot_price * 0.8, spot_price * 1.2, n_options)
        maturities = np.array([0.25, 0.5, 1.0, 1.5, 2.0])  # Years
        
        data = []
        for strike in strikes:
            for maturity in maturities:
                # Generate realistic option prices using Black-Scholes approximation
                moneyness = strike / spot_price
                time_value = np.sqrt(maturity)
                
                # Simplified pricing logic
                if moneyness < 1.0:  # In-the-money call
                    intrinsic_value = spot_price - strike
                    time_value_factor = 0.1 * time_value
                    call_price = intrinsic_value + time_value_factor
                    put_price = time_value_factor
                elif moneyness > 1.0:  # Out-of-the-money call
                    time_value_factor = 0.15 * time_value
                    call_price = time_value_factor
                    put_price = strike - spot_price + time_value_factor
                else:  # At-the-money
                    time_value_factor = 0.12 * time_value
                    call_price = put_price = time_value_factor
                
                # Add some noise to make it realistic
                noise = np.random.normal(0, 0.01)
                call_price = max(0.01, call_price + noise)
                put_price = max(0.01, put_price + noise)
                
                data.append({
                    'strike': strike,
                    'maturity': maturity,
                    'call_price': call_price,
                    'put_price': put_price,
                    'moneyness': moneyness,
                    'time_to_maturity': maturity
                })
        
        return pd.DataFrame(data)
    
    def create_sample_option_specs(self, 
                                 spot_price: float = 100.0,
                                 n_options: int = 10) -> List[Dict[str, Any]]:
        """
        Create sample option specifications for testing.
        
        Args:
            spot_price: Current spot price
            n_options: Number of options to generate
            
        Returns:
            List of option specifications
        """
        from ..pricing.payoff_functions import EuropeanPayoff, AsianPayoff, BarrierPayoff
        
        option_specs = []
        
        # European options
        strikes = np.linspace(spot_price * 0.9, spot_price * 1.1, n_options // 2)
        for i, strike in enumerate(strikes):
            maturity = 0.5 + (i % 3) * 0.5  # 0.5, 1.0, or 1.5 years
            
            # Call option
            call_payoff = EuropeanPayoff(strike, "call")
            option_specs.append({
                'option_id': f'EUR_CALL_{i}',
                'option_type': 'call',
                'strike': strike,
                'maturity': maturity,
                'spot': spot_price,
                'payoff_function': call_payoff,
                'market_price': self._estimate_market_price(spot_price, strike, maturity, 'call')
            })
            
            # Put option
            put_payoff = EuropeanPayoff(strike, "put")
            option_specs.append({
                'option_id': f'EUR_PUT_{i}',
                'option_type': 'put',
                'strike': strike,
                'maturity': maturity,
                'spot': spot_price,
                'payoff_function': put_payoff,
                'market_price': self._estimate_market_price(spot_price, strike, maturity, 'put')
            })
        
        # Asian options
        asian_strikes = [spot_price * 0.95, spot_price, spot_price * 1.05]
        for i, strike in enumerate(asian_strikes):
            maturity = 1.0
            asian_payoff = AsianPayoff(strike, "call", "arithmetic")
            option_specs.append({
                'option_id': f'ASIAN_CALL_{i}',
                'option_type': 'call',
                'strike': strike,
                'maturity': maturity,
                'spot': spot_price,
                'payoff_function': asian_payoff,
                'market_price': self._estimate_market_price(spot_price, strike, maturity, 'call') * 0.8
            })
        
        # Barrier options
        barrier_levels = [spot_price * 0.8, spot_price * 1.2]
        for i, barrier in enumerate(barrier_levels):
            maturity = 1.0
            barrier_payoff = BarrierPayoff(spot_price, barrier, "down-and-out", "call")
            option_specs.append({
                'option_id': f'BARRIER_CALL_{i}',
                'option_type': 'call',
                'strike': spot_price,
                'maturity': maturity,
                'spot': spot_price,
                'barrier': barrier,
                'payoff_function': barrier_payoff,
                'market_price': self._estimate_market_price(spot_price, spot_price, maturity, 'call') * 0.6
            })
        
        return option_specs
    
    def _estimate_market_price(self, 
                              spot: float, 
                              strike: float, 
                              maturity: float, 
                              option_type: str) -> float:
        """
        Estimate market price using simplified Black-Scholes approximation.
        
        Args:
            spot: Current spot price
            strike: Option strike price
            maturity: Time to maturity
            option_type: "call" or "put"
            
        Returns:
            Estimated market price
        """
        # Simplified Black-Scholes approximation
        moneyness = strike / spot
        time_value = np.sqrt(maturity)
        
        if option_type.lower() == "call":
            if moneyness < 1.0:  # In-the-money
                intrinsic_value = spot - strike
                time_value_factor = 0.1 * time_value
                return intrinsic_value + time_value_factor
            else:  # Out-of-the-money
                time_value_factor = 0.15 * time_value
                return time_value_factor
        else:  # put
            if moneyness > 1.0:  # In-the-money
                intrinsic_value = strike - spot
                time_value_factor = 0.1 * time_value
                return intrinsic_value + time_value_factor
            else:  # Out-of-the-money
                time_value_factor = 0.15 * time_value
                return time_value_factor
    
    def save_market_data(self, 
                        data: pd.DataFrame, 
                        filename: str = "market_data.csv") -> None:
        """
        Save market data to CSV file.
        
        Args:
            data: Market data DataFrame
            filename: Output filename
        """
        filepath = self.data_dir / filename
        data.to_csv(filepath, index=False)
        print(f"Market data saved to {filepath}")
    
    def load_market_data(self, 
                        filename: str = "market_data.csv") -> pd.DataFrame:
        """
        Load market data from CSV file.
        
        Args:
            filename: Input filename
            
        Returns:
            Market data DataFrame
        """
        filepath = self.data_dir / filename
        if filepath.exists():
            return pd.read_csv(filepath)
        else:
            print(f"File {filepath} not found. Creating sample data.")
            return self.create_sample_market_data()
    
    def create_calibration_dataset(self, 
                                 spot_price: float = 100.0,
                                 n_strikes: int = 10,
                                 n_maturities: int = 5) -> Dict[str, Any]:
        """
        Create a dataset for model calibration.
        
        Args:
            spot_price: Current spot price
            n_strikes: Number of strike prices
            n_maturities: Number of maturities
            
        Returns:
            Dictionary containing calibration data
        """
        # Generate strikes and maturities
        strikes = np.linspace(spot_price * 0.8, spot_price * 1.2, n_strikes)
        maturities = np.linspace(0.25, 2.0, n_maturities)
        
        # Create option specifications
        option_specs = []
        market_prices = []
        
        for strike in strikes:
            for maturity in maturities:
                # Call option
                call_price = self._estimate_market_price(spot_price, strike, maturity, 'call')
                option_specs.append({
                    'spot': spot_price,
                    'strike': strike,
                    'maturity': maturity,
                    'option_type': 'call'
                })
                market_prices.append(call_price)
                
                # Put option
                put_price = self._estimate_market_price(spot_price, strike, maturity, 'put')
                option_specs.append({
                    'spot': spot_price,
                    'strike': strike,
                    'maturity': maturity,
                    'option_type': 'put'
                })
                market_prices.append(put_price)
        
        return {
            'option_specs': option_specs,
            'market_prices': np.array(market_prices),
            'spot_price': spot_price,
            'n_options': len(option_specs)
        }
    
    def create_portfolio_spec(self, 
                            spot_price: float = 100.0,
                            initial_capital: float = 1000000.0) -> Dict[str, Any]:
        """
        Create a sample portfolio specification for stress testing.
        
        Args:
            spot_price: Current spot price
            initial_capital: Initial portfolio capital
            
        Returns:
            Portfolio specification
        """
        from ..pricing.payoff_functions import EuropeanPayoff
        
        # Create a simple portfolio with one European call option
        option_spec = {
            'spot': spot_price,
            'strike': spot_price,  # At-the-money
            'maturity': 1.0,       # 1 year
            'option_type': 'call',
            'payoff_function': EuropeanPayoff(spot_price, 'call')
        }
        
        return {
            'option_spec': option_spec,
            'n_options': 100,      # 100 options
            'initial_capital': initial_capital,
            'transaction_cost': 0.001,  # 0.1%
            'hedge_duration': 1.0,     # 1 year
            'base_spot': spot_price,
            'base_volatility': 0.2,
            'base_interest_rate': 0.05
        }

