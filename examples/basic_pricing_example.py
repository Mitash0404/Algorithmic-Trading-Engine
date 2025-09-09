"""
Basic option pricing example demonstrating the Monte Carlo engine.
"""

import numpy as np
import time
from src.models.heston_model import HestonModel
from src.models.sabr_model import SABRModel
from src.pricing.monte_carlo_engine import MonteCarloEngine
from src.pricing.payoff_functions import EuropeanPayoff, AsianPayoff, BarrierPayoff
from src.utils.data_loader import DataLoader


def main():
    """Run basic pricing examples."""
    print("=== Exotic Option Monte Carlo Pricer ===")
    print("Basic Pricing Example\n")
    
    # Initialize models
    print("1. Initializing Stochastic Volatility Models...")
    
    # Heston model
    heston_model = HestonModel(
        v0=0.04,      # Initial variance
        kappa=2.0,    # Mean reversion speed
        theta=0.04,   # Long-term variance
        rho=-0.7,     # Correlation
        sigma=0.5,    # Volatility of volatility
        risk_free_rate=0.05
    )
    print(f"   Heston Model: {heston_model}")
    
    # SABR model
    sabr_model = SABRModel(
        alpha=0.2,    # Initial volatility
        beta=0.5,     # CEV parameter
        rho=-0.1,     # Correlation
        nu=0.5,       # Volatility of volatility
        risk_free_rate=0.05
    )
    print(f"   SABR Model: {sabr_model}")
    
    # Initialize Monte Carlo engine
    print("\n2. Setting up Monte Carlo Engine...")
    mc_engine = MonteCarloEngine(
        model=heston_model,
        n_paths=100000,
        n_steps=252,
        use_antithetic=True,
        use_control_variates=True,
        seed=42
    )
    print(f"   Monte Carlo Engine: {mc_engine}")
    
    # Market parameters
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    
    print(f"\n3. Pricing Options (Spot: {spot}, Strike: {strike}, Maturity: {maturity} years)")
    print("   " + "="*60)
    
    # Price European call option
    print("\n   European Call Option:")
    start_time = time.time()
    european_call_price = mc_engine.price_european_call(spot, strike, maturity)
    pricing_time = time.time() - start_time
    
    print(f"     Price: ${european_call_price:.4f}")
    print(f"     Pricing Time: {pricing_time:.3f} seconds")
    
    # Price European put option
    print("\n   European Put Option:")
    start_time = time.time()
    european_put_price = mc_engine.price_european_put(spot, strike, maturity)
    pricing_time = time.time() - start_time
    
    print(f"     Price: ${european_put_price:.4f}")
    print(f"     Pricing Time: {pricing_time:.3f} seconds")
    
    # Price Asian option
    print("\n   Asian Call Option (Arithmetic Average):")
    start_time = time.time()
    asian_call_price = mc_engine.price_asian_option(
        spot, strike, maturity, "call", "arithmetic"
    )
    pricing_time = time.time() - start_time
    
    print(f"     Price: ${asian_call_price:.4f}")
    print(f"     Pricing Time: {pricing_time:.3f} seconds")
    
    # Price Barrier option
    barrier = 90.0
    print(f"\n   Barrier Call Option (Down-and-Out, Barrier: {barrier}):")
    start_time = time.time()
    barrier_call_price = mc_engine.price_barrier_option(
        spot, strike, maturity, barrier, "down-and-out", "call"
    )
    pricing_time = time.time() - start_time
    
    print(f"     Price: ${barrier_call_price:.4f}")
    print(f"     Pricing Time: {pricing_time:.3f} seconds")
    
    # Compare with analytical Heston prices
    print("\n4. Comparing with Analytical Heston Prices...")
    print("   " + "="*60)
    
    try:
        heston_analytical_call = heston_model.get_analytical_price(spot, strike, maturity, "call")
        heston_analytical_put = heston_model.get_analytical_price(spot, strike, maturity, "put")
        
        print(f"\n   European Call:")
        print(f"     Monte Carlo: ${european_call_price:.4f}")
        print(f"     Analytical:  ${heston_analytical_call:.4f}")
        print(f"     Difference:  ${abs(european_call_price - heston_analytical_call):.4f}")
        print(f"     Accuracy:    {100 * (1 - abs(european_call_price - heston_analytical_call) / heston_analytical_call):.2f}%")
        
        print(f"\n   European Put:")
        print(f"     Monte Carlo: ${european_put_price:.4f}")
        print(f"     Analytical:  ${heston_analytical_put:.4f}")
        print(f"     Difference:  ${abs(european_put_price - heston_analytical_put):.4f}")
        print(f"     Accuracy:    {100 * (1 - abs(european_put_price - heston_analytical_put) / heston_analytical_put):.2f}%")
        
    except Exception as e:
        print(f"   Analytical pricing failed: {e}")
    
    # Performance statistics
    print("\n5. Performance Statistics...")
    print("   " + "="*60)
    
    perf_stats = mc_engine.get_performance_stats()
    if perf_stats:
        print(f"   Total Pricings: {perf_stats['total_pricings']}")
        print(f"   Average Time:   {perf_stats['average_pricing_time']:.3f} seconds")
        print(f"   Total Time:     {perf_stats['total_time']:.3f} seconds")
        print(f"   Fastest:        {perf_stats['fastest_pricing']:.3f} seconds")
        print(f"   Slowest:        {perf_stats['slowest_pricing']:.3f} seconds")
    
    # Convergence analysis
    print("\n6. Convergence Analysis...")
    print("   " + "="*60)
    
    convergence_metrics = mc_engine._get_convergence_metrics()
    if convergence_metrics:
        print(f"   Convergence Rate: {convergence_metrics.get('convergence_rate', 'N/A'):.4f}")
        print(f"   Final Price:      ${convergence_metrics.get('final_price', 0):.4f}")
        print(f"   Number of Paths:  {convergence_metrics.get('n_paths', 0):,}")
    
    print("\n=== Example Completed Successfully! ===")


if __name__ == "__main__":
    main()

