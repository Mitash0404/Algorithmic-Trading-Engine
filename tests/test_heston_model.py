"""
Tests for the Heston stochastic volatility model.
"""

import pytest
import numpy as np
from src.models.heston_model import HestonModel


class TestHestonModel:
    """Test cases for the Heston model."""
    
    def test_initialization(self):
        """Test Heston model initialization."""
        model = HestonModel(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            rho=-0.7,
            sigma=0.5,
            risk_free_rate=0.05
        )
        
        assert model.v0 == 0.04
        assert model.kappa == 2.0
        assert model.theta == 0.04
        assert model.rho == -0.7
        assert model.sigma == 0.5
        assert model.risk_free_rate == 0.05
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Valid parameters
        model = HestonModel()
        assert model.validate_parameters() is True
        
        # Invalid parameters - Feller condition violated
        with pytest.raises(ValueError):
            HestonModel(v0=0.01, kappa=1.0, theta=0.01, sigma=0.5)
        
        # Invalid parameters - negative values
        with pytest.raises(ValueError):
            HestonModel(v0=-0.01, kappa=2.0, theta=0.04, sigma=0.5)
        
        with pytest.raises(ValueError):
            HestonModel(v0=0.04, kappa=-2.0, theta=0.04, sigma=0.5)
        
        with pytest.raises(ValueError):
            HestonModel(v0=0.04, kappa=2.0, theta=-0.04, sigma=0.5)
        
        with pytest.raises(ValueError):
            HestonModel(v0=0.04, kappa=2.0, theta=0.04, sigma=-0.5)
        
        # Invalid correlation
        with pytest.raises(ValueError):
            HestonModel(v0=0.04, kappa=2.0, theta=0.04, rho=1.5, sigma=0.5)
    
    def test_get_parameters(self):
        """Test getting model parameters."""
        model = HestonModel(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            rho=-0.7,
            sigma=0.5,
            risk_free_rate=0.05
        )
        
        params = model.get_parameters()
        expected_params = {
            'v0': 0.04,
            'kappa': 2.0,
            'theta': 0.04,
            'rho': -0.7,
            'sigma': 0.5,
            'risk_free_rate': 0.05
        }
        
        assert params == expected_params
    
    def test_set_parameters(self):
        """Test setting model parameters."""
        model = HestonModel()
        
        # Set new parameters
        model.set_parameters(v0=0.06, kappa=3.0, theta=0.06)
        
        assert model.v0 == 0.06
        assert model.kappa == 3.0
        assert model.theta == 0.06
        
        # Test invalid parameter
        with pytest.raises(ValueError):
            model.set_parameters(invalid_param=0.1)
    
    def test_generate_paths(self):
        """Test Monte Carlo path generation."""
        model = HestonModel(seed=42)
        
        spot = 100.0
        maturity = 1.0
        n_paths = 1000
        n_steps = 252
        
        asset_paths, volatility_paths = model.generate_paths(
            spot, maturity, n_paths, n_steps
        )
        
        # Check shapes
        assert asset_paths.shape == (n_paths, n_steps + 1)
        assert volatility_paths.shape == (n_paths, n_steps + 1)
        
        # Check initial values
        assert np.allclose(asset_paths[:, 0], spot)
        assert np.allclose(volatility_paths[:, 0], model.v0)
        
        # Check that asset prices are positive
        assert np.all(asset_paths > 0)
        
        # Check that volatilities are positive
        assert np.all(volatility_paths > 0)
        
        # Check that paths are reasonable
        assert np.all(asset_paths < 1000)  # Not exploding
        assert np.all(volatility_paths < 1.0)  # Volatility < 100%
    
    def test_path_generation_reproducibility(self):
        """Test that path generation is reproducible with the same seed."""
        model = HestonModel()
        
        spot = 100.0
        maturity = 1.0
        n_paths = 100
        n_steps = 50
        
        # Generate paths with same seed
        paths1, vol1 = model.generate_paths(spot, maturity, n_paths, n_steps, seed=42)
        paths2, vol2 = model.generate_paths(spot, maturity, n_paths, n_steps, seed=42)
        
        # Should be identical
        assert np.allclose(paths1, paths2)
        assert np.allclose(vol1, vol2)
        
        # Different seeds should give different results
        paths3, vol3 = model.generate_paths(spot, maturity, n_paths, n_steps, seed=43)
        
        # Should be different (with high probability)
        assert not np.allclose(paths1, paths3)
    
    def test_analytical_pricing(self):
        """Test analytical option pricing."""
        model = HestonModel(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            rho=-0.7,
            sigma=0.5,
            risk_free_rate=0.05
        )
        
        spot = 100.0
        strike = 100.0
        maturity = 1.0
        
        # Test call option
        call_price = model.get_analytical_price(spot, strike, maturity, "call")
        assert call_price > 0
        assert call_price < spot  # Call price should be less than spot
        
        # Test put option
        put_price = model.get_analytical_price(spot, strike, maturity, "put")
        assert put_price > 0
        assert put_price < strike  # Put price should be less than strike
        
        # Test invalid option type
        with pytest.raises(ValueError):
            model.get_analytical_price(spot, strike, maturity, "invalid")
    
    def test_implied_volatility_surface(self):
        """Test implied volatility surface calculation."""
        model = HestonModel(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            rho=-0.7,
            sigma=0.5,
            risk_free_rate=0.05
        )
        
        spot = 100.0
        strikes = np.array([90, 100, 110])
        maturities = np.array([0.5, 1.0])
        
        iv_surface = model.get_implied_volatility_surface(spot, strikes, maturities, "call")
        
        # Check shape
        assert iv_surface.shape == (len(strikes), len(maturities))
        
        # Check that implied volatilities are positive
        assert np.all(iv_surface > 0)
        
        # Check that implied volatilities are reasonable (< 200%)
        assert np.all(iv_surface < 2.0)
    
    def test_string_representation(self):
        """Test string representation of the model."""
        model = HestonModel(
            v0=0.04,
            kappa=2.0,
            theta=0.04,
            rho=-0.7,
            sigma=0.5,
            risk_free_rate=0.05
        )
        
        str_repr = str(model)
        assert "HestonModel" in str_repr
        assert "v0=0.0400" in str_repr
        assert "kappa=2.0000" in str_repr
    
    def test_risk_free_rate_override(self):
        """Test that risk-free rate can be overridden."""
        model = HestonModel(risk_free_rate=0.03)
        
        # Override risk-free rate
        model.set_parameters(risk_free_rate=0.07)
        assert model.risk_free_rate == 0.07
        
        # Test in path generation
        spot = 100.0
        maturity = 1.0
        n_paths = 100
        n_steps = 50
        
        asset_paths, _ = model.generate_paths(spot, maturity, n_paths, n_steps, seed=42)
        
        # Higher risk-free rate should lead to higher expected asset prices
        expected_return = 0.07 * maturity
        actual_return = np.mean(np.log(asset_paths[:, -1] / asset_paths[:, 0]))
        
        # Allow for some variance in Monte Carlo simulation
        assert abs(actual_return - expected_return) < 0.1


if __name__ == "__main__":
    pytest.main([__file__])

