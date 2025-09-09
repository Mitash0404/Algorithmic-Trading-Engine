/*
Fast Operations C++ Extension for Algorithmic Trading Engine
Provides high-performance implementations for latency-critical operations
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <unordered_map>

namespace py = pybind11;

// High-performance cointegration calculation
class FastCointegration {
private:
    std::vector<double> buffer1;
    std::vector<double> buffer2;
    std::vector<double> spread_buffer;
    std::vector<double> returns_buffer;
    
public:
    FastCointegration(size_t max_size = 10000) {
        buffer1.reserve(max_size);
        buffer2.reserve(max_size);
        spread_buffer.reserve(max_size);
        returns_buffer.reserve(max_size);
    }
    
    // Fast Engle-Granger cointegration test
    std::tuple<bool, double, double, double> test_cointegration(
        const std::vector<double>& prices1,
        const std::vector<double>& prices2,
        double significance_level = 0.05
    ) {
        if (prices1.size() != prices2.size() || prices1.size() < 50) {
            return {false, 1.0, 0.0, 0.0};
        }
        
        // Calculate spread using OLS
        double beta = calculate_ols_beta(prices1, prices2);
        double alpha = calculate_ols_alpha(prices1, prices2, beta);
        
        // Calculate spread
        spread_buffer.clear();
        for (size_t i = 0; i < prices1.size(); ++i) {
            spread_buffer.push_back(prices1[i] - alpha - beta * prices2[i]);
        }
        
        // Test for stationarity using ADF
        double adf_stat = calculate_adf_statistic(spread_buffer);
        double p_value = calculate_adf_pvalue(adf_stat, prices1.size());
        
        // Calculate half-life
        double half_life = calculate_half_life(spread_buffer);
        
        bool is_cointegrated = p_value < significance_level && half_life > 0;
        
        return {is_cointegrated, p_value, beta, half_life};
    }
    
    // Fast z-score calculation with rolling statistics
    double calculate_zscore(
        const std::vector<double>& spread,
        size_t window = 20
    ) {
        if (spread.size() < window) return 0.0;
        
        size_t n = spread.size();
        double current_spread = spread[n - 1];
        
        // Calculate rolling mean and std efficiently
        double sum = 0.0;
        double sum_sq = 0.0;
        
        for (size_t i = n - window; i < n; ++i) {
            sum += spread[i];
            sum_sq += spread[i] * spread[i];
        }
        
        double mean = sum / window;
        double variance = (sum_sq / window) - (mean * mean);
        double std_dev = std::sqrt(std::max(variance, 0.0));
        
        if (std_dev < 1e-10) return 0.0;
        
        return (current_spread - mean) / std_dev;
    }
    
    // Fast correlation matrix calculation
    py::array_t<double> calculate_correlation_matrix(
        const std::vector<std::vector<double>>& returns_data
    ) {
        size_t n_assets = returns_data.size();
        if (n_assets == 0) return py::array_t<double>();
        
        size_t n_periods = returns_data[0].size();
        if (n_periods < 2) return py::array_t<double>();
        
        // Create output array
        auto result = py::array_t<double>({n_assets, n_assets});
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);
        
        // Calculate correlations
        for (size_t i = 0; i < n_assets; ++i) {
            for (size_t j = 0; j < n_assets; ++j) {
                if (i == j) {
                    ptr[i * n_assets + j] = 1.0;
                } else {
                    ptr[i * n_assets + j] = calculate_correlation(
                        returns_data[i], returns_data[j]
                    );
                }
            }
        }
        
        return result;
    }
    
    // Fast volatility calculation
    std::vector<double> calculate_volatility(
        const std::vector<double>& returns,
        size_t window = 20
    ) {
        std::vector<double> volatility;
        volatility.reserve(returns.size());
        
        for (size_t i = 0; i < returns.size(); ++i) {
            if (i < window - 1) {
                volatility.push_back(0.0);
                continue;
            }
            
            double sum = 0.0;
            for (size_t j = i - window + 1; j <= i; ++j) {
                sum += returns[j] * returns[j];
            }
            
            volatility.push_back(std::sqrt(sum / window));
        }
        
        return volatility;
    }
    
    // Fast Kelly criterion calculation
    double calculate_kelly_fraction(
        const std::vector<double>& returns,
        double risk_free_rate = 0.02
    ) {
        if (returns.size() < 30) return 0.0;
        
        double mean_return = 0.0;
        double variance = 0.0;
        
        for (double ret : returns) {
            mean_return += ret;
            variance += ret * ret;
        }
        
        mean_return /= returns.size();
        variance = (variance / returns.size()) - (mean_return * mean_return);
        
        if (variance < 1e-10) return 0.0;
        
        double excess_return = mean_return - risk_free_rate;
        return excess_return / variance;
    }
    
private:
    // Fast OLS beta calculation
    double calculate_ols_beta(
        const std::vector<double>& y,
        const std::vector<double>& x
    ) {
        size_t n = y.size();
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
        
        for (size_t i = 0; i < n; ++i) {
            sum_x += x[i];
            sum_y += y[i];
            sum_xy += x[i] * y[i];
            sum_x2 += x[i] * x[i];
        }
        
        double numerator = n * sum_xy - sum_x * sum_y;
        double denominator = n * sum_x2 - sum_x * sum_x;
        
        return (denominator != 0) ? numerator / denominator : 0.0;
    }
    
    // Fast OLS alpha calculation
    double calculate_ols_alpha(
        const std::vector<double>& y,
        const std::vector<double>& x,
        double beta
    ) {
        size_t n = y.size();
        double sum_y = 0.0, sum_x = 0.0;
        
        for (size_t i = 0; i < n; ++i) {
            sum_y += y[i];
            sum_x += x[i];
        }
        
        return (sum_y - beta * sum_x) / n;
    }
    
    // Fast ADF statistic calculation
    double calculate_adf_statistic(const std::vector<double>& spread) {
        if (spread.size() < 3) return 0.0;
        
        std::vector<double> diff_spread;
        diff_spread.reserve(spread.size() - 1);
        
        for (size_t i = 1; i < spread.size(); ++i) {
            diff_spread.push_back(spread[i] - spread[i-1]);
        }
        
        std::vector<double> lagged_spread(spread.begin(), spread.end() - 1);
        
        double beta = calculate_ols_beta(diff_spread, lagged_spread);
        double se = calculate_standard_error(diff_spread, lagged_spread, beta);
        
        return (se != 0) ? beta / se : 0.0;
    }
    
    // Fast standard error calculation
    double calculate_standard_error(
        const std::vector<double>& y,
        const std::vector<double>& x,
        double beta
    ) {
        size_t n = y.size();
        if (n < 3) return 1.0;
        
        double sum_residuals = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double residual = y[i] - beta * x[i];
            sum_residuals += residual * residual;
        }
        
        double mse = sum_residuals / (n - 2);
        double sum_x2 = 0.0;
        for (double xi : x) {
            sum_x2 += xi * xi;
        }
        
        return std::sqrt(mse / sum_x2);
    }
    
    // Fast ADF p-value approximation
    double calculate_adf_pvalue(double adf_stat, size_t n) {
        // Simplified p-value calculation using critical values
        if (adf_stat < -3.5) return 0.01;
        if (adf_stat < -3.0) return 0.025;
        if (adf_stat < -2.6) return 0.05;
        if (adf_stat < -2.3) return 0.1;
        return 0.5;
    }
    
    // Fast half-life calculation
    double calculate_half_life(const std::vector<double>& spread) {
        if (spread.size() < 3) return 0.0;
        
        std::vector<double> diff_spread;
        std::vector<double> lagged_spread;
        
        for (size_t i = 1; i < spread.size(); ++i) {
            diff_spread.push_back(spread[i] - spread[i-1]);
            lagged_spread.push_back(spread[i-1]);
        }
        
        double beta = calculate_ols_beta(diff_spread, lagged_spread);
        
        if (beta >= 0) return 0.0;
        
        return std::log(2.0) / std::abs(beta);
    }
    
    // Fast correlation calculation
    double calculate_correlation(
        const std::vector<double>& x,
        const std::vector<double>& y
    ) {
        if (x.size() != y.size() || x.size() < 2) return 0.0;
        
        size_t n = x.size();
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
        double sum_x2 = 0.0, sum_y2 = 0.0;
        
        for (size_t i = 0; i < n; ++i) {
            sum_x += x[i];
            sum_y += y[i];
            sum_xy += x[i] * y[i];
            sum_x2 += x[i] * x[i];
            sum_y2 += y[i] * y[i];
        }
        
        double numerator = n * sum_xy - sum_x * sum_y;
        double denominator = std::sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y));
        
        return (denominator != 0) ? numerator / denominator : 0.0;
    }
};

// High-performance signal generation
class FastSignalGenerator {
private:
    FastCointegration cointegration_calc;
    std::unordered_map<std::string, std::vector<double>> price_cache;
    std::mutex cache_mutex;
    
public:
    // Fast signal generation for multiple pairs
    std::vector<std::tuple<std::string, std::string, std::string, double>> generate_signals(
        const std::unordered_map<std::string, std::vector<double>>& prices,
        double z_threshold = 2.0,
        double exit_threshold = 0.5
    ) {
        std::vector<std::tuple<std::string, std::string, std::string, double>> signals;
        
        std::vector<std::string> symbols;
        for (const auto& pair : prices) {
            symbols.push_back(pair.first);
        }
        
        // Generate signals for all pairs
        for (size_t i = 0; i < symbols.size(); ++i) {
            for (size_t j = i + 1; j < symbols.size(); ++j) {
                const std::string& symbol1 = symbols[i];
                const std::string& symbol2 = symbols[j];
                
                auto signal = generate_pair_signal(
                    prices.at(symbol1), prices.at(symbol2),
                    symbol1, symbol2, z_threshold, exit_threshold
                );
                
                if (!std::get<2>(signal).empty()) {
                    signals.push_back(signal);
                }
            }
        }
        
        return signals;
    }
    
    // Fast pair signal generation
    std::tuple<std::string, std::string, std::string, double> generate_pair_signal(
        const std::vector<double>& prices1,
        const std::vector<double>& prices2,
        const std::string& symbol1,
        const std::string& symbol2,
        double z_threshold,
        double exit_threshold
    ) {
        // Test for cointegration
        auto coint_result = cointegration_calc.test_cointegration(prices1, prices2);
        bool is_cointegrated = std::get<0>(coint_result);
        
        if (!is_cointegrated) {
            return {symbol1, symbol2, "", 0.0};
        }
        
        double beta = std::get<2>(coint_result);
        
        // Calculate spread
        std::vector<double> spread;
        spread.reserve(prices1.size());
        for (size_t i = 0; i < prices1.size(); ++i) {
            spread.push_back(prices1[i] - beta * prices2[i]);
        }
        
        // Calculate z-score
        double z_score = cointegration_calc.calculate_zscore(spread);
        
        // Generate signal
        std::string action = "";
        if (z_score > z_threshold) {
            action = "SHORT_SPREAD";
        } else if (z_score < -z_threshold) {
            action = "LONG_SPREAD";
        } else if (std::abs(z_score) < exit_threshold) {
            action = "EXIT_SPREAD";
        }
        
        return {symbol1, symbol2, action, z_score};
    }
    
    // Fast position sizing calculation
    std::vector<double> calculate_position_sizes(
        const std::vector<std::tuple<std::string, std::string, std::string, double>>& signals,
        double capital,
        double risk_per_trade = 0.02
    ) {
        std::vector<double> position_sizes;
        position_sizes.reserve(signals.size());
        
        for (const auto& signal : signals) {
            double confidence = std::min(std::abs(std::get<3>(signal)) / 3.0, 1.0);
            double position_size = capital * risk_per_trade * confidence;
            position_sizes.push_back(position_size);
        }
        
        return position_sizes;
    }
};

// High-performance risk management
class FastRiskManager {
private:
    std::vector<double> returns_history;
    std::vector<double> drawdown_history;
    double max_drawdown_limit;
    
public:
    FastRiskManager(double max_drawdown = 0.15) : max_drawdown_limit(max_drawdown) {}
    
    // Fast VaR calculation
    double calculate_var(const std::vector<double>& returns, double confidence = 0.95) {
        if (returns.size() < 30) return 0.0;
        
        std::vector<double> sorted_returns = returns;
        std::sort(sorted_returns.begin(), sorted_returns.end());
        
        size_t index = static_cast<size_t>((1.0 - confidence) * sorted_returns.size());
        return sorted_returns[index];
    }
    
    // Fast max drawdown calculation
    double calculate_max_drawdown(const std::vector<double>& equity_curve) {
        if (equity_curve.size() < 2) return 0.0;
        
        double peak = equity_curve[0];
        double max_drawdown = 0.0;
        
        for (double value : equity_curve) {
            if (value > peak) {
                peak = value;
            }
            double drawdown = (peak - value) / peak;
            max_drawdown = std::max(max_drawdown, drawdown);
        }
        
        return max_drawdown;
    }
    
    // Fast Sharpe ratio calculation
    double calculate_sharpe_ratio(
        const std::vector<double>& returns,
        double risk_free_rate = 0.02,
        size_t periods_per_year = 252
    ) {
        if (returns.size() < 30) return 0.0;
        
        double mean_return = 0.0;
        double variance = 0.0;
        
        for (double ret : returns) {
            mean_return += ret;
            variance += ret * ret;
        }
        
        mean_return /= returns.size();
        variance = (variance / returns.size()) - (mean_return * mean_return);
        
        if (variance < 1e-10) return 0.0;
        
        double excess_return = mean_return - risk_free_rate;
        double volatility = std::sqrt(variance * periods_per_year);
        
        return excess_return / volatility;
    }
    
    // Fast risk check
    bool check_risk_limits(
        const std::vector<double>& returns,
        const std::vector<double>& equity_curve
    ) {
        double current_var = calculate_var(returns);
        double current_drawdown = calculate_max_drawdown(equity_curve);
        
        return current_drawdown <= max_drawdown_limit;
    }
};

// Performance monitoring
class PerformanceMonitor {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::vector<double> latencies;
    std::mutex latency_mutex;
    
public:
    PerformanceMonitor() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    // Start timing
    void start_timer() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    // End timing and record latency
    double end_timer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double latency_ms = duration.count() / 1000.0;
        
        std::lock_guard<std::mutex> lock(latency_mutex);
        latencies.push_back(latency_ms);
        
        // Keep only recent measurements
        if (latencies.size() > 10000) {
            latencies.erase(latencies.begin(), latencies.begin() + 5000);
        }
        
        return latency_ms;
    }
    
    // Get performance statistics
    std::tuple<double, double, double> get_statistics() {
        std::lock_guard<std::mutex> lock(latency_mutex);
        
        if (latencies.empty()) {
            return {0.0, 0.0, 0.0};
        }
        
        double min_latency = *std::min_element(latencies.begin(), latencies.end());
        double max_latency = *std::max_element(latencies.begin(), latencies.end());
        
        double sum = 0.0;
        for (double latency : latencies) {
            sum += latency;
        }
        double avg_latency = sum / latencies.size();
        
        return {min_latency, avg_latency, max_latency};
    }
};

// Python bindings
PYBIND11_MODULE(fast_operations, m) {
    m.doc() = "Fast Operations C++ Extension for Algorithmic Trading Engine";
    
    py::class_<FastCointegration>(m, "FastCointegration")
        .def(py::init<size_t>(), py::arg("max_size") = 10000)
        .def("test_cointegration", &FastCointegration::test_cointegration)
        .def("calculate_zscore", &FastCointegration::calculate_zscore)
        .def("calculate_correlation_matrix", &FastCointegration::calculate_correlation_matrix)
        .def("calculate_volatility", &FastCointegration::calculate_volatility)
        .def("calculate_kelly_fraction", &FastCointegration::calculate_kelly_fraction);
    
    py::class_<FastSignalGenerator>(m, "FastSignalGenerator")
        .def(py::init<>())
        .def("generate_signals", &FastSignalGenerator::generate_signals)
        .def("generate_pair_signal", &FastSignalGenerator::generate_pair_signal)
        .def("calculate_position_sizes", &FastSignalGenerator::calculate_position_sizes);
    
    py::class_<FastRiskManager>(m, "FastRiskManager")
        .def(py::init<double>(), py::arg("max_drawdown") = 0.15)
        .def("calculate_var", &FastRiskManager::calculate_var)
        .def("calculate_max_drawdown", &FastRiskManager::calculate_max_drawdown)
        .def("calculate_sharpe_ratio", &FastRiskManager::calculate_sharpe_ratio)
        .def("check_risk_limits", &FastRiskManager::check_risk_limits);
    
    py::class_<PerformanceMonitor>(m, "PerformanceMonitor")
        .def(py::init<>())
        .def("start_timer", &PerformanceMonitor::start_timer)
        .def("end_timer", &PerformanceMonitor::end_timer)
        .def("get_statistics", &PerformanceMonitor::get_statistics);
}
