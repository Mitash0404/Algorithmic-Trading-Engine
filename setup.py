#!/usr/bin/env python3
"""
Setup script for the Algorithmic Trading Engine.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="algorithmic-trading-engine",
    version="1.0.0",
    author="Mitash Shah",
    author_email="mitash.shah@example.com",
    description="A high-performance algorithmic trading engine with real-time market data processing, multi-strategy support, and comprehensive risk management",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/mitashshah/algorithmic_trading_engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.1",
            "loguru>=0.7.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "trading": [
            "ibapi>=9.76.1",
            "yfinance>=0.2.18",
            "pandas-datareader>=0.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "trading-engine=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="algorithmic trading, quantitative finance, trading engine, risk management, backtesting, cointegration, mean reversion",
    project_urls={
        "Bug Reports": "https://github.com/mitashshah/algorithmic_trading_engine/issues",
        "Source": "https://github.com/mitashshah/algorithmic_trading_engine",
        "Documentation": "https://algorithmic-trading-engine.readthedocs.io/",
    },
)
