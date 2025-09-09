#!/bin/bash

# GitHub Upload Script for Algorithmic Trading Engine
# This script will automatically upload your project to GitHub

set -e  # Exit on any error

echo "🚀 Starting GitHub upload process for Algorithmic Trading Engine..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "main.py" ] || [ ! -f "README.md" ]; then
    print_error "Please run this script from the algorithmic_trading_engine directory"
    exit 1
fi

# Check if git is installed
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install Git first."
    exit 1
fi

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    print_warning "GitHub CLI not found. Will use manual method."
    USE_GH_CLI=false
else
    print_success "GitHub CLI found. Will use automated method."
    USE_GH_CLI=true
fi

# Initialize git repository if not already done
if [ ! -d ".git" ]; then
    print_status "Initializing git repository..."
    git init
    print_success "Git repository initialized"
else
    print_status "Git repository already exists"
fi

# Add all files
print_status "Adding files to git..."
git add .
print_success "Files added to staging area"

# Check if there are changes to commit
if git diff --staged --quiet; then
    print_warning "No changes to commit. Repository might already be up to date."
else
    # Create initial commit
    print_status "Creating initial commit..."
    git commit -m "Initial commit: Algorithmic Trading Engine

- High-performance trading engine with C++ extensions
- Multi-strategy support (cointegration, mean reversion)
- Real-time market data processing
- Comprehensive risk management
- Interactive Brokers integration
- Bayesian optimization for hyperparameter tuning
- Backtesting framework
- Performance analytics and reporting"
    print_success "Initial commit created"
fi

# Create GitHub repository
if [ "$USE_GH_CLI" = true ]; then
    print_status "Creating GitHub repository using GitHub CLI..."
    
    # Check if user is logged in
    if ! gh auth status &> /dev/null; then
        print_warning "Not logged into GitHub CLI. Please log in first:"
        echo "Run: gh auth login"
        exit 1
    fi
    
    # Create repository
    gh repo create algorithmic_trading_engine --public --description "A high-performance algorithmic trading engine with real-time market data processing, multi-strategy support, and comprehensive risk management" --source=. --remote=origin --push
    
    print_success "Repository created and pushed to GitHub!"
    print_success "Repository URL: https://github.com/$(gh api user --jq .login)/algorithmic_trading_engine"
    
else
    print_warning "GitHub CLI not available. Manual setup required."
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://github.com/new"
    echo "2. Create a new repository named 'algorithmic_trading_engine'"
    echo "3. Make it public"
    echo "4. Add description: 'A high-performance algorithmic trading engine with real-time market data processing, multi-strategy support, and comprehensive risk management'"
    echo "5. Do NOT initialize with README, .gitignore, or license (we already have them)"
    echo "6. Click 'Create repository'"
    echo ""
    echo "Then run these commands:"
    echo "git remote add origin https://github.com/YOUR_USERNAME/algorithmic_trading_engine.git"
    echo "git branch -M main"
    echo "git push -u origin main"
    echo ""
    read -p "Press Enter after you've created the repository and run the commands..."
fi

# Verify upload
print_status "Verifying upload..."
if git remote get-url origin &> /dev/null; then
    REPO_URL=$(git remote get-url origin)
    print_success "Repository URL: $REPO_URL"
    
    # Check if we can fetch from remote
    if git fetch origin &> /dev/null; then
        print_success "✅ Upload successful! Your repository is now on GitHub."
        echo ""
        echo "🎉 Your Algorithmic Trading Engine is now live at:"
        echo "   $REPO_URL"
        echo ""
        echo "Next steps:"
        echo "1. Visit your repository on GitHub"
        echo "2. Add repository topics (algorithmic-trading, quantitative-finance, python, cpp)"
        echo "3. Set up branch protection rules"
        echo "4. Consider adding GitHub Actions for CI/CD"
        echo "5. Create your first release"
    else
        print_error "Failed to verify upload. Please check your repository manually."
    fi
else
    print_warning "No remote repository configured. Please add the remote manually."
fi

print_success "Script completed! 🚀"
