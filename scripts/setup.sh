#!/bin/bash

# ==============================================================================
# AI-Powered Trip Planner Backend - Complete Environment Setup Script
# ==============================================================================
# This script automates the entire local development setup process.
#
# Usage:
#   ./scripts/setup.sh
#
# Prerequisites:
#   - Homebrew (for macOS) or apt-get (for Debian/Ubuntu)
#   - Python 3.12+
#   - Node.js and npm
# ==============================================================================

# --- Configuration ---
PYTHON_VERSION="3.12"
NODE_VERSION="18"
VENV_DIR="venv"
ENV_FILE=".env"
ENV_EXAMPLE_FILE=".env.example"

# --- Helper Functions ---
function print_header() {
    echo "=============================================================================="
    echo " $1"
    echo "=============================================================================="
}

function print_success() {
    echo "✅ $1"
}

function print_error() {
    echo "❌ $1"
    exit 1
}

function check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is not installed. Please install it and try again."
    fi
}

# --- Main Script ---

# 1. Check Prerequisites
print_header "Checking prerequisites..."
check_command "python3"
check_command "node"
check_command "npm"
check_command "git"
print_success "Prerequisites checked successfully."

# 2. Install Dependencies
print_header "Installing dependencies..."
./scripts/install-dependencies.sh
print_success "Dependencies installed successfully."

# 3. Set up Python Virtual Environment
if [ ! -d "$VENV_DIR" ]; then
    print_header "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    print_success "Virtual environment created."
fi

print_header "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
print_success "Virtual environment activated."

print_header "Installing Python packages..."
pip install --upgrade pip
pip install -e .
print_success "Python packages installed."

# 4. Configure Environment Variables
if [ ! -f "$ENV_FILE" ]; then
    print_header "Configuring environment variables..."
    cp "$ENV_EXAMPLE_FILE" "$ENV_FILE"
    echo "Created .env file. Please edit it with your credentials."
else
    echo ".env file already exists. Skipping creation."
fi
print_success "Environment variables configured."

# 5. Set up Firebase
print_header "Setting up Firebase..."
./scripts/setup-firebase.py
print_success "Firebase setup complete."

# 6. Validate Environment
print_header "Validating environment setup..."
./scripts/check-setup.py
print_success "Environment validation successful."

# --- Final Instructions ---
print_header "Setup Complete!"
echo "Your local development environment is ready."
echo ""
echo "To start the development server, run:"
echo "  source $VENV_DIR/bin/activate"
echo "  uvicorn src.main:app --reload"
echo ""
echo "To start Firebase emulators, run:"
echo "  firebase emulators:start"
echo ""