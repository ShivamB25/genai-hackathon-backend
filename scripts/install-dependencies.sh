#!/bin/bash

# ==============================================================================
# AI-Powered Trip Planner Backend - Dependency Installation Script
# ==============================================================================
# This script installs system-level dependencies required for the project.
#
# Supported OS:
#   - macOS (with Homebrew)
#   - Debian/Ubuntu (with apt-get)
# ==============================================================================

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

function check_and_install_brew() {
    if ! command -v brew &> /dev/null; then
        print_header "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        print_success "Homebrew installed."
    else
        print_success "Homebrew is already installed."
    fi
}

function install_mac_deps() {
    print_header "Installing dependencies for macOS..."
    check_and_install_brew
    
    brew install python@3.12
    brew install node@18
    brew install firebase-cli
    brew install gcloud
    
    print_success "macOS dependencies installed."
}

function install_linux_deps() {
    print_header "Installing dependencies for Debian/Ubuntu..."
    
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-venv python3-pip curl
    
    # Install Node.js
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
    
    # Install Firebase CLI
    sudo curl -sL https://firebase.tools | bash
    
    # Install Google Cloud SDK
    sudo apt-get install -y apt-transport-https ca-certificates gnupg
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
    sudo apt-get update && sudo apt-get install -y google-cloud-sdk
    
    print_success "Linux dependencies installed."
}

# --- Main Script ---

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    *)          machine="UNKNOWN:${OS}"
esac

# Install dependencies based on OS
if [ "$machine" == "Mac" ]; then
    install_mac_deps
elif [ "$machine" == "Linux" ]; then
    install_linux_deps
else
    print_error "Unsupported operating system: $machine"
fi

echo ""
print_success "All system dependencies installed successfully."