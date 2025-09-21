#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Run tests with coverage
pytest --cov=src --cov-report=html --cov-report=term-missing

# Open the coverage report in the default browser
# (Optional, comment out if not needed)
open htmlcov/index.html