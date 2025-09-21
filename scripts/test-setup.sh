#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Install testing dependencies
pip install -e .[test]