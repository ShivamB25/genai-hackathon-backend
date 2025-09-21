#!/usr/bin/env python

# ==============================================================================
# AI-Powered Trip Planner Backend - Environment Validation Script
# ==============================================================================
# This script validates the local development environment to ensure all
# configurations and dependencies are correctly set up.
#
# Usage:
#   python scripts/check-setup.py
# ==============================================================================

import asyncio
import os
import shutil
import subprocess
import sys
from typing import Any, Tuple

from dotenv import load_dotenv

# --- Configuration ---
REQUIRED_PYTHON_VERSION = (3, 12)
REQUIRED_NODE_VERSION = (18, 0)
REQUIRED_ENV_VARS = [
    "GOOGLE_CLOUD_PROJECT",
    "FIREBASE_PROJECT_ID",
    "GOOGLE_MAPS_API_KEY",
    "JWT_SECRET_KEY",
]


# --- Helper Functions ---
def print_header(message):
    print("=" * 78)
    print(f" {message}")
    print("=" * 78)


def print_success(message):
    print(f"✅ {message}")


def print_error(message):
    print(f"❌ {message}", file=sys.stderr)


def _is_version_compatible(
    current_version: Tuple[Any, ...], required_version: Tuple[int, ...]
) -> bool:
    return current_version >= required_version


def check_python_version(required_version: Tuple[int, ...]) -> bool:
    current_version = sys.version_info[: len(required_version)]
    if _is_version_compatible(current_version, required_version):
        print_success(
            f"python3 version is compatible ({'.'.join(map(str, current_version))})."
        )
        return True

    print_error(
        "python3 version is too old "
        f"({'.'.join(map(str, current_version))}). "
        f"Required: {'.'.join(map(str, required_version))}+"
    )
    return False


def check_node_version(required_version: Tuple[int, ...]) -> bool:
    node_path = shutil.which("node")
    if not node_path:
        print_error("node is not installed or not in PATH.")
        return False

    try:
        result = subprocess.run(  # noqa: S603 - command path resolved via shutil.which
            [node_path, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print_error(f"Unable to determine node version: {exc}")
        return False

    version_str = result.stdout.strip()
    version = tuple(map(int, version_str.split("v")[-1].split(".")))

    if _is_version_compatible(version, required_version):
        print_success(f"node version is compatible ({version_str}).")
        return True

    print_error(
        f"node version is too old ({version_str}). "
        f"Required: {'.'.join(map(str, required_version))}+"
    )
    return False


# --- Main Script ---


# 5. Check Service Connectivity
async def check_services():
    print_header("Checking external service connectivity...")
    try:
        from src.auth.firebase_auth import initialize_firebase
        from src.core.config import settings
        from src.maps_services.maps_client import get_maps_client

        # Check Firebase
        initialize_firebase()
        print_success("Firebase Admin SDK initialized successfully.")

        # Check Google Maps
        await get_maps_client()
        print_success("Google Maps client initialized successfully.")

        # Check Vertex AI
        # A simple check to see if the library can be initialized without errors
        from src.ai_services.model_config import initialize_vertex_ai

        initialize_vertex_ai()
        print_success("Vertex AI initialized successfully.")

    except Exception as e:
        print_error(f"Failed to connect to external services: {e}")
        sys.exit(1)


async def main():
    # 1. Check Python Version
    print_header("Checking Python version...")
    if not check_python_version(REQUIRED_PYTHON_VERSION):
        sys.exit(1)

    # 2. Check Node.js Version
    print_header("Checking Node.js version...")
    if not check_node_version(REQUIRED_NODE_VERSION):
        sys.exit(1)

    # 3. Check Dependencies
    print_header("Checking dependencies...")
    try:
        import fastapi
        import firebase_admin
        import googlemaps
        import vertexai

        print_success("Core Python dependencies are installed.")
    except ImportError as e:
        print_error(
            f"Missing Python dependency: {e.name}. Please run 'pip install -e .'"
        )
        sys.exit(1)

    # 4. Check Environment Variables
    print_header("Checking environment variables...")
    load_dotenv()
    missing_vars = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]
    if not missing_vars:
        print_success("All required environment variables are set.")
    else:
        print_error(f"Missing environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

    await check_services()

    print_header("Environment validation successful!")
    print("Your development environment is ready to go.")


if __name__ == "__main__":
    asyncio.run(main())
