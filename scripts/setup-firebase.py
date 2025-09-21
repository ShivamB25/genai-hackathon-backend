#!/usr/bin/env python

# ==============================================================================
# AI-Powered Trip Planner Backend - Firebase Setup Script
# ==============================================================================
# This script automates Firebase initialization and emulator setup for local
# development.
#
# Usage:
#   python scripts/setup-firebase.py
#
# Prerequisites:
#   - Firebase CLI installed (npm install -g firebase-tools)
#   - Logged into Firebase (firebase login)
# ==============================================================================

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

# --- Configuration ---
FIREBASE_CONFIG_FILE = "firebase.json"
EMULATOR_CONFIG = {
    "auth": {"port": 9099},
    "firestore": {"port": 8080},
    "ui": {"enabled": True, "port": 4000},
}


# --- Helper Functions ---
def print_header(message):
    print("=" * 78)
    print(f" {message}")
    print("=" * 78)


def print_success(message):
    print(f"✅ {message}")


def print_error(message):
    print(f"❌ {message}", file=sys.stderr)
    sys.exit(1)

SAFE_EXECUTABLES = {"firebase"}


def _resolve_executable(executable: str) -> str:
    if executable not in SAFE_EXECUTABLES:
        print_error(f"Command not permitted: {executable}")

    resolved = shutil.which(executable)
    if not resolved:
        print_error(f"Command not found: {executable}")
    return resolved


def run_command(command: Sequence[str], capture_output: bool = False) -> str:
    if not command:
        print_error("Empty command provided.")

    resolved_command = [_resolve_executable(command[0]), *command[1:]]

    try:
        process = subprocess.run(
            resolved_command,
            check=True,
            text=True,
            capture_output=capture_output,
        )  # noqa: S603 - executable validated against SAFE_EXECUTABLES
        return process.stdout.strip() if capture_output else ""
    except subprocess.CalledProcessError as exc:
        print_error(
            "Command failed: "
            f"{' '.join(resolved_command)}\n{(exc.stderr or exc.stdout or '').strip()}"
        )
    except FileNotFoundError:
        print_error(f"Command not found: {resolved_command[0]}")

    return ""


# --- Main Script ---

# 1. Check Firebase CLI
print_header("Checking Firebase CLI installation...")
run_command(["firebase", "--version"])
print_success("Firebase CLI is installed.")

# 2. Check Firebase Login Status
print_header("Checking Firebase login status...")
try:
    run_command(["firebase", "login:list"])
    print_success("You are logged into Firebase.")
except subprocess.CalledProcessError:
    print(
        "You are not logged into Firebase. Please run 'firebase login' and try again."
    )
    sys.exit(1)

# 3. Select Firebase Project
print_header("Selecting Firebase project...")
project_id_output = run_command(
    [
        "firebase",
        "projects:list",
        "--json",
    ],
    capture_output=True,
)
if project_id_output:
    project_id = json.loads(project_id_output).get("projectId")
else:
    project_id = None

if not project_id:
    print_error("No Firebase projects found. Please create one.")

if project_id:
    run_command(["firebase", "use", project_id])
    print_success(f"Using Firebase project: {project_id}")
else:
    print_error("Project ID could not be determined.")

# 4. Initialize Firebase Emulators
print_header("Initializing Firebase Emulators...")
if project_id:
    run_command(
        [
            "firebase",
            "init",
            "emulators",
            "--project",
            project_id,
        ]
    )
    print_success("Firebase Emulators initialized.")
else:
    print_error("Project ID not found, cannot initialize emulators.")

# 5. Configure firebase.json
print_header("Configuring firebase.json...")
config_data = {"emulators": EMULATOR_CONFIG}
config_path = Path(FIREBASE_CONFIG_FILE)
config_path.write_text(json.dumps(config_data, indent=2))
print_success(f"{FIREBASE_CONFIG_FILE} configured for Auth and Firestore emulators.")

# --- Final Instructions ---
print_header("Firebase Setup Complete!")
print("Your Firebase environment is configured for local development.")
print("")
print("To start the emulators, run:")
print("  firebase emulators:start")
print("")
print(
    f"Emulator UI will be available at: http://localhost:{EMULATOR_CONFIG['ui']['port']}"
)
print(
    f"Firestore Emulator will be at: http://localhost:{EMULATOR_CONFIG['firestore']['port']}"
)
print(f"Auth Emulator will be at: http://localhost:{EMULATOR_CONFIG['auth']['port']}")
print("")
