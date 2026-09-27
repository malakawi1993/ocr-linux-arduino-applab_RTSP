#!/usr/bin/env python3
"""
Arduino App Lab entry point for Edge Impulse OCR Demo.

This script is the standard App Lab entry point (python/main.py).
It applies dependency mocks, then delegates to the main web_inference module.

Usage (from the app root directory):
    python3 python/main.py

Or via App Lab CLI:
    arduino-app-cli app start .
"""

import os
import sys

# Apply mocks BEFORE importing edge_impulse_linux (which web_inference imports)
from utils.mock_dependencies import apply_mocks
apply_mocks()

# Add the project root to Python path so web_inference can be imported.
# When run via App Lab CLI, the app is mounted at /app/ and this file
# is at /app/python/main.py. We need /app/ on sys.path.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Also add /app/ explicitly for the Docker container environment
if os.path.isfile('/app/web_inference.py'):
    sys.path.insert(0, '/app')

# Change working directory to project root so model auto-detection works
os.chdir(project_root)

from web_inference import main

if __name__ == '__main__':
    main()
