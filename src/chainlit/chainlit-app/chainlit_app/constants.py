from __future__ import annotations

import os

# Directories
MODULE_FOLDER = os.path.dirname(os.path.abspath(__file__))
APP_FOLDER = os.path.dirname(MODULE_FOLDER)
PROJECT_PACKAGE = os.path.dirname(os.path.dirname(APP_FOLDER))
ROOT = os.path.dirname(PROJECT_PACKAGE)

# Files
CONFIG_FILE = os.path.join(APP_FOLDER, "config.yaml")

# Logging
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
