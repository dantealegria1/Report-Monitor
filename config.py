"""
Configuration module for Report Monitor
 application.
Handles environment variables and application settings.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")

# Validate required environment variables
def validate_config():
    """Validate that all required environment variables are set."""
    missing = [
        k for k, v in {
            "DB_USER": DB_USER,
            "DB_PASSWORD": DB_PASS,
            "DB_HOST": DB_HOST,
            "DB_NAME": DB_NAME
        }.items() if not v
    ]
    if missing:
        raise RuntimeError(f"Faltan variables de entorno: {', '.join(missing)}")

# Application settings
CACHE_TTL = 600  # Cache time-to-live in seconds
MAX_DURATION_HOURS = 6  # Maximum reasonable report duration
TOP_REPORTS_LIMIT = 200  # Number of top reports to show in filters
