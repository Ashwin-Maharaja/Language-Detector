"""
api/index.py
=============
Vercel Serverless Function entry point for the Multi-Script Language Detection Flask app.
"""

import os
import sys

# Ensure project root is in sys.path so 'src' imports resolve properly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.deploy import app

# Expose 'app' object for Vercel WSGI runner
app = app
