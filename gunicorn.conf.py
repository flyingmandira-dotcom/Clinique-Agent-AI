# Gunicorn configuration file for Render deployment
# This file is automatically loaded by Gunicorn when starting up.

import os

# Use the Uvicorn ASGI worker class so that Gunicorn can run the FastAPI ASGI app
worker_class = "uvicorn.workers.UvicornWorker"

# Adjust timeout for LLM agent processing
timeout = 120

# Number of worker processes
workers = int(os.environ.get("WEB_CONCURRENCY", 2))
