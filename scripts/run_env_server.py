#!/usr/bin/env python3
"""Run the AIC FastAPI/OpenEnv environment service locally."""
from __future__ import annotations

import sys
from pathlib import Path

import uvicorn


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


if __name__ == "__main__":
    uvicorn.run("aic.server.env_api:app", host="0.0.0.0", port=8000, reload=False)