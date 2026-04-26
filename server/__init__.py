"""OpenEnv server package shim.

This file exists to satisfy the OpenEnv multi-mode deployment layout
(`openenv validate` expects `server/app.py`). The actual FastAPI app lives in
:mod:`aic.server.env_api`; ``server.app`` re-exports it so both layouts work.
"""
