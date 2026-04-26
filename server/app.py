"""OpenEnv multi-mode deployment shim.

Re-exports the FastAPI app defined in :mod:`aic.server.env_api` and provides a
``main`` entrypoint for the ``[project.scripts] server`` console script.

Run locally::

    uvicorn server.app:app --host 0.0.0.0 --port 8000

Or via the console script (after ``pip install -e .``)::

    server
"""
from __future__ import annotations

import os

from aic.server.env_api import app

__all__ = ["app", "main"]


def main() -> None:
    """Console-script entrypoint that starts uvicorn for the AIC environment."""
    import uvicorn

    host = os.environ.get("OPENENV_HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("OPENENV_PORT", "8000")))
    uvicorn.run("server.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
