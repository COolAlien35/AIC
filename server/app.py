"""OpenEnv server entry point expected by validators."""
from __future__ import annotations

import os

import uvicorn

from aic.server.env_api import app


def main() -> None:
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

