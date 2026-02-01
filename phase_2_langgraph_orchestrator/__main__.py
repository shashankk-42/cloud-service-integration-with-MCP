"""Entry point for running the orchestrator as a module."""

import asyncio
from .workflow import main

if __name__ == "__main__":
    asyncio.run(main())
