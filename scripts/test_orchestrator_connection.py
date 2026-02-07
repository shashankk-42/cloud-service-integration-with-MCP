import asyncio
import logging
import sys
import os

# Ensure we can import from the project root
sys.path.append(os.getcwd())

from mcp.client.sse import sse_client
from mcp import ClientSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_orch_conn")

# ============== IPv4 Workaround ==============
import socket
_original_getaddrinfo = socket.getaddrinfo

def _getaddrinfo_ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = _getaddrinfo_ipv4_only
# ============================================

async def run_test(endpoint_url: str):
    logger.info(f"Connecting to MCP server at {endpoint_url}")
    
    try:
        # Mimic agents.py logic exactly
        logger.info("Starting manual context manager enter...")
        cm = sse_client(url=f"{endpoint_url}/sse")
        read_stream, write_stream = await cm.__aenter__()
        logger.info("SSE stream opened.")
        
        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()
        logger.info("Session context entered.")
        
        # Initialize session
        logger.info("Initializing session...")
        await session.initialize()
        logger.info("Session initialized successfully!")
        
        # List tools to verify
        result = await session.list_tools()
        logger.info(f"Tools found: {[t.name for t in result.tools]}")
        
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
    finally:
        # Cleanup isn't even in agents.py, but let's see if we get this far
        logger.info("Test complete")

if __name__ == "__main__":
    endpoint = "http://localhost:8000"
    if len(sys.argv) > 1:
        endpoint = sys.argv[1]
        
    asyncio.run(run_test(endpoint))
