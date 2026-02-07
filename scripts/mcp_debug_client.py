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
logger = logging.getLogger("mcp_debug_client")

async def run_debug_session(endpoint_url: str):
    logger.info(f"Connecting to MCP server at {endpoint_url}")
    
    try:
        async with sse_client(url=f"{endpoint_url}/sse") as (read_stream, write_stream):
            logger.info("SSE connection established")
            
            async with ClientSession(read_stream, write_stream) as session:
                logger.info("Initializing session...")
                await session.initialize()
                logger.info("Session initialized")
                
                # List tools
                logger.info("Listing tools...")
                result = await session.list_tools()
                logger.info(f"Tools found: {[t.name for t in result.tools]}")
                
                # Try a specific tool call that was failing in the orchestrator
                logger.info("Calling get_cost_estimate...")
                cost_params = {
                    "resource_type": "compute",
                    "instance_type": "t3.medium",
                    "region": "us-east-1",
                    "duration_hours": 720,
                    "quantity": 1
                }
                
                result = await session.call_tool("get_cost_estimate", cost_params)
                logger.info(f"Tool call result: {result}")
                
    except Exception as e:
        logger.error(f"Error during debug session: {e}", exc_info=True)

if __name__ == "__main__":
    # Default to AWS MCP server port
    endpoint = "http://localhost:8000"
    if len(sys.argv) > 1:
        endpoint = sys.argv[1]
        
    asyncio.run(run_debug_session(endpoint))
