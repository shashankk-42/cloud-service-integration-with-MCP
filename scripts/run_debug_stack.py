import subprocess
import time
import sys
import os
import signal

def main():
    # Set unique port for debugging
    env = os.environ.copy()
    env["MCP_PORT"] = "8005"
    env["PYTHONUNBUFFERED"] = "1"
    
    # Start Server
    print("Starting AWS MCP Server on port 8005...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "phase_1_mcp_core.servers.aws.aws_server"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    try:
        # Wait for server to start
        time.sleep(5)
        
        # Start Client
        print("Starting Debug Client...")
        client_process = subprocess.run(
            [sys.executable, "scripts/mcp_debug_client.py", "http://localhost:8005"],
            capture_output=True,
            text=True
        )
        
        print("\n=== Client Output ===")
        print(client_process.stdout)
        print(client_process.stderr)
        
        print("\n=== Server Output ===")
        # Read whatever the server has output so far
        # We need to be careful not to block, but since we are killing it soon it's okay?
        # Actually, let's read non-blocking or just kill and read.
        
        # Give server a moment to flush if it crashed
        time.sleep(1)
        
    finally:
        print("\nStopping Server...")
        server_process.terminate()
        try:
            outs, errs = server_process.communicate(timeout=5)
            print(outs)
        except subprocess.TimeoutExpired:
            server_process.kill()
            outs, errs = server_process.communicate()
            print(outs)

if __name__ == "__main__":
    main()
