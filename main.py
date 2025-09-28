#!/usr/bin/env python3
import uvicorn
import os
import logging
from dotenv import load_dotenv, find_dotenv

f = load_dotenv(find_dotenv(), override=True)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.getLogger().setLevel(log_level)
level = logging.getLevelNamesMapping().get(log_level, logging.INFO)
logging.basicConfig(level=level)


if __name__ == "__main__":
    host = os.getenv("SERVER_HOST", "localhost")
    port = int(os.getenv("SERVER_PORT", 8080))
    
    uvicorn.run(
        "src.server.app:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "false",
        log_level=log_level.lower(),
        ws_ping_interval=20,  # Send ping every 20 seconds
        ws_ping_timeout=300,  # Wait 5 minutes for pong response
        ws_max_size=16777216,  # 16MB max message size
        timeout_keep_alive=300  # Keep connection alive for 5 minutes
    )