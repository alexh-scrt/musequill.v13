from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import logging
from typing import Dict
from datetime import datetime
import os

from src.server.models import WebSocketMessage, MessageType, ContentRequest
from src.workflow.orchestrator import WorkflowOrchestrator
from src.llm.ollama_client import OllamaClient
from src.server.profile_endpoints import router as profile_router


logger = logging.getLogger(__name__)

app = FastAPI(title="Musequill Content Creator")

# Add CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include profile generation endpoints
app.include_router(profile_router)

# Store active connections
active_connections: Dict[str, WebSocket] = {}


@app.get("/")
async def root():
    return {"message": "Musequill Content Creator API", "status": "running"}


@app.get("/health")
async def health_check():
    try:
        # Check Ollama connection
        client = OllamaClient()
        models = await client.list_models()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "ollama_connected": len(models) > 0,
            "available_models": models
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connection_id = f"conn_{datetime.now().timestamp()}"
    active_connections[connection_id] = websocket
    
    try:
        logger.info(f"New WebSocket connection: {connection_id}")
        
        # Send initial connection message
        await websocket.send_json({
            "type": MessageType.STATUS.value,
            "content": "Connected to Musequill Content Creator",
            "timestamp": datetime.now().isoformat()
        })
        
        while True:
            # Receive message from client with 5-minute timeout
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=300.0  # 5 minutes timeout for debugging
                )
                message_data = json.loads(data)
            except asyncio.TimeoutError:
                # Send keepalive ping on timeout
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                })
                continue
            
            # Handle content request
            if message_data.get("type") == "content_request":
                request = ContentRequest(**message_data.get("data", {}))
                
                # Send status update
                await websocket.send_json({
                    "type": MessageType.STATUS.value,
                    "content": f"Starting content creation for: {request.topic}",
                    "timestamp": datetime.now().isoformat()
                })
                
                workflow = WorkflowOrchestrator(
                    evaluator_profile=request.evaluator_profile
                )
                    
                quality = float(os.getenv("QUALITY_THRESHOLD", "75.0"))
                # Run the workflow with streaming
                async for response in workflow.run_async(
                    topic=request.topic,
                    max_iterations=request.max_iterations,
                    quality_threshold=quality
                ):
                    # Send agent responses to client
                    await websocket.send_json({
                        "type": MessageType.AGENT_RESPONSE.value,
                        "content": response.content,
                        "agent_id": response.agent_id,
                        "metadata": {
                            "iteration": response.iteration,
                            "is_final": response.is_final
                        },
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Small delay for better streaming experience
                    await asyncio.sleep(0.1)
                
                # Send completion message
                await websocket.send_json({
                    "type": MessageType.COMPLETE.value,
                    "content": "Content creation completed",
                    "timestamp": datetime.now().isoformat()
                })
            
            # Handle ping/keepalive
            elif message_data.get("type") == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({
                "type": MessageType.ERROR.value,
                "content": f"Server error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            })
        except:
            pass
    finally:
        if connection_id in active_connections:
            del active_connections[connection_id]


@app.on_event("startup")
async def startup_event():
    logger.info("Musequill server starting up...")
    logger.info(f"Server running on {os.getenv('SERVER_HOST', 'localhost')}:{os.getenv('SERVER_PORT', 8000)}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Musequill server shutting down...")
    # Close all active WebSocket connections
    for conn_id, websocket in active_connections.items():
        try:
            await websocket.close()
        except:
            pass