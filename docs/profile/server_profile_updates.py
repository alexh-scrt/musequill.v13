# Update to src/server/models.py - Add evaluator_profile to ContentRequest

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class AgentMode(str, Enum):
    COLLABORATOR = "collaborator"
    SIMPLE = "simple"


class MessageType(str, Enum):
    USER_INPUT = "user_input"
    AGENT_RESPONSE = "agent_response"
    STATUS = "status"
    ERROR = "error"
    STOP = "stop"


class ContentRequest(BaseModel):
    topic: str
    mode: AgentMode = AgentMode.COLLABORATOR
    max_iterations: Optional[int] = 3
    stream: Optional[bool] = True
    evaluator_profile: Optional[str] = "general"  # NEW FIELD


# Update to src/server/app.py - Handle profile in request

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
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=300.0
                )
                message_data = json.loads(data)
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                })
                continue
            
            # Handle content request
            if message_data.get("type") == "content_request":
                request = ContentRequest(**message_data.get("data", {}))
                
                # Extract evaluator profile from request
                evaluator_profile = request.evaluator_profile or "general"
                
                # Send status update with profile info
                await websocket.send_json({
                    "type": MessageType.STATUS.value,
                    "content": f"Starting content creation for: {request.topic} (Profile: {evaluator_profile})",
                    "timestamp": datetime.now().isoformat()
                })
                
                # Create orchestrator with profile
                workflow = WorkflowOrchestrator(
                    evaluator_profile=evaluator_profile
                )
                    
                quality = float(os.getenv("QUALITY_THRESHOLD", "75.0"))
                
                # Run the workflow with streaming
                async for response in workflow.run_async(
                    topic=request.topic,
                    max_iterations=request.max_iterations,
                    quality_threshold=quality,
                    evaluator_profile=evaluator_profile  # Pass profile
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