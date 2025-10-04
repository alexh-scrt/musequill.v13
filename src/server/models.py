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


class WebSocketMessage(BaseModel):
    type: MessageType
    content: str
    agent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __init__(self, **data):
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now()
        super().__init__(**data)


class ContentRequest(BaseModel):
    topic: str
    mode: AgentMode = AgentMode.COLLABORATOR
    max_iterations: Optional[int] = 3
    stream: Optional[bool] = True
    evaluator_profile: Optional[str] = "general"


class AgentResponse(BaseModel):
    agent_id: str
    content: str
    iteration: int
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None