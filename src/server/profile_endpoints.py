"""
Profile-aware content generation endpoints for FastAPI.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import json
import logging
from datetime import datetime
import uuid

from src.agents.profile_aware_generator import ProfileAwareGenerator, Document

logger = logging.getLogger(__name__)

# Create router for profile endpoints
router = APIRouter(prefix="/api/profile", tags=["profile"])


class ProfileGenerationRequest(BaseModel):
    """Request model for profile-aware generation"""
    topic: str
    profile: str = "scientific"
    max_depth: int = 3
    single_depth: Optional[int] = None
    session_id: Optional[str] = None
    stream: bool = False


class ProfileGenerationResponse(BaseModel):
    """Response model for profile generation"""
    session_id: str
    topic: str
    profile: str
    depths_generated: List[int]
    content: Dict[int, str]
    metadata: Dict[int, Dict[str, Any]]
    generation_time: float
    

class GenerationStatus(BaseModel):
    """Status of ongoing generation"""
    session_id: str
    status: str  # "pending", "processing", "completed", "failed"
    current_depth: Optional[int] = None
    total_depths: int
    message: Optional[str] = None
    

# Store for active generation tasks
active_generations: Dict[str, GenerationStatus] = {}


@router.post("/generate", response_model=ProfileGenerationResponse)
async def generate_profile_content(
    request: ProfileGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate profile and depth-aware content.
    
    This endpoint generates content with profile-specific requirements
    and depth differentiation to minimize repetition.
    """
    start_time = datetime.now()
    
    # Create session ID if not provided
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
    
    try:
        # Initialize generator
        generator = ProfileAwareGenerator(session_id=session_id)
        
        # Validate profile
        available_profiles = generator.get_available_profiles()
        if request.profile not in available_profiles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid profile. Available: {available_profiles}"
            )
        
        # Update status
        active_generations[session_id] = GenerationStatus(
            session_id=session_id,
            status="processing",
            current_depth=1,
            total_depths=request.max_depth if not request.single_depth else 1,
            message=f"Generating {request.profile} content"
        )
        
        if request.single_depth:
            # Generate single depth
            logger.info(f"Generating single depth {request.single_depth} for {request.topic}")
            
            content, metadata = await generator.generate_depth_section(
                topic=request.topic,
                profile=request.profile,
                depth=request.single_depth,
                previous_depths=None
            )
            
            response_content = {request.single_depth: content}
            response_metadata = {request.single_depth: metadata}
            depths_generated = [request.single_depth]
            
        else:
            # Generate full document
            logger.info(f"Generating full document for {request.topic}")
            
            document = await generator.generate_document(
                topic=request.topic,
                profile=request.profile,
                max_depth=request.max_depth
            )
            
            response_content = document.sections
            response_metadata = document.metadata
            depths_generated = list(document.sections.keys())
        
        # Update status
        active_generations[session_id].status = "completed"
        active_generations[session_id].message = "Generation successful"
        
        # Calculate generation time
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return ProfileGenerationResponse(
            session_id=session_id,
            topic=request.topic,
            profile=request.profile,
            depths_generated=depths_generated,
            content=response_content,
            metadata=response_metadata,
            generation_time=generation_time
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        
        # Update status
        if session_id in active_generations:
            active_generations[session_id].status = "failed"
            active_generations[session_id].message = str(e)
        
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{session_id}", response_model=GenerationStatus)
async def get_generation_status(session_id: str):
    """Get the status of an ongoing generation."""
    if session_id not in active_generations:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return active_generations[session_id]


@router.get("/profiles")
async def list_available_profiles():
    """List all available content profiles."""
    generator = ProfileAwareGenerator(session_id="temp")
    profiles = generator.get_available_profiles()
    
    # Get profile details from config
    profile_details = {}
    for profile in profiles:
        profile_details[profile] = {
            "name": profile,
            "max_depth": generator.get_max_depth_for_profile(profile),
            "description": f"Content profile for {profile.replace('_', ' ')} audience"
        }
    
    return {
        "profiles": profile_details,
        "default": "scientific"
    }


@router.post("/generate/stream")
async def generate_profile_content_stream(request: ProfileGenerationRequest):
    """
    Generate content with streaming responses.
    
    Streams progress updates as content is generated for each depth.
    """
    session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
    
    async def generate_stream():
        """Generator function for streaming responses"""
        try:
            generator = ProfileAwareGenerator(session_id=session_id)
            
            # Send initial status
            yield json.dumps({
                "type": "status",
                "session_id": session_id,
                "message": "Starting generation",
                "timestamp": datetime.now().isoformat()
            }) + "\n"
            
            if request.single_depth:
                # Single depth generation
                yield json.dumps({
                    "type": "progress",
                    "depth": request.single_depth,
                    "status": "generating",
                    "timestamp": datetime.now().isoformat()
                }) + "\n"
                
                content, metadata = await generator.generate_depth_section(
                    topic=request.topic,
                    profile=request.profile,
                    depth=request.single_depth
                )
                
                yield json.dumps({
                    "type": "content",
                    "depth": request.single_depth,
                    "content": content,
                    "metadata": metadata,
                    "timestamp": datetime.now().isoformat()
                }) + "\n"
                
            else:
                # Multi-depth generation
                document = Document(
                    topic=request.topic,
                    profile=request.profile,
                    sections={},
                    metadata={}
                )
                
                for depth in range(1, request.max_depth + 1):
                    # Send progress update
                    yield json.dumps({
                        "type": "progress",
                        "depth": depth,
                        "total": request.max_depth,
                        "status": "generating",
                        "timestamp": datetime.now().isoformat()
                    }) + "\n"
                    
                    # Get previous depths for context
                    previous_depths = document.get_previous_depths() if depth > 1 else None
                    
                    # Generate content for this depth
                    content, metadata = await generator.generate_depth_section(
                        topic=request.topic,
                        profile=request.profile,
                        depth=depth,
                        previous_depths=previous_depths
                    )
                    
                    # Add to document
                    document.add_section(depth, content, metadata)
                    
                    # Stream the content
                    yield json.dumps({
                        "type": "content",
                        "depth": depth,
                        "content": content,
                        "metadata": metadata,
                        "timestamp": datetime.now().isoformat()
                    }) + "\n"
            
            # Send completion status
            yield json.dumps({
                "type": "complete",
                "session_id": session_id,
                "message": "Generation completed successfully",
                "timestamp": datetime.now().isoformat()
            }) + "\n"
            
        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            yield json.dumps({
                "type": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }) + "\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="application/x-ndjson"
    )


@router.get("/health")
async def profile_health_check():
    """Check if profile generation system is healthy."""
    try:
        # Try to initialize generator
        generator = ProfileAwareGenerator(session_id="health_check")
        profiles = generator.get_available_profiles()
        
        return {
            "status": "healthy",
            "available_profiles": profiles,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )