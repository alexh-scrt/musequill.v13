from typing import List, TypedDict, Dict, Any, Optional, Tuple

class TopicFocusedState(TypedDict):
    """Enhanced state with topic tracking and quality control"""
    original_topic: str         # The main topic to stay focused on
    current_depth_level: int    # Current depth of exploration (1-5)
    aspects_explored: List[str] # List of sub-aspects already covered
    topic_summary: str          # Concise summary of the topic so far
    last_followup_question: str # The last follow-up question asked
    iterations: int             # Number of iterations completed
    current_response: str       # Latest agent response content
    
    # Quality tracking fields
    generator_revisions: List[Tuple[str, float]]  # List of (content, score) tuples
    discriminator_revisions: List[Tuple[str, float]]  # List of (response, score) tuples
    evaluator_feedback: Optional[str]  # Feedback from evaluator for revision
    quality_scores: Dict[str, float]  # Track quality scores by agent
    generator_iterations: int  # Track generator revision iterations
    discriminator_iterations: int  # Track discriminator revision iterations
    best_generator_content: Optional[str]  # Best generator content for discriminator
    evaluation: Optional[Any]  # Store full EvaluationResult for reference
    context: Optional[Dict[str, Any]]  # Context for evaluation (references, figures, etc.)