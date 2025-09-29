from typing import List, TypedDict

class TopicFocusedState(TypedDict):
    """Enhanced state with topic tracking"""
    original_topic: str         # The main topic to stay focused on
    current_depth_level: int    # Current depth of exploration (1-5)
    aspects_explored: List[str] # List of sub-aspects already covered
    topic_summary: str          # Concise summary of the topic so far
    last_followup_question: str # The last follow-up question asked
    iterations: int             # Number of iterations completed
    current_response: str       # Latest agent response content