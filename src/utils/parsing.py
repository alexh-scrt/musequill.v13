import re
import logging

logger = logging.getLogger(__name__)


def strip_think(text: str) -> str:
    """Remove <think>...</think> tags and their contents from text.
    
    This function removes reasoning/thinking tags that some LLM models
    (especially reasoning models) output before their final answer.
    
    Args:
        text: Input text that may contain <think> tags
        
    Returns:
        Text with all <think>...</think> sections removed and cleaned up
        
    Examples:
        >>> strip_think("Hello <think>reasoning here</think> world")
        'Hello  world'
        
        >>> strip_think("<think>thinking...</think>Final answer")
        'Final answer'
        
        >>> strip_think("Text with <THINK>case insensitive</THINK> tags")
        'Text with  tags'
    """
    if not text:
        return text
    
    original_length = len(text)
    
    pattern = r'<think>.*?</think>'
    cleaned = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    
    cleaned = cleaned.strip()
    
    if len(cleaned) < original_length:
        removed_chars = original_length - len(cleaned)
        logger.debug(f"Stripped {removed_chars} characters from <think> tags")
    
    return cleaned