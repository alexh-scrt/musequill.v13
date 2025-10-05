"""
Model configuration for memory-efficient operation.
"""

import os

# Model selection based on memory constraints
MODEL_CONFIGS = {
    "large": {
        "writer": "llama3.3:70b",
        "editor": "qwen3:32b",
        "discriminator": "llama3.3:70b"
    },
    "medium": {
        "writer": "qwen3:32b",
        "editor": "qwen3:32b",
        "discriminator": "qwen3:32b"
    },
    "small": {
        "writer": "qwen3:8b",
        "editor": "qwen3:8b",
        "discriminator": "qwen3:8b"
    },
    "tiny": {
        "writer": "qwen2.5:3b",
        "editor": "qwen2.5:3b",
        "discriminator": "qwen2.5:3b"
    }
}

def get_model_config(profile: str = None):
    """
    Get model configuration based on profile or environment.
    
    Default to 'small' for memory safety.
    """
    if profile is None:
        profile = os.getenv("MODEL_PROFILE", "small")
    
    config = MODEL_CONFIGS.get(profile, MODEL_CONFIGS["small"])
    
    # Allow environment overrides
    config = {
        "writer": os.getenv("WRITER_MODEL", config["writer"]),
        "editor": os.getenv("EDITOR_MODEL", config["editor"]),
        "discriminator": os.getenv("DISCRIMINATOR_MODEL", config["discriminator"])
    }
    
    return config

# Ollama generation parameters for memory safety
GENERATION_PARAMS = {
    "temperature": 0.7,
    "num_ctx": 8192,  # Context window size
    "num_predict": 2048,  # Max tokens to generate
    "num_gpu": 1,  # Number of GPUs to use
    "num_thread": 8,  # Number of CPU threads
}