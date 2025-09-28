from typing import Dict, Any, Literal

ProfileType = Literal["creative", "balanced", "conservative"]


class GeneratorProfileFactory:
    """
    Profile factory for GeneratorAgent LLM parameters.
    
    Role: Conversational opener that provides concise answers and engaging follow-ups.
    - Creative: More exploratory, diverse vocabulary, open-ended responses
    - Balanced: Default conversational flow with moderate creativity  
    - Conservative: Focused, predictable, straightforward answers
    """
    
    @staticmethod
    def get(profile: ProfileType = "balanced") -> Dict[str, Any]:
        profiles = {
            "creative": {
                "temperature": 0.95,
                "top_p": 0.95,
                "top_k": 40,
                "repeat_penalty": 1.08,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.5,
                "typical_p": 0.85,
                "min_p": 0.04,
                "stop": ["\nFollow-up:", "Follow-up:"]
            },
            "balanced": {
                "temperature": 0.85,
                "top_p": 0.9,
                "top_k": 50,
                "repeat_penalty": 1.1,
                "frequency_penalty": 0.4,
                "presence_penalty": 0.4,
                "typical_p": 0.9,
                "min_p": 0.06,
                "stop": ["\nFollow-up:", "Follow-up:"]
            },
            "conservative": {
                "temperature": 0.65,
                "top_p": 0.85,
                "top_k": 60,
                "repeat_penalty": 1.15,
                "frequency_penalty": 0.5,
                "presence_penalty": 0.3,
                "typical_p": 0.95,
                "min_p": 0.08,
                "stop": ["\nFollow-up:", "Follow-up:"]
            }
        }
        
        if profile not in profiles:
            raise ValueError(f"Invalid profile '{profile}'. Must be one of: {list(profiles.keys())}")
        
        return profiles[profile]


class DiscriminatorProfileFactory:
    """
    Profile factory for DiscriminatorAgent LLM parameters.
    
    Role: Deep-dive collaborator that adds analytical depth and determines conversation closure.
    - Creative: Explores tangents, diverse perspectives, extends conversation
    - Balanced: Analytical but accessible, reasonable depth
    - Conservative: Crisp analysis, quicker to reach STOP, efficiency-focused
    """
    
    @staticmethod
    def get(profile: ProfileType = "balanced") -> Dict[str, Any]:
        profiles = {
            "creative": {
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 50,
                "repeat_penalty": 1.08,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.3,
                "typical_p": 0.9,
                "min_p": 0.05,
                "stop": ["\nSTOP", "STOP"]
            },
            "balanced": {
                "temperature": 0.6,
                "top_p": 0.85,
                "top_k": 60,
                "repeat_penalty": 1.12,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.25,
                "typical_p": 0.95,
                "min_p": 0.06,
                "stop": ["\nSTOP", "STOP"]
            },
            "conservative": {
                "temperature": 0.45,
                "top_p": 0.8,
                "top_k": 70,
                "repeat_penalty": 1.15,
                "frequency_penalty": 0.4,
                "presence_penalty": 0.2,
                "typical_p": 0.98,
                "min_p": 0.08,
                "stop": ["\nSTOP", "STOP"]
            }
        }
        
        if profile not in profiles:
            raise ValueError(f"Invalid profile '{profile}'. Must be one of: {list(profiles.keys())}")
        
        return profiles[profile]