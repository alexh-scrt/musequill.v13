"""
Configuration system for similarity detection.

Provides centralized configuration for easy tuning of the similarity
detection system without code changes.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class SimilarityConfig:
    """
    Centralized configuration for similarity detection.
    
    All thresholds and parameters can be tuned through environment
    variables or configuration files.
    """
    
    # Similarity thresholds
    identical_threshold: float = 0.90
    very_similar_threshold: float = 0.75
    similar_threshold: float = 0.60
    
    # Information gain requirements
    min_novelty_ratio: float = 0.40  # 40% new concepts required
    
    # Behavioral flags
    auto_skip_identical: bool = True
    always_flag_never_skip: bool = False
    use_enhanced_similarity: bool = True
    
    # Processing limits
    max_similarity_attempts: int = 5
    similarity_relaxed_threshold: float = 0.90
    
    # Content type specific settings
    check_tables: bool = True
    check_paragraphs: bool = True
    check_equations: bool = False  # Equations often similar by nature
    check_examples: bool = True
    
    # Logging and debugging
    log_all_decisions: bool = True
    save_review_queue: bool = True
    generate_reports: bool = True
    
    # Corpus settings
    paragraph_min_length: int = 50
    sliding_window_size: int = 3
    sliding_window_activation_threshold: int = 100
    
    # Profile-specific overrides
    profile: str = "balanced"
    
    @classmethod
    def from_env(cls) -> 'SimilarityConfig':
        """
        Load configuration from environment variables.
        
        Environment variables follow the pattern:
        SIMILARITY_<SETTING_NAME> for similarity-specific settings
        """
        config = cls()
        
        # Load thresholds
        config.identical_threshold = float(os.getenv(
            "SIMILARITY_IDENTICAL_THRESHOLD",
            str(config.identical_threshold)
        ))
        config.very_similar_threshold = float(os.getenv(
            "SIMILARITY_VERY_SIMILAR_THRESHOLD",
            str(config.very_similar_threshold)
        ))
        config.similar_threshold = float(os.getenv(
            "SIMILARITY_SIMILAR_THRESHOLD",
            str(config.similar_threshold)
        ))
        
        # Load information gain settings
        config.min_novelty_ratio = float(os.getenv(
            "MIN_NOVELTY_RATIO",
            str(config.min_novelty_ratio)
        ))
        
        # Load behavioral flags
        config.auto_skip_identical = os.getenv(
            "AUTO_SKIP_IDENTICAL",
            str(config.auto_skip_identical)
        ).lower() in ["true", "1", "yes"]
        
        config.always_flag_never_skip = os.getenv(
            "ALWAYS_FLAG_NEVER_SKIP",
            str(config.always_flag_never_skip)
        ).lower() in ["true", "1", "yes"]
        
        config.use_enhanced_similarity = os.getenv(
            "USE_ENHANCED_SIMILARITY",
            str(config.use_enhanced_similarity)
        ).lower() in ["true", "1", "yes"]
        
        # Load processing limits
        config.max_similarity_attempts = int(os.getenv(
            "MAX_SIMILARITY_ATTEMPTS",
            str(config.max_similarity_attempts)
        ))
        config.similarity_relaxed_threshold = float(os.getenv(
            "SIMILARITY_RELAXED_THRESHOLD",
            str(config.similarity_relaxed_threshold)
        ))
        
        # Load content type settings
        config.check_tables = os.getenv(
            "SIMILARITY_CHECK_TABLES",
            str(config.check_tables)
        ).lower() in ["true", "1", "yes"]
        
        config.check_paragraphs = os.getenv(
            "SIMILARITY_CHECK_PARAGRAPHS",
            str(config.check_paragraphs)
        ).lower() in ["true", "1", "yes"]
        
        config.check_equations = os.getenv(
            "SIMILARITY_CHECK_EQUATIONS",
            str(config.check_equations)
        ).lower() in ["true", "1", "yes"]
        
        # Load corpus settings
        config.paragraph_min_length = int(os.getenv(
            "PARAGRAPH_MIN_LENGTH",
            str(config.paragraph_min_length)
        ))
        
        # Load profile
        config.profile = os.getenv(
            "SIMILARITY_PROFILE",
            config.profile
        )
        
        return config
    
    @classmethod
    def from_file(cls, filepath: str) -> 'SimilarityConfig':
        """
        Load configuration from a JSON file.
        
        Args:
            filepath: Path to the configuration file
            
        Returns:
            SimilarityConfig instance
        """
        config_path = Path(filepath)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Create config with loaded values
        config = cls()
        
        # Update with file values
        for key, value in config_data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    @classmethod
    def get_profile_config(cls, profile: str) -> 'SimilarityConfig':
        """
        Get configuration for a specific profile.
        
        Profiles allow different configurations for different use cases:
        - strict: Very low tolerance for similarity
        - balanced: Middle ground (default)
        - relaxed: More permissive, allows more similarity
        - creative: Focus on novelty and creativity
        - technical: Focus on technical accuracy over novelty
        
        Args:
            profile: Profile name
            
        Returns:
            SimilarityConfig with profile-specific settings
        """
        # Start with environment config as base
        config = cls.from_env()
        config.profile = profile
        
        if profile == "strict":
            config.identical_threshold = 0.85
            config.very_similar_threshold = 0.70
            config.similar_threshold = 0.55
            config.min_novelty_ratio = 0.50
            config.auto_skip_identical = True
            config.always_flag_never_skip = False
            
        elif profile == "balanced":
            # Use defaults
            pass
            
        elif profile == "relaxed":
            config.identical_threshold = 0.95
            config.very_similar_threshold = 0.85
            config.similar_threshold = 0.70
            config.min_novelty_ratio = 0.30
            config.auto_skip_identical = False
            config.always_flag_never_skip = False
            
        elif profile == "creative":
            config.identical_threshold = 0.88
            config.very_similar_threshold = 0.73
            config.similar_threshold = 0.58
            config.min_novelty_ratio = 0.45
            config.auto_skip_identical = True
            config.check_equations = False
            config.check_examples = True
            
        elif profile == "technical":
            config.identical_threshold = 0.92
            config.very_similar_threshold = 0.80
            config.similar_threshold = 0.65
            config.min_novelty_ratio = 0.35
            config.auto_skip_identical = False
            config.check_equations = True
            config.check_examples = True
            config.always_flag_never_skip = True  # Always review technical content
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """
        Export configuration to JSON.
        
        Args:
            filepath: Optional path to save the JSON file
            
        Returns:
            JSON string representation
        """
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check threshold ordering
        if not (0 <= self.similar_threshold <= self.very_similar_threshold <= self.identical_threshold <= 1.0):
            raise ValueError(
                f"Invalid threshold ordering. Must be: "
                f"0 <= similar ({self.similar_threshold}) "
                f"<= very_similar ({self.very_similar_threshold}) "
                f"<= identical ({self.identical_threshold}) <= 1.0"
            )
        
        # Check novelty ratio
        if not (0 <= self.min_novelty_ratio <= 1.0):
            raise ValueError(
                f"Invalid novelty ratio: {self.min_novelty_ratio}. "
                f"Must be between 0 and 1."
            )
        
        # Check attempts
        if self.max_similarity_attempts < 1:
            raise ValueError(
                f"Invalid max attempts: {self.max_similarity_attempts}. "
                f"Must be at least 1."
            )
        
        # Check relaxed threshold
        if not (self.very_similar_threshold <= self.similarity_relaxed_threshold <= 1.0):
            raise ValueError(
                f"Invalid relaxed threshold: {self.similarity_relaxed_threshold}. "
                f"Should be between very_similar ({self.very_similar_threshold}) and 1.0"
            )
        
        return True
    
    def describe(self) -> str:
        """
        Generate human-readable description of the configuration.
        
        Returns:
            Formatted description string
        """
        lines = [
            "=" * 60,
            "SIMILARITY DETECTION CONFIGURATION",
            f"Profile: {self.profile}",
            "=" * 60,
            "",
            "THRESHOLDS",
            "-" * 40,
            f"  Identical:      {self.identical_threshold:.1%}",
            f"  Very Similar:   {self.very_similar_threshold:.1%}",
            f"  Similar:        {self.similar_threshold:.1%}",
            f"  Relaxed:        {self.similarity_relaxed_threshold:.1%}",
            "",
            "INFORMATION GAIN",
            "-" * 40,
            f"  Min Novelty:    {self.min_novelty_ratio:.1%}",
            "",
            "BEHAVIOR",
            "-" * 40,
            f"  Auto Skip:      {self.auto_skip_identical}",
            f"  Always Flag:    {self.always_flag_never_skip}",
            f"  Enhanced Mode:  {self.use_enhanced_similarity}",
            f"  Max Attempts:   {self.max_similarity_attempts}",
            "",
            "CONTENT TYPES",
            "-" * 40,
            f"  Check Tables:   {self.check_tables}",
            f"  Check Paragraphs: {self.check_paragraphs}",
            f"  Check Equations: {self.check_equations}",
            f"  Check Examples: {self.check_examples}",
            "=" * 60
        ]
        
        return "\n".join(lines)


# Global configuration instance
_global_config: Optional[SimilarityConfig] = None


def get_config() -> SimilarityConfig:
    """
    Get the global configuration instance.
    
    Loads from environment on first call.
    
    Returns:
        Global SimilarityConfig instance
    """
    global _global_config
    
    if _global_config is None:
        _global_config = SimilarityConfig.from_env()
        _global_config.validate()
    
    return _global_config


def set_config(config: SimilarityConfig):
    """
    Set the global configuration.
    
    Args:
        config: New configuration to use
    """
    global _global_config
    config.validate()
    _global_config = config


def reset_config():
    """Reset configuration to defaults."""
    global _global_config
    _global_config = None