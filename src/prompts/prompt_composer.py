"""
Prompt composition system for profile and depth-aware content generation.
"""

from typing import Dict, List, Optional
import yaml
import re
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationContext:
    """Context for generating content at a specific depth"""
    topic: str
    depth: int
    profile: str
    previous_depths: Optional[Dict[int, str]] = None
    

class PromptComposer:
    """
    Composes prompts by combining:
    1. Base system prompt
    2. Profile configuration
    3. Depth framework
    4. Context from previous depths
    """
    
    def __init__(self, 
                 profile_config_path: Optional[str] = None,
                 depth_framework_path: Optional[str] = None):
        # Use default paths if not provided
        base_path = Path(__file__).parent.parent  # src/ directory
        
        if profile_config_path is None:
            profile_config_path = base_path / "config" / "profiles.yaml"
        if depth_framework_path is None:
            depth_framework_path = base_path / "config" / "depth_framework.yaml"
            
        self.profiles = self._load_yaml(profile_config_path)
        self.depth_framework = self._load_yaml(depth_framework_path)
        self.base_prompt = self._load_base_prompt()
    
    def _load_yaml(self, path: str) -> dict:
        """Load YAML configuration file"""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}
        
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_base_prompt(self) -> str:
        """Load the base system prompt common to all generations"""
        return """
You are an expert content generator creating depth-differentiated content.

Your task is to generate content that:
1. Matches the specified profile and audience
2. Fulfills the requirements of the specified depth level
3. Builds on previous depths without repetition
4. Adds new information and insights

CRITICAL: Each depth must be SUBSTANTIVELY DIFFERENT from other depths.
Do not simply restate or rephrase previous content.
"""
    
    def compose_prompt(self, context: GenerationContext) -> str:
        """
        Main method: Compose a complete prompt for generation
        """
        # Check if profile exists
        if context.profile not in self.profiles.get('profiles', {}):
            logger.warning(f"Profile '{context.profile}' not found. Using default.")
            return self._compose_fallback_prompt(context)
            
        profile_config = self.profiles['profiles'][context.profile]
        depth_key = f'depth_{context.depth}'
        
        if depth_key not in profile_config:
            logger.warning(f"Depth {context.depth} not configured for profile {context.profile}")
            return self._compose_fallback_prompt(context)
            
        depth_config_profile = profile_config[depth_key]
        depth_config_universal = self.depth_framework.get('depth_framework', {}).get(depth_key, {})
        
        # Build prompt sections
        sections = [
            self.base_prompt,
            self._build_topic_section(context.topic),
            self._build_profile_section(profile_config, depth_config_profile),
            self._build_depth_section(depth_config_universal, context.depth),
            self._build_constraints_section(depth_config_profile),
            self._build_context_section(context, depth_config_universal),
            self._build_examples_section(depth_config_universal),
            self._build_output_requirements(depth_config_profile)
        ]
        
        return "\n\n".join(filter(None, sections))
    
    def _compose_fallback_prompt(self, context: GenerationContext) -> str:
        """Fallback prompt when profile/depth not configured"""
        return f"""
{self.base_prompt}

## TOPIC
{context.topic}

## DEPTH LEVEL: {context.depth}

Generate content appropriate for depth level {context.depth}.
Each depth should provide different perspectives and information.
"""
    
    def _build_topic_section(self, topic: str) -> str:
        return f"""
## TOPIC
{topic}
"""
    
    def _build_profile_section(self, profile_config: dict, depth_config: dict) -> str:
        return f"""
## PROFILE: {profile_config.get('name', 'Unknown')}
Target Audience: {profile_config.get('target_audience', 'General')}

Purpose at this depth: {depth_config.get('purpose', 'Generate relevant content')}
Tone: {depth_config.get('tone', 'professional')}
Complexity Level: {depth_config.get('complexity_level', 'appropriate')}
"""
    
    def _build_depth_section(self, depth_config: dict, depth: int) -> str:
        if not depth_config:
            return f"## DEPTH LEVEL: {depth}"
            
        return f"""
## DEPTH LEVEL: {depth}
Universal Purpose: {depth_config.get('universal_purpose', '')}
Key Question to Answer: {depth_config.get('question_pattern', '')}
Information Density: {depth_config.get('information_density', 'medium')}
Abstraction Level: {depth_config.get('abstraction_level', 'appropriate')}
"""
    
    def _build_constraints_section(self, depth_config: dict) -> str:
        """Build explicit constraints section"""
        constraints = []
        
        # Required elements
        if 'required_elements' in depth_config:
            req = depth_config['required_elements']
            constraints.append(f"MUST INCLUDE: {', '.join(req)}")
        
        # Forbidden elements
        if 'forbidden_elements' in depth_config:
            forb = depth_config['forbidden_elements']
            constraints.append(f"MUST NOT INCLUDE: {', '.join(forb)}")
        
        # Quantitative requirements
        if 'min_equations' in depth_config:
            constraints.append(f"Minimum equations: {depth_config['min_equations']}")
        
        if 'max_equations' in depth_config:
            constraints.append(f"Maximum equations: {depth_config['max_equations']}")
        
        if 'min_citations' in depth_config:
            constraints.append(f"Minimum citations: {depth_config['min_citations']}")
        
        if 'min_examples' in depth_config:
            constraints.append(f"Minimum examples: {depth_config['min_examples']}")
        
        if not constraints:
            return ""
        
        return f"""
## CONSTRAINTS
{chr(10).join(f"- {c}" for c in constraints)}
"""
    
    def _build_context_section(self, context: GenerationContext, depth_config: dict) -> str:
        """Build section with context from previous depths"""
        if not context.previous_depths or context.depth == 1:
            return ""
        
        # Extract concepts from previous depths
        previous_concepts = {}
        for prev_depth, prev_content in context.previous_depths.items():
            concepts = self._extract_key_concepts(prev_content)
            previous_concepts[prev_depth] = concepts
        
        context_instructions = depth_config.get('context_usage', {}).get('instructions', '')
        
        context_text = f"""
## CONTEXT FROM PREVIOUS DEPTHS

{context_instructions}

"""
        
        for prev_depth, concepts in previous_concepts.items():
            if concepts:
                context_text += f"""
Depth {prev_depth} covered these concepts:
{', '.join(concepts)}

"""
        
        context_text += """
CRITICAL ANTI-REPETITION INSTRUCTIONS:
- DO NOT restate or re-explain concepts from previous depths
- DO NOT use similar phrasing or examples as previous depths
- ASSUME the reader has already read previous depths
- BUILD ON previous concepts, don't repeat them
- Add NEW information, NEW perspectives, or NEW insights
"""
        
        return context_text
    
    def _extract_key_concepts(self, content: str) -> List[str]:
        """
        Extract key concepts from previous depth content.
        """
        if not content:
            return []
            
        concepts = set()
        
        # Extract quoted terms
        quoted = re.findall(r'"([^"]+)"', content)
        concepts.update(quoted[:5])  # Limit to avoid too many
        
        # Extract terms like "von Neumann entropy", "Wheeler-DeWitt equation"
        technical_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', content)
        concepts.update(technical_terms[:5])
        
        # Extract equation references
        equation_refs = re.findall(r'\b(?:equation|formula|principle|law|theorem)\s+\w+', content, re.IGNORECASE)
        concepts.update(equation_refs[:3])
        
        return list(concepts)[:10]  # Return top 10 concepts
    
    def _build_examples_section(self, depth_config: dict) -> str:
        """Build examples of good/bad content for this depth"""
        examples = depth_config.get('examples', {})
        if not examples:
            return ""
        
        text = "## EXAMPLES\n\n"
        
        if 'good' in examples:
            text += "✓ GOOD examples for this depth:\n"
            for ex in examples['good']:
                text += f"  - {ex}\n"
            text += "\n"
        
        if 'bad' in examples:
            text += "✗ BAD examples to avoid:\n"
            for ex in examples['bad']:
                text += f"  - {ex}\n"
        
        return text
    
    def _build_output_requirements(self, depth_config: dict) -> str:
        """Final instructions for output format"""
        return """
## OUTPUT REQUIREMENTS

Generate content that:
1. Directly answers the key question for this depth level
2. Matches the specified tone and complexity
3. Meets all quantitative requirements (equations, citations, examples)
4. Includes all required elements
5. Avoids all forbidden elements
6. Does NOT repeat content from previous depths

Write in clear, well-structured prose appropriate for the target audience.
"""

    def get_profile_names(self) -> List[str]:
        """Get list of available profile names"""
        return list(self.profiles.get('profiles', {}).keys())
    
    def get_max_depth(self, profile: str) -> int:
        """Get maximum depth available for a profile"""
        if profile not in self.profiles.get('profiles', {}):
            return 3  # Default
            
        profile_config = self.profiles['profiles'][profile]
        depths = [int(key.split('_')[1]) for key in profile_config.keys() 
                 if key.startswith('depth_')]
        
        return max(depths) if depths else 3