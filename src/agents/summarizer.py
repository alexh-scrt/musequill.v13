import logging
import os
from typing import Optional, Dict, Any, List, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from src.agents.base import BaseAgent
from src.agents.evaluator import EvaluatorAgent
from src.states.topic_focused import TopicFocusedState
from src.prompts.summarizer_prompts import SUMMARIZER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class SummarizerAgent(BaseAgent):
    """
    Agent that synthesizes conversations into powerful summaries with quality assessment.
    
    Responsibilities:
    - Synthesize multi-turn conversations into coherent summaries
    - Extract key insights and themes
    - Assess depth and quality of exploration
    - Provide actionable conclusions
    - Identify gaps and future directions
    """
    
    def __init__(
        self, 
        agent_id: str = "summarizer", 
        model: Optional[str] = None, 
        session_id: Optional[str] = None, 
        llm_params: Optional[Dict[str, Any]] = None,
        evaluator_profile: str = "general"
    ):
        super().__init__(
            agent_id, 
            web_search=False,  # Summarizer doesn't need web search
            model=model, 
            session_id=session_id, 
            llm_params=llm_params or {"temperature": 0.3}  # Lower temp for consistency
        )
        
        # Initialize evaluator for quality assessment
        self.evaluator = EvaluatorAgent(
            session_id=session_id,
            llm_params={"temperature": 0.3},
            profile=evaluator_profile
        )
        
        logger.info(f"SummarizerAgent initialized with model: {self.model}")
    
    async def process(
        self, 
        prompt: str = "", 
        state: Optional[TopicFocusedState] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate comprehensive summary and assessment of conversation.
        
        Args:
            prompt: Optional prompt (usually not needed for summarization)
            state: Conversation state with full history
            
        Returns:
            Tuple of (summary_text, state_updates)
        """
        if state is None:
            state = {}
        
        logger.info("📊 Starting conversation summarization and assessment...")
        
        # Extract conversation history
        conversation_history = await self.get_recent_context(n=50)  # Get full history
        
        original_topic = state.get('original_topic', 'Unknown topic')
        depth_level = state.get('current_depth_level', 1)
        aspects_explored = state.get('aspects_explored', [])
        iterations = state.get('iterations', 0)
        
        # Phase 1: Extract key themes and insights
        logger.info("🔍 Extracting key themes and insights...")
        themes_insights = await self._extract_themes_and_insights(
            conversation_history, 
            original_topic
        )
        
        # Phase 2: Synthesize main points
        logger.info("📝 Synthesizing main discussion points...")
        main_points = await self._synthesize_main_points(
            conversation_history,
            original_topic,
            themes_insights
        )
        
        # Phase 3: Assess conversation quality
        logger.info("⚖️ Assessing conversation quality...")
        quality_assessment = await self._assess_conversation_quality(
            conversation_history,
            original_topic,
            depth_level,
            iterations
        )
        
        # Phase 4: Identify gaps and future directions
        logger.info("🔭 Identifying gaps and future directions...")
        gaps_future = await self._identify_gaps_and_future(
            conversation_history,
            original_topic,
            aspects_explored
        )
        
        # Phase 5: Generate final summary
        logger.info("✨ Generating final comprehensive summary...")
        final_summary = await self._generate_final_summary(
            original_topic=original_topic,
            themes_insights=themes_insights,
            main_points=main_points,
            quality_assessment=quality_assessment,
            gaps_future=gaps_future,
            iterations=iterations,
            depth_level=depth_level
        )
        
        # Save to memory
        await self.add_to_history("summarizer", final_summary)
        
        state_updates = {
            'final_summary': final_summary,
            'themes': themes_insights.get('themes', []),
            'quality_score': quality_assessment.get('overall_score', 0),
            'gaps_identified': gaps_future.get('gaps', [])
        }
        
        logger.info("✅ Summarization complete")
        return final_summary, state_updates
    
    async def _extract_themes_and_insights(
        self,
        conversation_history: List[Dict],
        original_topic: str
    ) -> Dict[str, Any]:
        """Extract recurring themes and key insights from conversation."""
        
        # Format conversation for analysis
        conversation_text = self._format_conversation_for_analysis(conversation_history)
        
        prompt = f"""
        Analyze the following conversation about "{original_topic}" and extract:
        
        1. **Recurring Themes** (3-5 major themes that emerged)
        2. **Key Insights** (5-7 most important discoveries or realizations)
        3. **Conceptual Depth** (How deeply was each aspect explored?)
        
        Conversation:
        {conversation_text}
        
        Provide your analysis in a structured format:
        
        THEMES:
        - Theme 1: [description]
        - Theme 2: [description]
        ...
        
        KEY INSIGHTS:
        - Insight 1: [description]
        - Insight 2: [description]
        ...
        
        DEPTH ANALYSIS:
        - Aspect: [name] | Depth: [surface/moderate/deep] | Coverage: [brief description]
        """
        
        analysis = await self.generate_with_llm(
            prompt, 
            system_prompt=SUMMARIZER_SYSTEM_PROMPT
        )
        
        # Parse the analysis (simple parsing for now)
        themes = self._extract_section(analysis, "THEMES:")
        insights = self._extract_section(analysis, "KEY INSIGHTS:")
        depth = self._extract_section(analysis, "DEPTH ANALYSIS:")
        
        return {
            'themes': themes,
            'insights': insights,
            'depth_analysis': depth,
            'raw_analysis': analysis
        }
    
    async def _synthesize_main_points(
        self,
        conversation_history: List[Dict],
        original_topic: str,
        themes_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize the main discussion points into a coherent narrative."""
        
        conversation_text = self._format_conversation_for_analysis(conversation_history)
        themes = '\n'.join(themes_insights.get('themes', []))
        
        prompt = f"""
        Given the conversation about "{original_topic}" and the identified themes:
        
        THEMES:
        {themes}
        
        Synthesize the main discussion points into 5-7 key takeaways. Each takeaway should:
        - Capture a core idea explored in the conversation
        - Be self-contained and clear
        - Show progression from basic to advanced understanding
        
        Format as:
        
        MAIN TAKEAWAYS:
        1. [First key point]
        2. [Second key point]
        ...
        
        PROGRESSION:
        - Started with: [initial understanding]
        - Evolved through: [intermediate stages]
        - Culminated in: [final insights]
        """
        
        synthesis = await self.generate_with_llm(
            prompt,
            system_prompt=SUMMARIZER_SYSTEM_PROMPT
        )
        
        takeaways = self._extract_section(synthesis, "MAIN TAKEAWAYS:")
        progression = self._extract_section(synthesis, "PROGRESSION:")
        
        return {
            'takeaways': takeaways,
            'progression': progression,
            'raw_synthesis': synthesis
        }
    
    async def _assess_conversation_quality(
        self,
        conversation_history: List[Dict],
        original_topic: str,
        depth_level: int,
        iterations: int
    ) -> Dict[str, Any]:
        """Assess the quality of the conversation along multiple dimensions."""
        
        # Count turns by each agent
        generator_turns = len([t for t in conversation_history if t.get('role') == 'user'])
        discriminator_turns = len([t for t in conversation_history if t.get('role') == 'discriminator'])
        
        # Reconstruct full conversation text for evaluation
        full_conversation = '\n\n'.join([
            f"{turn.get('role', 'unknown').upper()}: {turn.get('content', '')}"
            for turn in conversation_history
        ])
        
        # Use evaluator for quantitative assessment
        eval_result = await self.evaluator.evaluate(
            content=full_conversation,
            previous_content=None,
            context={'type': 'conversation'}
        )
        
        # Qualitative assessment
        prompt = f"""
        Assess the quality of this conversation about "{original_topic}":
        
        Metrics:
        - Total turns: {iterations}
        - Generator turns: {generator_turns}
        - Discriminator turns: {discriminator_turns}
        - Depth level reached: {depth_level}/5
        - Quantitative score: {eval_result.total_score:.1f}/100
        
        Provide qualitative assessment on:
        1. **Topic Adherence**: Did the conversation stay focused on the original topic?
        2. **Depth of Exploration**: How thoroughly was the topic explored?
        3. **Intellectual Rigor**: Were claims supported? Was reasoning sound?
        4. **Engagement Quality**: Did the agents build on each other's contributions?
        5. **Practical Value**: Are there actionable insights or learnings?
        
        Rate each dimension: Excellent / Good / Adequate / Poor
        Provide brief justification for each rating.
        """
        
        qualitative = await self.generate_with_llm(
            prompt,
            system_prompt=SUMMARIZER_SYSTEM_PROMPT
        )
        
        # Calculate overall score
        # Weight: 60% quantitative (evaluator) + 40% qualitative factors
        qualitative_score = self._parse_qualitative_score(qualitative)
        overall_score = (eval_result.total_score * 0.6) + (qualitative_score * 0.4)
        
        return {
            'overall_score': overall_score,
            'quantitative_score': eval_result.total_score,
            'qualitative_score': qualitative_score,
            'tier': eval_result.tier,
            'evaluation_details': eval_result,
            'qualitative_assessment': qualitative,
            'metrics': {
                'total_turns': iterations,
                'generator_turns': generator_turns,
                'discriminator_turns': discriminator_turns,
                'depth_level': depth_level
            }
        }
    
    async def _identify_gaps_and_future(
        self,
        conversation_history: List[Dict],
        original_topic: str,
        aspects_explored: List[str]
    ) -> Dict[str, Any]:
        """Identify gaps in coverage and suggest future directions."""
        
        conversation_text = self._format_conversation_for_analysis(conversation_history)
        explored = ', '.join(aspects_explored) if aspects_explored else 'None explicitly tracked'
        
        prompt = f"""
        Given the conversation about "{original_topic}":
        
        Aspects explored: {explored}
        
        Identify:
        
        1. **Gaps in Coverage** (3-5 important aspects that were NOT adequately covered)
        2. **Unexplored Angles** (Alternative perspectives or approaches that could be valuable)
        3. **Future Directions** (3-5 specific questions or topics for deeper exploration)
        4. **Practical Applications** (How could these insights be applied?)
        
        Format:
        
        GAPS:
        - Gap 1: [description]
        - Gap 2: [description]
        ...
        
        UNEXPLORED ANGLES:
        - Angle 1: [description]
        ...
        
        FUTURE DIRECTIONS:
        - Direction 1: [specific question or topic]
        - Direction 2: [specific question or topic]
        ...
        
        PRACTICAL APPLICATIONS:
        - Application 1: [description]
        ...
        """
        
        analysis = await self.generate_with_llm(
            prompt,
            system_prompt=SUMMARIZER_SYSTEM_PROMPT
        )
        
        gaps = self._extract_section(analysis, "GAPS:")
        angles = self._extract_section(analysis, "UNEXPLORED ANGLES:")
        future = self._extract_section(analysis, "FUTURE DIRECTIONS:")
        applications = self._extract_section(analysis, "PRACTICAL APPLICATIONS:")
        
        return {
            'gaps': gaps,
            'unexplored_angles': angles,
            'future_directions': future,
            'practical_applications': applications,
            'raw_analysis': analysis
        }
    
    async def _generate_final_summary(
        self,
        original_topic: str,
        themes_insights: Dict[str, Any],
        main_points: Dict[str, Any],
        quality_assessment: Dict[str, Any],
        gaps_future: Dict[str, Any],
        iterations: int,
        depth_level: int
    ) -> str:
        """Generate the final comprehensive summary document."""
        
        summary_parts = []
        
        # Header
        summary_parts.append("-" * 60)
        summary_parts.append('\n')
        summary_parts.append(f"# CONVERSATION SUMMARY")
        summary_parts.append('\n')
        summary_parts.append(f"**Topic**: {original_topic}")
        summary_parts.append('\n')
        
        # Overview
        summary_parts.append("## OVERVIEW")
        summary_parts.append(f"- Topic: {original_topic}")
        summary_parts.append(f"- Conversation turns: {iterations}")
        summary_parts.append(f"- Depth level reached: {depth_level}/5")
        summary_parts.append(f"- Overall quality score: {quality_assessment['overall_score']:.1f}/100 ({quality_assessment['tier']})")
        summary_parts.append("")
        
        # Key Themes
        summary_parts.append("## KEY THEMES")
        for theme in themes_insights.get('themes', [])[:5]:
            summary_parts.append(f"  {theme}")
        summary_parts.append("")
        
        # Main Takeaways
        summary_parts.append("## MAIN TAKEAWAYS")
        for takeaway in main_points.get('takeaways', [])[:7]:
            summary_parts.append(f"  {takeaway}")
        summary_parts.append("")
        
        # Key Insights
        summary_parts.append("## KEY INSIGHTS")
        for insight in themes_insights.get('insights', [])[:7]:
            summary_parts.append(f"  {insight}")
        summary_parts.append("")
        
        # Intellectual Progression
        summary_parts.append("## INTELLECTUAL PROGRESSION")
        for prog in main_points.get('progression', [])[:3]:
            summary_parts.append(f"  {prog}")
        summary_parts.append("")
        
        # Quality Assessment
        summary_parts.append("## QUALITY ASSESSMENT")
        summary_parts.append(f"**Overall Score: {quality_assessment['overall_score']:.1f}/100**")
        summary_parts.append("")
        summary_parts.append("**Quantitative Metrics:**")
        metrics = quality_assessment['evaluation_details'].metrics
        
        # Access MetricScore objects properly
        if 'conceptual_novelty' in metrics:
            summary_parts.append(f"- Conceptual Novelty: {metrics['conceptual_novelty'].percentage:.0f}%")
        if 'claim_density' in metrics:
            summary_parts.append(f"- Claim Density: {metrics['claim_density'].percentage:.0f}%")
        if 'structural_coherence' in metrics:
            summary_parts.append(f"- Structural Coherence: {metrics['structural_coherence'].percentage:.0f}%")
        summary_parts.append("")
        summary_parts.append("**Qualitative Assessment:**")
        # Extract key lines from qualitative assessment
        qual_lines = quality_assessment.get('qualitative_assessment', '').split('\n')[:10]
        for line in qual_lines:
            if line.strip():
                summary_parts.append(f"  {line.strip()}")
        summary_parts.append("")
        
        # Gaps and Future Directions
        summary_parts.append("## GAPS IN COVERAGE")
        for gap in gaps_future.get('gaps', [])[:5]:
            summary_parts.append(f"  {gap}")
        summary_parts.append("")
        
        summary_parts.append("## FUTURE DIRECTIONS")
        for direction in gaps_future.get('future_directions', [])[:5]:
            summary_parts.append(f"  {direction}")
        summary_parts.append("")
        
        # Practical Applications
        summary_parts.append("## PRACTICAL APPLICATIONS")
        for app in gaps_future.get('practical_applications', [])[:5]:
            summary_parts.append(f"  {app}")
        summary_parts.append("")
        
        # Conclusion
        summary_parts.append("## CONCLUSION")
        summary_parts.append(f"This conversation successfully explored {original_topic} across {iterations} turns, ")
        summary_parts.append(f"reaching depth level {depth_level}/5. The discussion achieved a quality score of ")
        summary_parts.append(f"{quality_assessment['overall_score']:.1f}/100, rated as '{quality_assessment['tier']}'.")
        summary_parts.append("")
        
        if quality_assessment['overall_score'] >= 75:
            summary_parts.append("The conversation demonstrated strong intellectual rigor, staying focused on the ")
            summary_parts.append("original topic while exploring multiple facets with appropriate depth. The insights ")
            summary_parts.append("generated are valuable and the progression shows clear advancement in understanding.")
        elif quality_assessment['overall_score'] >= 60:
            summary_parts.append("The conversation provided good coverage of the topic with reasonable depth. ")
            summary_parts.append("There are opportunities to explore some aspects more thoroughly in future discussions.")
        else:
            summary_parts.append("The conversation covered basic aspects of the topic but could benefit from deeper ")
            summary_parts.append("exploration. Consider revisiting with more focused questions on the identified gaps.")
        
        summary_parts.append("")
        summary_parts.append("=" * 80)
        
        return "\n".join(summary_parts)
    
    def _format_conversation_for_analysis(
        self, 
        conversation_history: List[Dict],
        max_chars: int = 4000
    ) -> str:
        """Format conversation history for LLM analysis."""
        formatted = []
        total_chars = 0
        
        for turn in conversation_history:
            role = turn.get('role', 'unknown').upper()
            content = turn.get('content', '')
            
            # Truncate if too long
            if total_chars + len(content) > max_chars:
                remaining = max_chars - total_chars
                content = content[:remaining] + "..."
                formatted.append(f"{role}: {content}")
                break
            
            formatted.append(f"{role}: {content}")
            total_chars += len(content)
        
        return "\n\n".join(formatted)
    
    def _extract_section(self, text: str, header: str) -> List[str]:
        """Extract bullet points from a section with the given header."""
        lines = text.split('\n')
        in_section = False
        items = []
        
        for line in lines:
            if header in line:
                in_section = True
                continue
            
            if in_section:
                # Check if we've hit another section header (all caps + colon)
                if line.strip() and line.strip().isupper() and ':' in line:
                    break
                
                # Extract bullet points
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or 
                           line[0].isdigit() and '.' in line[:3]):
                    # Clean up the bullet/number
                    cleaned = line.lstrip('-•0123456789. ').strip()
                    if cleaned:
                        items.append(cleaned)
        
        return items
    
    def _parse_qualitative_score(self, qualitative_text: str) -> float:
        """Parse qualitative assessment into a numeric score (0-100)."""
        # Simple heuristic: count Excellent/Good/Adequate/Poor ratings
        text_lower = qualitative_text.lower()
        
        excellent = text_lower.count('excellent')
        good = text_lower.count('good')
        adequate = text_lower.count('adequate')
        poor = text_lower.count('poor')
        
        # Weight: Excellent=100, Good=75, Adequate=50, Poor=25
        total_ratings = excellent + good + adequate + poor
        
        if total_ratings == 0:
            return 75.0  # Default to "Good"
        
        weighted_sum = (excellent * 100 + good * 75 + adequate * 50 + poor * 25)
        return weighted_sum / total_ratings