# Key changes to WorkflowOrchestrator __init__ method:

def __init__(
    self, 
    session_id: Optional[str] = None, 
    generator_profile: str = "balanced", 
    discriminator_profile: str = "balanced",
    evaluator_profile: str = "general"  # NEW PARAMETER
):
    """
    Initialize workflow orchestrator with agent profiles.
    
    Args:
        session_id: Session identifier
        generator_profile: Generator LLM profile (creative/balanced/conservative)
        discriminator_profile: Discriminator LLM profile
        evaluator_profile: Evaluator domain profile (scientific/technology/etc.)
    """
    self.session_id = session_id or f"session_{uuid.uuid4().hex[:12]}"
    
    # Get LLM parameters from profile factories
    gen_params = GeneratorProfileFactory.get(generator_profile)
    disc_params = DiscriminatorProfileFactory.get(discriminator_profile)
    
    logger.info(f"Generator profile: {generator_profile}")
    logger.info(f"Discriminator profile: {discriminator_profile}")
    logger.info(f"Evaluator profile: {evaluator_profile}")
    
    self.generator = GeneratorAgent(
        session_id=self.session_id, 
        llm_params=gen_params
    )
    
    self.discriminator = DiscriminatorAgent(
        session_id=self.session_id, 
        llm_params=disc_params
    )        

    # Initialize evaluators with profile
    self.generator_evaluator = EvaluatorAgent(
        session_id=self.session_id,
        profile=evaluator_profile  # Pass profile here
    )

    self.discriminator_evaluator = EvaluatorAgent(
        session_id=self.session_id,
        profile=evaluator_profile  # Pass profile here
    )

    self.summarizer = SummarizerAgent(
        session_id=self.session_id,
        llm_params={'temperature': 0.3}
    )

    # ... rest of initialization ...


# Update run_async signature to accept profile:

async def run_async(
    self,
    topic: str,
    max_iterations: int = 3,
    quality_threshold: float = 75.0,
    evaluator_profile: str = "general"  # NEW PARAMETER (optional override)
) -> AsyncGenerator[AgentResponse, None]:
    """
    Run the orchestrated workflow with streaming responses.
    
    Args:
        topic: The topic to discuss
        max_iterations: Maximum conversation iterations
        quality_threshold: Quality score threshold
        evaluator_profile: Override evaluator profile for this run (optional)
    """
    
    # If profile override provided, reinitialize evaluators
    if evaluator_profile != self.generator_evaluator.profile:
        logger.info(f"Overriding evaluator profile to: {evaluator_profile}")
        
        self.generator_evaluator = EvaluatorAgent(
            session_id=self.session_id,
            profile=evaluator_profile
        )
        
        self.discriminator_evaluator = EvaluatorAgent(
            session_id=self.session_id,
            profile=evaluator_profile
        )
    
    # ... rest of run_async implementation ...