# Musequill v13

AI-powered creative writing system that orchestrates multiple specialized agents to generate long-form narrative content with similarity detection to prevent repetition.

## Architecture Overview

```mermaid
graph TB
    subgraph "External Services"
        Tavily[Tavily API<br/>Web Search]
    end
    
    subgraph "Infrastructure Layer (Docker)"
        Ollama[Ollama<br/>:11434<br/>LLM Service<br/>gpt-oss:120b]
        ChromaDB[ChromaDB<br/>:8000<br/>Vector Storage]
        Redis[Redis<br/>:16379<br/>Cache/State]
    end
    
    subgraph "Application Layer"
        Server[FastAPI Server<br/>:8080<br/>WebSocket]
        
        subgraph "Workflow Orchestration"
            Orchestrator[WorkflowOrchestrator<br/>LangGraph StateGraph]
            Memory[MemorySaver<br/>Checkpointing]
        end
        
        subgraph "Agent System"
            Generator[GeneratorAgent<br/>ReAct + Web Search]
            Discriminator[DiscriminatorAgent<br/>ReAct + Web Search]
            GenEvaluator[Generator Evaluator<br/>10 Metrics]
            DiscEvaluator[Discriminator Evaluator<br/>10 Metrics]
            Summarizer[SummarizerAgent<br/>Final Summary]
        end
        
        subgraph "Similarity Detection"
            SimilarityCorpus[SimilarityCorpus<br/>Shared Corpus]
            GenChecker[Generator<br/>SimilarityChecker]
            DiscChecker[Discriminator<br/>SimilarityChecker]
            Feedback[SimilarityFeedback<br/>Generator]
        end
        
        subgraph "Storage Management"
            ChromaManager[ChromaManager<br/>Chapter Storage]
            MemoryManager[MemoryManager<br/>Conversation History]
        end
    end
    
    subgraph "Data Flow"
        State[TopicFocusedState<br/>Workflow State]
        Output[Output Files<br/>conversation_log_*.md]
    end
    
    %% Client connections
    Client[WebSocket Client] -->|WS| Server
    
    %% Infrastructure connections
    Server --> Orchestrator
    Orchestrator --> Memory
    
    %% Agent workflow
    Orchestrator --> Generator
    Generator --> GenChecker
    GenChecker --> SimilarityCorpus
    Generator --> GenEvaluator
    GenEvaluator --> SimilarityCorpus
    
    Orchestrator --> Discriminator
    Discriminator --> DiscChecker
    DiscChecker --> SimilarityCorpus
    Discriminator --> DiscEvaluator
    DiscEvaluator --> SimilarityCorpus
    
    Orchestrator --> Summarizer
    
    %% External service connections
    Generator --> Tavily
    Discriminator --> Tavily
    
    %% LLM connections
    Generator --> Ollama
    Discriminator --> Ollama
    GenEvaluator --> Ollama
    DiscEvaluator --> Ollama
    Summarizer --> Ollama
    
    %% Storage connections
    SimilarityCorpus --> ChromaDB
    ChromaManager --> ChromaDB
    MemoryManager --> Redis
    
    %% State management
    Orchestrator --> State
    State --> Output
    
    %% Feedback loop
    Feedback --> GenChecker
    Feedback --> DiscChecker
    
    classDef infrastructure fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef agent fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef similarity fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef storage fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef external fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class Ollama,ChromaDB,Redis infrastructure
    class Generator,Discriminator,GenEvaluator,DiscEvaluator,Summarizer agent
    class SimilarityCorpus,GenChecker,DiscChecker,Feedback similarity
    class ChromaManager,MemoryManager storage
    class Tavily external
```

## Workflow Phases

```mermaid
graph LR
    subgraph "Phase 1: Generation"
        G1[Generator] --> GE1[Evaluate]
        GE1 -->|Quality < Threshold| GR1[Revise]
        GR1 --> G1
        GE1 -->|Quality Met| GS1[Store Best]
    end
    
    subgraph "Phase 2: Discrimination"
        D1[Discriminator] --> DE1[Evaluate]
        DE1 -->|Quality < Threshold| DR1[Revise]
        DR1 --> D1
        DE1 -->|Quality Met| DS1[Store Best]
    end
    
    subgraph "Phase 3: Iteration"
        IT1{Max Iterations?}
        IT1 -->|No| G1
        IT1 -->|Yes| S1[Summarize]
    end
    
    GS1 --> D1
    DS1 --> IT1
    S1 --> END[Output]
```

## Similarity Detection Flow

```mermaid
sequenceDiagram
    participant Agent
    participant SimilarityChecker
    participant SimilarityCorpus
    participant ChromaDB
    participant Feedback
    
    Agent->>Agent: Generate content
    Agent->>SimilarityChecker: check_similarity(content)
    SimilarityChecker->>SimilarityCorpus: search_similar_content()
    SimilarityCorpus->>ChromaDB: query vectors
    ChromaDB-->>SimilarityCorpus: similar paragraphs
    SimilarityCorpus-->>SimilarityChecker: ParagraphMatch[]
    
    alt Similarity > Threshold
        SimilarityChecker->>Feedback: generate_feedback()
        Feedback-->>SimilarityChecker: detailed feedback
        SimilarityChecker-->>Agent: SimilarityResult(is_unique=False)
        Agent->>Agent: Augment prompt with feedback
        Agent->>Agent: Retry generation (max 5 attempts)
    else Similarity < Threshold
        SimilarityChecker-->>Agent: SimilarityResult(is_unique=True)
        Agent->>SimilarityCorpus: store_content()
        SimilarityCorpus->>ChromaDB: store vectors
    end
```

## Key Components

### Infrastructure Services

- **Ollama** (port 11434): LLM service running gpt-oss:120b model with GPU support
- **ChromaDB** (port 8000): Vector database for similarity search and chapter storage
- **Redis** (port 16379): Cache and temporary state management

### Core Agents

1. **GeneratorAgent**: ReAct agent with Tavily web search for research-driven content creation
2. **DiscriminatorAgent**: Challenges and deepens the generator's content
3. **EvaluatorAgent**: Sophisticated evaluation using 10 metrics with domain-specific profiles
4. **SummarizerAgent**: Creates final summary of conversation

### Similarity Detection System

- **SimilarityCorpus**: Shared corpus storing best revisions from all agents
- **SimilarityChecker**: Checks new content against corpus (threshold: 0.85)
- **SimilarityFeedback**: Generates actionable feedback for repetitive content
- **Paragraph-level chunking**: Granular similarity detection
- **Sliding window search**: Optimized for large corpus (>100 chunks)

### Evaluation Metrics

1. Conceptual Novelty Rate (CNR)
2. Claim Density
3. Mathematical Rigor Index (MRI)
4. Structural Compression Ratio (SCR)
5. Citation/Reference Metrics
6. Empirical Grounding Score
7. Structural Coherence
8. Notation Consistency
9. Figure/Table Utility
10. Theoretical Parsimony

## Installation

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA support (for Ollama)
- Python 3.10+
- Tavily API key

### Setup

1. Clone the repository:
```bash
git clone <repository>
cd musequill.v13
```

2. Copy environment configuration:
```bash
cp .env.example .env
# Edit .env with your configurations
```

3. Start infrastructure services:
```bash
docker-compose up -d
```

4. Install Python dependencies:
```bash
pip install -r requirements.txt
```

5. Run the application:
```bash
python main.py
```

## Configuration

### Environment Variables

```bash
# Server
SERVER_HOST=localhost
SERVER_PORT=8080

# Models
GENERATOR_MODEL=gpt-oss:120b
DISCRIMINATOR_MODEL=gpt-oss:120b

# Evaluation
QUALITY_THRESHOLD=75.0
MAX_REFINEMENT_ITERATIONS=3

# Similarity Detection
SIMILARITY_THRESHOLD=0.85
SIMILARITY_RELAXED_THRESHOLD=0.90
MAX_SIMILARITY_ATTEMPTS=5
PARAGRAPH_MIN_LENGTH=50
SLIDING_WINDOW_SIZE=3
SLIDING_WINDOW_ACTIVATION_THRESHOLD=100

# External Services
TAVILY_API_KEY=your_api_key
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_HOST=localhost
CHROMA_PORT=8000
REDIS_URL=redis://localhost:16379
```

## Usage

### WebSocket API

Connect to `ws://localhost:8080/ws` and send:

```json
{
  "type": "content_request",
  "data": {
    "topic": "Your topic here",
    "max_iterations": 3
  },
  "workflow": "orchestrator"
}
```

### Testing Similarity Detection

Run the test script:
```bash
python test_similarity.py
```

## Development

### Project Structure

```
musequill.v13/
├── src/
│   ├── agents/           # Agent implementations
│   ├── common/           # Shared data structures
│   ├── exceptions/       # Custom exceptions
│   ├── llm/             # LLM client wrapper
│   ├── prompts/         # Agent prompts
│   ├── server/          # FastAPI server
│   ├── states/          # Workflow states
│   ├── storage/         # Storage managers
│   ├── utils/           # Utility functions
│   └── workflow/        # Orchestration logic
├── outputs/             # Generated content
├── docker-compose.yml   # Infrastructure services
├── main.py             # Application entry point
└── test_similarity.py  # Similarity testing
```

### Adding New Agents

1. Extend `BaseAgent` class
2. Implement `process()` method
3. Add similarity checking if generating content
4. Register in orchestrator workflow

### Monitoring

- Server logs: `http://localhost:8080/health`
- ChromaDB: `http://localhost:8000/api/v1/heartbeat`
- Ollama: `http://localhost:11434/api/tags`
- Redis: `redis-cli -p 16379 ping`

## License

[Your License Here]

## Contributing

[Contributing guidelines]