# Implementation Instructions for Claude Code

## Overview
Implement a similarity detection system that prevents content repetition in the Musequill conversational workflow. The system checks newly generated content against a shared corpus of previously accepted revisions and provides detailed feedback when repetition is detected.

---

## File Structure

### New Files to Create

1. **`src/agents/similarity_checker.py`**
   - Core similarity detection logic
   - Feedback generation
   - Paragraph-level comparison

2. **`src/storage/similarity_corpus.py`**
   - ChromaDB wrapper for similarity corpus
   - Chunking and storage logic
   - Vector search interface

3. **`src/agents/similarity_feedback.py`**
   - Feedback generation utilities
   - Pattern matching for naturally repetitive content
   - Feedback formatting

4. **`src/exceptions/similarity.py`**
   - Custom exceptions (TopicExhaustionError, etc.)

### Files to Modify

1. **`src/agents/generator.py`**
   - Add similarity checking loop in `process()` method
   - Integrate `_generate_unique_content()` method
   - Add prompt augmentation with similarity feedback

2. **`src/agents/discriminator.py`**
   - Same changes as generator.py (parallel implementation)

3. **`src/agents/evaluator.py`**
   - Add method to store best revision in similarity corpus
   - Call storage after evaluation completes

4. **`src/workflow/orchestrator.py`**
   - Update `_generator_evaluator()` to store best revision
   - Update `_discriminator_evaluator()` to store best revision
   - Initialize shared similarity corpus in `__init__`

5. **`.env.example`** and **`.env`**
   - Add similarity-related configuration variables

---

## Detailed Implementation Steps

### Step 1: Create Exception Classes

**File:** `src/exceptions/similarity.py`

**Instructions:**
- Create a new file `src/exceptions/similarity.py`
- Define `TopicExhaustionError(Exception)` class
  - Should accept message and optional metadata (attempts, best_similarity)
  - Include `__str__` method for logging
- Define `SimilarityCorpusError(Exception)` for storage errors
- Add docstrings explaining when each exception is raised

**Acceptance Criteria:**
- Exceptions are properly typed with Optional metadata
- Clear error messages for debugging

---

### Step 2: Create Data Classes

**File:** `src/agents/similarity_checker.py` (top of file)

**Instructions:**
- Import necessary libraries: `dataclasses`, `typing`, `logging`
- Define `@dataclass ParagraphMatch`:
  ```
  - query_paragraph: str
  - matched_paragraph: str
  - similarity_score: float
  - stored_content_id: str
  - paragraph_index: int
  - matched_index: int
  ```
- Define `@dataclass SimilarityResult`:
  ```
  - overall_similarity: float
  - is_unique: bool
  - paragraph_matches: List[ParagraphMatch]
  - unique_paragraphs: List[int]
  - feedback: str
  - attempts_remaining: int
  ```
- Add type hints and docstrings to all fields

**Acceptance Criteria:**
- Data classes are frozen (immutable)
- All fields have type annotations
- Docstrings explain each field's purpose

---

### Step 3: Implement SimilarityCorpus

**File:** `src/storage/similarity_corpus.py`

**Instructions:**

1. **Class initialization:**
   - Constructor accepts `session_id: str`
   - Connect to ChromaDB (reuse pattern from `chroma_manager.py`)
   - Create collection: `f"similarity_corpus_{session_id}"`
   - Use `_sanitize_collection_name()` method (copy from `chroma_manager.py`)
   - Initialize logger

2. **Method: `store_content(content: str, agent_id: str, metadata: Dict) -> str`:**
   - Split content into paragraphs using `_split_paragraphs()`
   - Generate unique ID: `f"{agent_id}_iter{iteration}_rev{revision}"`
   - Store full content as one document with metadata
   - Store each paragraph as a separate chunked document with:
     - `parent_id`: the full content ID
     - `chunk_index`: paragraph position
     - `chunk_text`: paragraph content
     - `is_chunk`: True
     - `agent_id`: passed from caller
   - Log storage success with content ID
   - Return content ID

3. **Method: `_split_paragraphs(content: str) -> List[str]`:**
   - Split on `\n\n`
   - Filter out paragraphs < 50 characters (use `PARAGRAPH_MIN_LENGTH` env var)
   - Strip whitespace from each paragraph
   - Return list of paragraph strings

4. **Method: `search_similar_paragraphs(paragraph: str, threshold: float = 0.85) -> List[Dict]`:**
   - Use ChromaDB's `query()` method to find similar chunks
   - Query parameters:
     - `query_texts=[paragraph]`
     - `n_results=5`
     - `where={"is_chunk": True}`
     - `include=["documents", "metadatas", "distances"]`
   - Filter results where similarity > threshold
   - Convert distance to similarity: `similarity = 1 - distance`
   - Return list of matches with metadata

5. **Method: `search_similar_content(content: str) -> List[ParagraphMatch]`:**
   - Split content into paragraphs using `_split_paragraphs()`
   - For each paragraph:
     - Call `search_similar_paragraphs(paragraph)`
     - Convert to `ParagraphMatch` objects
   - Check if sliding window needed using `should_use_sliding_window()`
   - If sliding window activated:
     - Use `_sliding_window_search()` instead
   - Return all matches

6. **Method: `should_use_sliding_window() -> bool`:**
   - Query collection for total chunk count
   - Use ChromaDB's `count()` method with `where={"is_chunk": True}`
   - Return `True` if count > `SLIDING_WINDOW_ACTIVATION_THRESHOLD` (default 100)

7. **Method: `_sliding_window_search(paragraphs: List[str], window_size: int = 3) -> List[ParagraphMatch]`:**
   - Create sliding windows of `window_size` paragraphs
   - Concatenate each window into a single string
   - Search for similarity of concatenated windows
   - Map results back to individual paragraphs
   - Return matches

8. **Method: `get_corpus_stats() -> Dict`:**
   - Return:
     - `total_documents`: count of non-chunk documents
     - `total_chunks`: count of chunk documents
     - `agents`: unique agent_ids in corpus
     - `oldest_timestamp`: earliest timestamp in corpus
   - Use ChromaDB's `get()` method with metadata filtering

9. **Method: `clear_session()`:**
   - Delete collection for this session
   - Use ChromaDB's `delete_collection()` method
   - Log cleanup

**Acceptance Criteria:**
- All methods have proper error handling (try/except)
- Logging at INFO level for storage, DEBUG for searches
- Type hints on all methods
- Docstrings with Args/Returns sections
- Unit testable (dependency injection for ChromaDB client)

---

### Step 4: Implement SimilarityFeedback Generator

**File:** `src/agents/similarity_feedback.py`

**Instructions:**

1. **Define constants at module level:**
   ```python
   IGNORE_PATTERNS = [
       r'^As (discussed|mentioned|stated) (earlier|previously|before)',
       r'^(Recall|Remember) (that|from|how)',
       r'^In summary',
       r'^To recap',
       r'^Follow-up:',
   ]
   ```

2. **Function: `is_naturally_repetitive(paragraph: str) -> bool`:**
   - Check if paragraph matches any IGNORE_PATTERNS
   - Use `re.match()` with `re.IGNORECASE`
   - Return True if matches any pattern

3. **Function: `filter_natural_repetitions(matches: List[ParagraphMatch]) -> List[ParagraphMatch]`:**
   - Filter out matches where `query_paragraph` is naturally repetitive
   - Return filtered list

4. **Function: `generate_similarity_feedback(matches: List[ParagraphMatch]) -> str`:**
   - Filter natural repetitions first
   - If no matches remain, return empty string
   - Categorize matches:
     - High similarity: `>= 0.90`
     - Medium similarity: `0.85 <= score < 0.90`
   - Build feedback string with sections:
     - Header with count
     - "CRITICAL - Near-duplicate paragraphs" (top 3 high similarity)
     - "Moderately similar paragraphs" (top 2 medium)
     - "Suggestions to avoid repetition" (bullet list)
   - Format each match:
     ```
     {index}. Paragraph {match.paragraph_index+1}: {match.similarity_score:.0%} similar
        Preview: {match.query_paragraph[:100]}...
        ➜ REWRITE this section with a fresh perspective.
     ```
   - Return formatted feedback string

5. **Function: `augment_prompt_with_feedback(original_prompt: str, feedback: str) -> str`:**
   - Prepend feedback to original prompt
   - Add instructions:
     - "IMPORTANT: Your previous response was too similar to earlier content."
     - Include the detailed feedback
     - "Please regenerate with: [bullet list of suggestions]"
   - Return augmented prompt

**Acceptance Criteria:**
- Feedback is human-readable and actionable
- Preview snippets are truncated appropriately
- Pattern matching is case-insensitive
- Function has unit tests

---

### Step 5: Implement SimilarityChecker

**File:** `src/agents/similarity_checker.py`

**Instructions:**

1. **Class initialization:**
   - Constructor accepts `session_id: str` and `corpus: SimilarityCorpus`
   - Store references
   - Initialize logger
   - Load threshold from env: `SIMILARITY_THRESHOLD` (default 0.85)
   - Load max attempts from env: `MAX_SIMILARITY_ATTEMPTS` (default 5)

2. **Method: `async check_similarity(content: str) -> SimilarityResult`:**
   - Call `corpus.search_similar_content(content)`
   - Get list of `ParagraphMatch` objects
   - Filter natural repetitions using `filter_natural_repetitions()`
   - Calculate `overall_similarity`: max similarity score from matches
   - Determine `is_unique`: `overall_similarity < self.threshold`
   - Identify `unique_paragraphs`: indices not in matches
   - Generate feedback using `generate_similarity_feedback(matches)`
   - Return `SimilarityResult` with all fields populated

3. **Method: `get_unique_paragraph_indices(content: str, matches: List[ParagraphMatch]) -> List[int]`:**
   - Split content into paragraphs
   - Create set of matched indices from matches
   - Return list of indices NOT in matched set

4. **Property: `attempts_remaining`:**
   - Track attempts across multiple checks
   - Decrement on each check
   - Raise `TopicExhaustionError` if attempts exhausted

**Acceptance Criteria:**
- All async methods use `await` properly
- Error handling for corpus failures
- Logging at appropriate levels
- Type hints throughout

---

### Step 6: Modify Generator Agent

**File:** `src/agents/generator.py`

**Instructions:**

1. **Update `__init__` method:**
   - After calling `super().__init__()`, add:
   - Initialize `SimilarityCorpus(session_id)` as `self.similarity_corpus`
   - Initialize `SimilarityChecker(session_id, self.similarity_corpus)` as `self.similarity_checker`
   - Load `MAX_SIMILARITY_ATTEMPTS` from env (default 5)
   - Load `SIMILARITY_RELAXED_THRESHOLD` from env (default 0.90)

2. **Update `process()` method:**
   - Before existing logic, add:
   - Call `content, attempts = await self._generate_unique_content(prompt, state)`
   - Log attempts used
   - Continue with existing logic using the unique content

3. **Add new method: `async _generate_unique_content(prompt: str, state: dict) -> Tuple[str, int]`:**
   - Initialize:
     ```python
     attempts = 0
     best_content = None
     best_similarity = 1.0
     current_prompt = prompt
     ```
   - Loop while `attempts < self.max_similarity_attempts`:
     - Generate content: Call existing LLM generation logic
     - Strip think tags using `strip_think()`
     - Check similarity: `result = await self.similarity_checker.check_similarity(content)`
     - Track best: Update `best_content` and `best_similarity` if lower
     - If `result.is_unique`:
       - Log success with similarity score
       - Return `(content, attempts)`
     - Else:
       - Log warning with similarity score and attempt number
       - Augment prompt: `current_prompt = self._augment_prompt_with_similarity_feedback(prompt, result)`
       - Increment attempts
   - After loop (max attempts reached):
     - Log warning about max attempts
     - Check if `best_similarity < self.similarity_relaxed_threshold`:
       - If yes: Log acceptance with relaxed threshold, return best_content
       - If no: Log error about topic exhaustion, return best_content anyway (don't raise exception per requirements)
   - Return `(best_content, attempts)`

4. **Add new method: `_augment_prompt_with_similarity_feedback(original_prompt: str, result: SimilarityResult) -> str`:**
   - Import from `similarity_feedback` module
   - Call `augment_prompt_with_feedback(original_prompt, result.feedback)`
   - Return augmented prompt

**Acceptance Criteria:**
- No breaking changes to existing revision/iteration logic
- Similarity checking is invisible to evaluator
- Proper async/await usage
- Logging shows similarity attempts
- Graceful degradation when max attempts reached

---

### Step 7: Modify Discriminator Agent

**File:** `src/agents/discriminator.py`

**Instructions:**
- Apply **identical changes** as Step 6 to `DiscriminatorAgent`
- Same `__init__` additions
- Same `process()` modifications
- Same `_generate_unique_content()` method
- Same `_augment_prompt_with_similarity_feedback()` method
- Copy-paste approach is acceptable here for consistency

**Acceptance Criteria:**
- Discriminator has parity with Generator
- No code duplication via inheritance if possible (consider extracting to mixin)

---

### Step 8: Modify Evaluator Agent

**File:** `src/agents/evaluator.py`

**Instructions:**

1. **Update `__init__` method:**
   - Add parameter: `similarity_corpus: Optional[SimilarityCorpus] = None`
   - If `session_id` provided and `similarity_corpus` is None:
     - Initialize `self.similarity_corpus = SimilarityCorpus(session_id)`
   - Else:
     - Store provided corpus: `self.similarity_corpus = similarity_corpus`

2. **Add new method: `async store_best_revision(content: str, agent_id: str, metadata: Dict)`:**
   - Check if `self.similarity_corpus` exists
   - If not, log warning and return early
   - Prepare metadata:
     ```python
     storage_metadata = {
         "agent_id": agent_id,
         "quality_score": metadata.get("quality_score"),
         "tier": metadata.get("tier"),
         "iteration": metadata.get("iteration", 0),
         "revision_number": metadata.get("revision_number", 0),
         "timestamp": datetime.now().isoformat()
     }
     ```
   - Call `await self.similarity_corpus.store_content(content, agent_id, storage_metadata)`
   - Log success with content preview (first 100 chars)

3. **Update `evaluate()` method:**
   - No changes needed (storage happens externally in orchestrator)

**Acceptance Criteria:**
- Storage is optional (works without similarity corpus)
- Metadata includes all relevant fields
- Logging shows what was stored

---

### Step 9: Modify Orchestrator

**File:** `src/workflow/orchestrator.py`

**Instructions:**

1. **Update `__init__` method:**
   - After initializing session_id, add:
   - Create shared corpus: `self.similarity_corpus = SimilarityCorpus(self.session_id)`
   - Pass corpus to evaluators when initializing:
     ```python
     self.generator_evaluator = EvaluatorAgent(
         session_id=self.session_id,
         profile=evaluator_profile,
         similarity_corpus=self.similarity_corpus
     )
     self.discriminator_evaluator = EvaluatorAgent(
         session_id=self.session_id,
         profile=evaluator_profile,
         similarity_corpus=self.similarity_corpus
     )
     ```

2. **Update `_generator_evaluator()` method:**
   - After selecting best revision (where `best_generator_content` is determined):
   - Add:
     ```python
     if best_content:
         await self.generator_evaluator.store_best_revision(
             content=best_content,
             agent_id="generator",
             metadata={
                 "quality_score": best_score,
                 "tier": evaluation_result.tier,
                 "iteration": state.get("iterations", 0),
                 "revision_number": generator_iterations
             }
         )
     ```

3. **Update `_discriminator_evaluator()` method:**
   - After selecting best revision (where best discriminator response is determined):
   - Add same storage call with `agent_id="discriminator"`

4. **Update `shutdown()` method (if exists) or `run_async()` finally block:**
   - Add: `await self.similarity_corpus.clear_session()`
   - Log cleanup

**Acceptance Criteria:**
- Corpus is shared between both evaluators
- Storage happens after evaluation completes
- Session cleanup works properly
- No breaking changes to existing orchestration

---

### Step 10: Add Configuration

**File:** `.env.example` and `.env`

**Instructions:**

Add these variables to both files:

```bash
# Similarity Detection Configuration
SIMILARITY_THRESHOLD=0.85                    # Accept if similarity < this (0.0-1.0)
SIMILARITY_RELAXED_THRESHOLD=0.90            # Fallback threshold after max attempts
MAX_SIMILARITY_ATTEMPTS=5                    # Max regeneration attempts per agent

# Content Chunking
PARAGRAPH_MIN_LENGTH=50                      # Min chars for a paragraph
CHUNK_OVERLAP=50                            # Overlap between chunks (chars)

# Sliding Window (for long content)
SLIDING_WINDOW_SIZE=3                        # Chunks per window
SLIDING_WINDOW_ACTIVATION_THRESHOLD=100      # Activate when corpus > N chunks
```

**Acceptance Criteria:**
- Variables documented with comments
- Default values are reasonable
- Types are clear (float, int)

---

### Step 11: Update Storage Module Exports

**File:** `src/storage/__init__.py`

**Instructions:**
- Add import: `from .similarity_corpus import SimilarityCorpus`
- Add to `__all__`: `"SimilarityCorpus"`

**Acceptance Criteria:**
- Module is importable from `src.storage`

---

### Step 12: Update Agents Module Exports

**File:** `src/agents/__init__.py` (if exists)

**Instructions:**
- Add imports:
  ```python
  from .similarity_checker import SimilarityChecker, SimilarityResult, ParagraphMatch
  from .similarity_feedback import generate_similarity_feedback
  ```
- Add to `__all__` if present

**Acceptance Criteria:**
- Classes are importable from `src.agents`

---

### Step 13: Create Exceptions Module Init

**File:** `src/exceptions/__init__.py`

**Instructions:**
- Create new file if doesn't exist
- Import: `from .similarity import TopicExhaustionError, SimilarityCorpusError`
- Export in `__all__`

**Acceptance Criteria:**
- Exceptions are importable from `src.exceptions`

---

## Testing Instructions

### Manual Testing

1. **Start the system:**
   ```bash
   python main.py
   ```

2. **Test similarity detection:**
   ```bash
   python client.py "Explain quantum entanglement" --max-iterations 5
   ```

3. **Expected behavior:**
   - First response should be unique (no similarity)
   - Subsequent responses should avoid repeating earlier content
   - Check logs for similarity check messages
   - Look for "⚠️ Content too similar" warnings if repetition occurs

4. **Test corpus storage:**
   - After conversation completes, check ChromaDB:
     ```python
     from src.storage.similarity_corpus import SimilarityCorpus
     corpus = SimilarityCorpus("test_session")
     stats = await corpus.get_corpus_stats()
     print(stats)
     ```
   - Should show stored documents and chunks

5. **Test feedback generation:**
   - Manually create a repetitive scenario
   - Check that feedback includes specific paragraph numbers
   - Verify feedback is actionable

### Edge Cases to Test

1. **First iteration** (empty corpus):
   - Should accept content immediately
   - No similarity checks needed

2. **Highly repetitive topic:**
   - Generate content about "define quantum entanglement" 10 times
   - Should eventually log warnings
   - Should accept best effort after max attempts

3. **Long content** (>1000 words):
   - Test sliding window activation
   - Check logs for "Activated sliding window search"

4. **Naturally repetitive phrases:**
   - Include "As discussed earlier" in content
   - Should NOT be flagged as repetitive

### Verification Checklist

- [ ] Similarity checks run before quality evaluation
- [ ] Best revisions stored in ChromaDB corpus
- [ ] Feedback is detailed and actionable
- [ ] Max attempts limit works (doesn't infinite loop)
- [ ] Relaxed threshold activates correctly
- [ ] Generator and Discriminator share corpus
- [ ] Session cleanup works (no orphaned collections)
- [ ] Logs show similarity scores
- [ ] No breaking changes to existing iteration counting
- [ ] Performance is acceptable (<1s per similarity check)

---

## Rollback Plan

If issues occur:

1. **Revert agent changes:**
   - Remove similarity checking from `generator.py` and `discriminator.py`
   - Keep new files but disable usage

2. **Disable via environment variable:**
   - Add `ENABLE_SIMILARITY_CHECK=false` to `.env`
   - Guard all similarity code with this check

3. **Remove corpus collections:**
   - Run cleanup script:
     ```python
     from src.storage.similarity_corpus import SimilarityCorpus
     corpus = SimilarityCorpus("session_id")
     await corpus.clear_session()
     ```

---

## Success Criteria

Implementation is complete when:

1. ✅ All new files created and properly structured
2. ✅ Agent modification doesn't break existing tests
3. ✅ Similarity detection works in manual testing
4. ✅ Feedback is generated and shown in logs
5. ✅ ChromaDB stores content correctly
6. ✅ No infinite loops or crashes
7. ✅ Configuration via environment variables works
8. ✅ Code passes type checking (`mypy src/`)
9. ✅ Documentation is clear (docstrings present)
10. ✅ Performance impact is minimal (<20% slowdown)

---

## Notes for Implementation

- **Code style:** Follow existing Musequill patterns (async/await, logging, type hints)
- **Error handling:** Use try/except with proper logging, don't let exceptions crash the workflow
- **Logging:** Use appropriate levels (DEBUG for searches, INFO for storage, WARNING for similarity issues)
- **Type hints:** All functions should have complete type annotations
- **Docstrings:** Use Google-style docstrings with Args/Returns sections
- **Testing:** Write unit tests for `similarity_feedback.py` functions
- **ChromaDB patterns:** Follow existing patterns from `chroma_manager.py` for consistency