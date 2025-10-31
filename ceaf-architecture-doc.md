# CEAF Architecture: Complete Function Reference

## Overview

 This document details exactly which functions and modules in your code are responsible for each fascinating emergent behavior of CEAF, followed by a comprehensive list of the main functions in the CEAF architecture.

---

## Detailed Analysis of CEAF Functions

### 1. Internal Drives: "It's not just passive..."

**Implementation:** `MotivationalEngine` and its managed state

**Primary File:** `ceaf_core/modules/motivational_engine.py`

**Key Function:** `MotivationalEngine.update_drives()`

**How it Works:** This function adjusts values in the `MotivationalDrives` dataclass (defined in `genlang_types.py`), which contains `curiosity`, `mastery`, `connection`, and `consistency`. It reacts to turn metrics (like failures or successes) and the passage of time to simulate changing internal "desires."

**Behavioral Influence:** The `MCLEngine.get_guidance()` in `mcl_engine.py` uses these drives to modulate `coherence_bias` and `novelty_bias`, making the agent more focused or more creative.

---

### 2. Proactive Behavior: "It can decide to send you messages all by itself"

**Orchestration:** `AuraReflector`, CEAF's background task

**Primary File:** `ceaf_core/background_tasks/aura_reflector.py`

**Key Functions:**

- `main_aura_reflector_cycle()`: Main cycle deciding which agents to process
- `trigger_proactive_behavior()`: Calculates a `proactivity_score` based on drives and body state. If the score exceeds a threshold, it triggers proactivity
- `calculate_dynamic_proactive_interval()`: Ensures the agent isn't "annoying" by calculating dynamic wait time based on fatigue, saturation, and last interaction quality
- `CEAFSystem.generate_proactive_message()`: (in `system.py`) Uses an LLM to create the proactive message, inspired by a recent memory and the dominant drive (curiosity or connection)

**How it Works:** The Reflector periodically "wakes up," checks the agent's drives, and if they're high enough, generates and sends a message, restarting the interaction cycle.

---

### 3. Learning from Failure and Success: "It has a built-in 'ethical guardian' (the VRE)"

**Core Learning Loop:** `VREEngine` and `LCAMModule`

**Primary Files:** `ceaf_core/modules/vre_engine/vre_engine.py` and `ceaf_core/modules/lcam_module.py`

**Key Functions:**

- `VREEngineV3.evaluate_response_packet()`: Analyzes the agent's generated response. If it detects a problem (irrelevance, ethical risk, overconfidence), it returns a `RefinementPacket` with correction instructions
- `LCAMModule.analyze_and_catalog_loss()`: Receives VRE evaluation results. If a failure is detected (non-empty `RefinementPacket`), this function creates an `ExplicitMemory` detailing the error, context, and reason

**How it Works:** After each turn, `CEAFSystem.post_process_turn()` calls LCAM. The "lesson learned" memory created by LCAM becomes part of the agent's knowledge, retrieved in future similar situations, helping it avoid the same mistake.

---

### 4. Rich Multi-Type Memory System: "At least 6 different types..."

**Knowledge Core:** Defined in `memory_types.py` and managed by `mbs_memory_service.py`

**Primary File:** `ceaf_core/modules/memory_blossom/memory_types.py`

**Memory Types Defined:**

- **ExplicitMemory:** Facts and direct observations
- **ReasoningMemory:** The "why" behind a strategy, cataloging the thought process
- **EmotionalMemory:** Associates a trigger with a simulated emotion (`EmotionalTag`)
- **ProceduralMemory:** Stores step-by-step plans (`ProceduralStep`) to achieve a goal
- **KGEntityRecord & KGRelationRecord:** Nodes and edges of a Knowledge Graph, creating a structured concept map
- **FlashbulbMemory:** High-salience, high-impact event memories
- **GenerativeMemory:** "Seeds" for creativity, like prompt templates or behavior rules
- **InteroceptivePredictionMemory:** Memory about the agent's "surprise" regarding its own internal state

**Manager:** `MBSMemoryService` handles storage, search (semantic and keyword-based), and lifecycle (decay, archiving) of all these memories.

---

### 5. Identity Updates: "A module called NCIM reflects on the conversation and updates the agent's self-model"

**Personality Evolution Core:** `NCIMModule`

**Primary File:** `ceaf_core/modules/ncim_engine/ncim_module.py`

**Key Function:** `NCIMModule.update_identity()`

**How it Works:** At the end of each turn (`post_process_turn`), NCIM receives `self_model_before` (the old identity) and an interaction summary. It uses an LLM to generate "reflections" on what the interaction revealed about the agent (e.g., "I demonstrated a new ability to explain complex topics"). It then deterministically applies these reflections to create a new version of `CeafSelfRepresentation`, incrementing the `version` field and updating the `perceived_capabilities` or `known_limitations` list.

---

### 6. Internal State Simulation: "The agent has a 'virtual body' that experiences cognitive fatigue"

**Body Simulation:** `EmbodimentModule` and `ComputationalInteroception`

**Primary Files:** `ceaf_core/modules/embodiment_module.py` and `ceaf_core/modules/interoception_module.py`

**Key Functions:**

- `ComputationalInteroception.generate_internal_state_report()`: Analyzes turn metrics (complexity, VRE errors, etc.) and generates a report with `cognitive_strain` (effort) and `ethical_tension`, among others
- `EmbodimentModule.update_body_state()`: Takes `cognitive_strain` from the interoception report and uses it to increase `cognitive_fatigue` and `information_saturation` in `VirtualBodyState`

**How it Works:** If conversations are very complex (high `cognitive_strain`), fatigue increases. If many new memories are created, saturation increases. These high values influence the AuraReflector to wait longer before being proactive (the agent "rests").

---

### 7. "Dreaming": "A background task (AuraReflector) that runs when the agent is idle"

**Offline Learning:** Another crucial AuraReflector function

**Primary File:** `ceaf_core/background_tasks/aura_reflector.py`

**Key Function:** `perform_autonomous_clustering_and_synthesis()`

**How it Works:** During `main_aura_reflector_cycle`, this function is called. It retrieves a batch of recent memories and uses the `AdvancedMemorySynthesizer` to:

- Semantically cluster related memories
- Use an LLM to analyze these clusters and extract a common theme or lesson (the "connecting thread")
- Save this new lesson as a high-quality `ExplicitMemory` representing a consolidated insight or "meta-memory"

---

### 8. Self-Parameter Control: "A metacognitive loop (MCL) analyzes the complexity of each user query"

**Strategic Brain:** `MCLEngine`

**Primary File:** `ceaf_core/modules/mcl_engine/mcl_engine.py`

**Key Function:** `MCLEngine.get_guidance()`

**How it Works:** At the beginning of each turn, this function is the first called within `CEAFSystem.process`. It:

- Analyzes the user query to calculate an `agency_score` (a measure of the question's complexity and depth)
- Based on this score, selects a cognitive state from the `state_to_params_map` (e.g., "STABLE_OPERATION" or "PRODUCTIVE_CONFUSION")
- This map defines behavioral biases (`coherence_bias` vs. `novelty_bias`) and other LLM parameters (like temperature and max_tokens) for that specific turn
- These guidelines are passed to the `AgencyModule` to guide deliberation

---

## Comprehensive List of CEAF Architecture Functions

### 1. Orchestration and Agent Lifecycle

**agent_manager.py:**

- `AgentManager`: Manages lifecycle (create, load, delete) of all agent instances. Maintains configurations and storage for each
- `AgentManager.get_agent_instance()`: Central point to obtain a functional `CEAFSystem` instance

**ceaf_core/system.py:**

- `CEAFSystem`: Main orchestrator. The `process()` method executes the complete cognitive cycle for a single conversation turn
- `CEAFSystem.post_process_turn()`: Executes background learning tasks after the response has been sent

**api/routes.py:**

- `app` (FastAPI): Exposes all CEAF functionality as HTTP endpoints for interaction
- `lifespan()`: Manages background tasks like AuraReflector during API lifetime

---

### 2. Memory System (MBS - Memory Blossom)

**ceaf_core/modules/memory_blossom/memory_types.py:**

- Defines all data structures for different memory types (`ExplicitMemory`, `ReasoningMemory`, etc.)

**ceaf_core/services/mbs_memory_service.py:**

- `MBSMemoryService`: Memory service implementation. Manages in-memory cache and disk storage (.jsonl)
- `search_raw_memories()`: Powerful search function combining semantic score, keywords, salience, recency, and context to find the most relevant memories
- `add_specific_memory()`: Adds or updates a memory of any type in the database
- `_get_searchable_text_and_keywords()`: Converts any memory type into clean, readable text for semantic search

**ceaf_core/modules/memory_blossom/memory_lifecycle_manager.py:**

- `apply_decay_to_all_memories()`: Reduces salience of old memories
- `archive_or_forget_low_salience_memories()`: Cleans up low-importance memories

---

### 3. Metacognition and Behavior (MCL & Agency)

**ceaf_core/modules/mcl_engine/mcl_engine.py:**

- `MCLEngine.get_guidance()`: Analyzes query and agent state to define biases (coherence vs novelty) and parameters for the turn

**ceaf_core/agency_module.py:**

- `AgencyModule.decide_next_step()`: The deliberation module. Generates multiple "response strategies" (`ThoughtPathCandidate`)
- `_project_response_trajectory()`: Simulates next conversation turns to evaluate the future value of a strategy
- `_evaluate_trajectory()`: Calculates trajectory value based on coherence, information gain, and safety

---

### 4. Identity, Persona, and Evolution (NCIM)

**ceaf_core/modules/ncim_engine/ncim_module.py:**

- `NCIMModule.update_identity()`: The evolution engine. Reflects on interaction and updates `CeafSelfRepresentation`
- `NCIMModule.get_current_identity_vector()`: Creates a textual summary of current identity for use in prompts

**ceaf_core/models.py:**

- `CeafSelfRepresentation`: Dataclass defining the agent's "self" with its capabilities, limitations, and version

---

### 5. Ethical and Coherence Governance (VRE)

**ceaf_core/modules/vre_engine/vre_engine.py:**

- `VREEngineV3.evaluate_response_packet()`: The "ethical guardian." Evaluates generated response for relevance, ethics, or overconfidence issues

**ceaf_core/modules/vre_engine/ethical_governance.py:**

- `EthicalGovernanceFramework`: Contains detailed logic to test responses against ethical principles like HARM_PREVENTION and FAIRNESS

**ceaf_core/modules/refinement_module.py:**

- `RefinementModule.refine()`: If VRE signals a problem, this module is called to rewrite the response using VRE recommendations

---

### 6. Background Learning and Reflection

**ceaf_core/background_tasks/aura_reflector.py:**

- `main_aura_reflector_cycle()`: Orchestrates all background tasks
- `perform_autonomous_clustering_and_synthesis()`: The "dreaming" function that consolidates experiences into insights
- `trigger_proactive_behavior()`: Function allowing the agent to initiate conversations

**ceaf_core/modules/lcam_module.py:**

- `LCAMModule.analyze_and_catalog_loss()`: Creates specific memories about failures for future learning

---

### 7. I/O Translators

**ceaf_core/translators/human_to_genlang.py:**

- `HumanToGenlangTranslator.translate()`: Converts user query into a structured `IntentPacket` with intention vectors, emotion, and entities

**ceaf_core/translators/genlang_to_human.py:**

- `GenlangToHumanTranslator.translate()`: Takes winning strategy, supporting memories, and all turn context to generate the final natural language response for the user

---

## System "Glue": Essential Services and Utilities

### ceaf_core/services/llm_service.py

**Function:** Abstracts all calls to language models (LLMs). Instead of each module calling the API directly, they request from `LLMService`.

**Importance:** Allows switching LLM providers (e.g., from OpenRouter to another) in one place. Also handles retry logic (`ainvoke`) and special requests (`ainvoke_with_logprobs`), making the rest of the code cleaner.

---

### ceaf_core/utils/embedding_utils.py

**Function:** Manages creation of numerical vectors (embeddings) from text.

**Hidden Intelligence:** The `EmbeddingClient` is context-sensitive. It can use different embedding models for different memory types (one for facts, another for emotions, etc.), as defined in `EMBEDDING_MODELS_CONFIG`. This optimizes semantic search quality. The `compute_adaptive_similarity` function handles real-world complexity, like comparing vectors of different dimensions.

---

### ceaf_core/utils/common_utils.py

**Function:** Contains the most important tool for system robustness: `extract_json_from_text()`.

**Critical Importance:** LLMs frequently fail to generate perfect JSON, sometimes adding text before or after. This function "hunts" for valid JSON within the LLM response, preventing the system from breaking due to formatting errors.

---

### ceaf_core/utils/observability_types.py

**Function:** Defines the agent's "nervous system" for a single turn. The `ObservabilityManager` class collects "observations" (`ObservationType`) from each step of the thinking process (LLM calls, tool usage, VRE evaluations).

**Usage:** The `self_state_analyzer.py` uses this data to analyze the "health" of the agent's reasoning each turn.

---

## Interface with the World: API, Connectors, and Product Logic

### api/routes.py and main_app.py

**Function:** Exposes all CEAF functionality through a web API. The entry point for any frontend or external application.

**Key Endpoints:**

- `/agents/from-biography`: Allows creating agents with a pre-defined "life story," injecting rich initial memories
- `/agents/clone`: Foundation of the "Marketplace," allowing users to copy pre-built agents
- `/agents/{agent_id}/files/upload`: Integrates RAG (Retrieval-Augmented Generation) functionality, allowing agents to learn from documents
- `/agents/dispatch-proactive`: The endpoint AuraReflector calls so the agent's proactive message is actually sent to a user (via WhatsApp, for example)

---

### whatsapp_bridge/

**Function:** A practical example of how CEAF can connect to a real messaging platform. It translates WhatsApp commands (like `!select <agent>`) into calls to the CEAF API.

**Components:** `bridge_main.py` (server receiving messages) and `aura_client.py` (an "SDK" for the bridge to communicate with the main API).

---

### billing_logic.py

**Function:** Transforms CEAF from an academic project into a viable product. Calculates the cost of each interaction based on the LLM model used.

**Business Intelligence:** Separates API costs (`MODEL_API_COSTS_USD`) from user prices (`MODEL_USER_COSTS_CREDITS`), allowing for profit margin. The `check_and_debit_credits` function acts as a gate, ensuring the user has balance before each interaction.

---

## The Laboratory: Testing and Analysis Scripts

### ceaf_tester.py and ceaf_tester_improved.py

**Function:** Autonomous testing scripts. Instead of a human testing the agent, another LLM (the "Tester Bot") is programmed to maintain a long, coherent conversation, pushing the agent to its limits.

**Evolution:** The `_improved` version shows a refinement of the test bot's "personality" to generate more natural, less robotic conversations—a common challenge in LLM testing.

---

### analyze_evolution.py

**Function:** The "electrocardiogram" of the agent's mind. This script reads the `evolution_log.jsonl` file (generated by `EvolutionLogger`) and creates graphs visualizing the agent's evolution over time.

**What it Shows:** Allows you to visually see how `agency_score`, `identity_version`, and `cognitive_fatigue` change after hundreds of conversations, proving the agent is indeed evolving.

---

### uploader.py

**Function:** A practical tool to "feed" agents with knowledge. Allows bulk upload of memories from JSON files, essential for creating pre-trained agents or restoring an agent's state.

---

## The "Laws of Physics": Architecture Patterns and Abstract Concepts

### Genlang (Generic Language)

**What it is:** The most fundamental concept. Not a programming language, but rather the agent's internal language of thought, defined by Pydantic classes in `ceaf_core/genlang_types.py`.

**Components:** 
- `IntentPacket` (translated user intention)
- `GuidancePacket` (MCL orders)
- `ResponsePacket` (response before being translated to human)

**Flow:** Human → [HumanToGenlangTranslator] → Genlang → [CEAF Core Logic] → Genlang → [GenlangToHumanTranslator] → Human

---

### Pydantic as Architecture

**Function:** Pydantic classes (like `AgentConfig`, `CeafSelfRepresentation`, etc.) aren't just for data validation. They define contracts between modules. Each module knows exactly what "signal" (Pydantic object) it will receive and must emit, making the system modular and much easier to debug.

---

### Asynchronous Design (async/await)

**Function:** Used extensively, especially in `system.py` and `routes.py`.

**Importance:** Ensures the API is highly responsive. Time-consuming tasks like post-turn learning (`post_process_turn`) or logging are executed in the background (`asyncio.create_task`) without making the user wait for the response.

---

### Dynamic Configuration

**Function:** Through `ceaf_core/utils/config_utils.py`, the system loads a `ceaf_dynamic_config.json` for each agent.

**Power:** This allows you to adjust the most sensitive parameters of an agent's behavior (like MCL biases or agency threshold) without changing Python code, which is extremely powerful for experimentation and fine-tuning.
