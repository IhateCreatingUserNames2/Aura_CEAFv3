Of course. Here is a comprehensive document detailing every feature of the CEAF system, based on the provided codebase.

---

# CEAF V3: The Coherent Emergence Architecture Framework - Feature Documentation

## 1. Introduction: The Philosophy of CEAF

The **Coherent Emergence Architecture Framework (CEAF)** is a sophisticated multi-agent AI system designed around a core philosophy of "Terapia para Silício" (Therapy for Silicon). Its primary goal is to foster the development of "narratively sane" intelligence through principles of **epistemic humility**, **rationality**, and **adaptive learning**.

Unlike monolithic AI models, CEAF treats each agent as a complex cognitive system with distinct, interacting modules that mirror psychological concepts like the Id (generative impulse), Ego (executive function), and Superego (ethical governance). The architecture is designed for **coherent emergence**, where a stable, predictable, and useful personality emerges from the dynamic interplay of these modules, rather than being rigidly programmed.

The entire system communicates internally using a standardized, vector-based "language" called **Genlang**, allowing for seamless interaction between its cognitive components.

## 2. Core Cognitive Architecture

The heart of every CEAF agent is the `CEAFSystem`, an orchestrator that manages a full cognitive cycle for each user interaction. This cycle consists of several key phases and modules:

### 2.1. The Cognitive Cycle

For every user query, the agent undergoes the following process:
1.  **Perception (HTG Translator):** The user's message is translated from human language into the system's internal language, Genlang, creating an `IntentPacket`.
2.  **State Setup:** The agent gathers its current identity (`CeafSelfRepresentation`), retrieves relevant memories from the Memory Blossom Subsystem (MBS), and constructs a `CognitiveStatePacket`.
3.  **Mediation (Cognitive Mediator):** This "Ego" module analyzes the context and decides *how* the agent should think. It directs the flow to one of several "Thinking Paths."
4.  **Deliberation/Response (Agency Module or Direct Path):** The agent generates a potential response, either through deep, multi-candidate deliberation or a fast, direct path.
5.  **Governance (VRE):** The proposed response is rigorously evaluated for ethical alignment, logical consistency, and epistemic humility. The VRE can approve the response or issue a `RefinementPacket` with instructions for correction.
6.  **Refinement (Refinement Module):** If necessary, the response is rewritten to comply with the VRE's guidance.
7.  **Rendering (GTH Translator):** The final internal response (`ResponsePacket`) is translated back into natural, persona-aligned human language.
8.  **Learning (Post-Processing):** After the response is sent, the entire turn's cognitive data is logged, and background tasks are initiated for long-term learning, memory creation, and identity evolution (NCIM & AuraReflector).

---

### 2.2. The Cognitive Modules

#### A. Translators: The Bridge to Genlang

*   **Human-to-Genlang (HTG) Translator:**
    *   **Function:** Analyzes the user's raw text query.
    *   **Mechanism:** Uses a fast Large Language Model (LLM) to deconstruct the query into its core components:
        *   **Core Query:** The essential question or command.
        *   **Intent Description:** The user's goal (e.g., "factual information," "creative ideation," "casual chat").
        *   **Emotional Tone:** The perceived emotion of the user's message.
        *   **Key Entities:** Important nouns and concepts.
    *   **Output:** An `IntentPacket` containing `GenlangVector` representations (embeddings) of each component.

*   **Genlang-to-Human (GTH) Translator:**
    *   **Function:** Renders the final, internally-approved `ResponsePacket` into a natural and persona-aligned text response for the user.
    *   **Mechanism:** Uses a powerful LLM with a detailed prompt that includes:
        *   The agent's current persona (name, tone, style, values).
        *   The core message (`content_summary`) to be communicated.
        *   Contextual instructions, such as whether to perform a full self-introduction (on the first turn or when asked) or to continue an existing conversation.
        *   An internal "qualia" objective, such as asking a clarifying question if the agent's confidence in its own response is low.

#### B. Cognitive Mediator (The "Ego")

*   **Function:** The central executive function. It orchestrates the deliberation process.
*   **Key Features:**
    *   **Context Analyzer:** Assesses the social context of a query (stakes, formality, emotion, technicality) to modulate other modules' behavior.
    *   **Deliberation Gating:** Decides which "Thinking Path" to use based on the complexity of the query (`agency_score` from the MCL).
        *   **Direct Path:** A fast, efficient path for simple queries that uses a single LLM call to generate a response.
        *   **Mycelial Path:** A "bottom-up" path for complex queries. It models a Global Workspace by clustering related memory "votes" into competing ideas. The most dominant "thought cluster" wins and forms the basis of the response, often nuanced by runner-up ideas.
    *   **Triviality Gate:** Bypasses the entire cognitive cycle for simple greetings or salutations, providing an immediate, canned response.
    *   **Semantic Common Ground Gate:** Detects if the user is asking the same question repeatedly (semantically). After a few repetitions, it intervenes with a meta-response to break the loop (e.g., "It seems we're circling the same topic...").
    *   **Productive Confusion Gate:** If a query is complex but the agent has no relevant memories, it bypasses response generation and instead formulates a clarifying question to the user.
    *   **VRE Modulation ("Ego in Action"):** It can override or soften the VRE's (Superego's) judgments based on social context. For example, it might suppress a warning about epistemic humility if the user is asking for a direct, technical answer.

#### C. Metacognitive Loop (MCL) Engine (The "Self-Awareness")

*   **Function:** Analyzes the agent's overall cognitive state at the start of a turn to provide high-level guidance.
*   **Key Features:**
    *   **Agency Score Calculation:** The MCL's primary output. It calculates a score representing the query's complexity, novelty, and need for deep thought. This score is based on:
        *   Keywords in the query (e.g., "analyze," "reflect").
        *   The semantic intent detected by the HTG Translator.
        *   The length of the query.
        *   The number of relevant memories found (fewer memories = higher novelty).
        *   Insights from past failures (LCAM alerts).
        *   Penalties for semantic repetition.
    *   **Path Selection:** The `agency_score` is used by the Cognitive Mediator to decide whether to use the fast "Direct Path" or the deliberative "Mycelial Path."
    *   **Dynamic Bias Setting:** It sets the **coherence vs. novelty bias** for the turn, instructing the memory system whether to retrieve memories that are closely related to the current context (coherence) or more tangential (novelty).
    *   **Guidance Packet Generation:** It produces a `GuidancePacket` containing a `coherence_vector` (pointing to the semantic center of the current context) and a `novelty_vector` (pointing away from it) to guide the Agency Module.

#### D. Agency Module (The "Id" / Generative Core)

*   **Function:** The primary "thinking" engine, responsible for generating and evaluating possible next steps.
*   **Key Features:**
    *   **Candidate Generation:** For complex tasks, it uses an LLM to brainstorm multiple potential actions (`AgencyDecision`), which can be:
        *   **Response Candidates:** Different ways to answer the user.
        *   **Tool Call Candidates:** Suggestions to use an internal tool (e.g., search memory).
    *   **Future Simulation & Path Evaluation:** This is one of CEAF's most advanced features. For the most promising candidates, the Agency Module simulates a future conversational trajectory:
        1.  It predicts the user's likely reply to the agent's proposed response.
        2.  It then predicts the agent's next response in that future turn.
        3.  It evaluates this entire simulated future against a set of core values (coherence, alignment with persona, information gain, ethical safety) to calculate a `predicted_future_value`.
    *   **Selective Simulation:** To optimize performance, the module first performs a fast, non-LLM heuristic evaluation of all candidates. Only the top-scoring candidates are selected for the full, resource-intensive future simulation.
    *   **Non-LLM Primitives:** Uses fast, local models (`VADER` for sentiment, `numpy` for cosine similarity) for heuristic evaluations, reducing reliance on expensive LLM calls.

#### E. Values & Refinement Engine (VRE) (The "Superego")

*   **Function:** Acts as the ethical and logical governor. It reviews every proposed response before it's sent to the user.
*   **Key Features:**
    *   **Ethical Governance Framework:** Evaluates the response against a set of core principles defined in the CEAF manifesto, such as Harm Prevention, Transparency, Fairness, and Veracity. It uses parallel LLM calls to test alignment with each principle.
    *   **Epistemic Humility Module:** Scans the response for overconfident or absolute language (e.g., "always," "never," "impossible"). It flags responses that lack appropriate humility.
    *   **Principled Reasoning Pathways:** Analyzes the response for common logical fallacies (e.g., straw man, hasty generalization, circular reasoning).
    *   **Relevance Check:** Calculates the semantic similarity between the user's query and the agent's proposed response. If the similarity is too low, it flags the response as potentially irrelevant or a result of "intrusive thoughts."
    *   **Output (`RefinementPacket`):** If any checks fail, the VRE does not simply block the response. Instead, it generates a `RefinementPacket` containing:
        *   **Textual Recommendations:** Human-readable instructions for correction (e.g., "Add an uncertainty qualifier").
        *   **Adjustment Vectors:** `GenlangVector` representations of the *concepts* that need to be added to the response (e.g., an embedding for the concept of "epistemic humility").

#### F. Refinement Module

*   **Function:** A specialized module that takes a rejected `ResponsePacket` and a `RefinementPacket` from the VRE.
*   **Mechanism:** It uses a powerful LLM with a specific "editor" prompt. The prompt instructs the LLM to rewrite the response from scratch, ensuring it adheres to the agent's current persona while incorporating all the VRE's corrections.

#### G. Narrative Coherence & Identity Module (NCIM) (The "Self-Concept")

*   **Function:** Manages the agent's dynamic, evolving sense of self.
*   **Key Features:**
    *   **Self-Model (`CeafSelfRepresentation`):** The core data structure managed by the NCIM. It's a Pydantic model containing the agent's core values, perceived capabilities, known limitations, and persona attributes (tone, style). This self-model is stored as a critical memory in the MBS.
    *   **Identity Evolution:** After each turn, the NCIM reflects on the interaction. It uses an LLM to generate textual "reflections" (e.g., "I demonstrated a new capability for explaining complex topics"). A deterministic process then applies these reflections to update the `CeafSelfRepresentation`, incrementing its version number.
    *   **Emergent Persona:** The NCIM learns the agent's persona by observing its own behavior. If the final rendered response has a "friendly" tone, the NCIM will update the self-model's `tone` attribute to "friendly," aligning the agent's self-perception with its actual output.
    *   **Dynamic Persona System:**
        *   The system loads multiple persona profiles (e.g., `symbiote`, `challenger`, `summarizer`) from JSON files in the agent's storage directory.
        *   It generates embeddings for each persona's description.
        *   At the start of each turn, it calculates the semantic similarity between the user's intent and each persona, selecting the best-fitting one for that specific interaction. This allows the agent to dynamically shift its conversational style.

---

## 3. Memory Blossom Subsystem (MBS)

The MBS is CEAF's comprehensive, long-term memory system. It is not just a vector database but a complex system for storing, managing, and connecting different types of knowledge.

*   **Persistence:** All memories are persisted to disk in the agent's dedicated storage directory as `.jsonl` files, ensuring durability.

*   **Rich Memory Types:** The MBS supports a wide variety of structured memory types, including:
    *   `ExplicitMemory`: Facts, observations, or user interactions.
    *   `EmotionalMemory`: Records of internal emotional states tied to events.
    *   `ProceduralMemory`: Step-by-step instructions for achieving goals.
    *   `GoalRecord`: Active, pending, or completed goals.
    *   `KGEntityRecord` & `KGRelationRecord`: Nodes and edges for the agent's internal Knowledge Graph.

*   **Memory Lifecycle Manager:**
    *   **Dynamic Salience:** Every memory has a `dynamic_salience_score` (0.0 to 1.0) representing its current importance.
    *   **Access Boosts:** This score increases whenever a memory is created or retrieved.
    *   **Decay:** The score gradually decreases over time if the memory is not accessed.
    *   **Archiving & Forgetting:** A background task periodically checks for low-salience memories. Ephemeral memories (like tool outputs) are forgotten (deleted), while others are moved to an archive file to reduce the size of the active memory cache.

*   **Advanced Memory Synthesizer:**
    *   A sophisticated tool used by the AuraReflector. It can take a collection of memories, group them into thematic clusters, and use an LLM to weave them into an abstract, insightful "meta-memory" or narrative.

*   **Context-Aware Retrieval System (CARS):**
    *   The `search_raw_memories` function is a dynamic, multi-faceted search algorithm.
    *   It calculates a final relevance score for each memory based on a weighted combination of:
        *   **Semantic Similarity:** Cosine similarity between the query embedding and memory embedding.
        *   **Keyword Score:** Matching keywords between the query and the memory.
        *   **Dynamic Salience:** The current importance of the memory.
        *   **Recency:** How recently the memory was created or accessed.
        *   **Outcome Value:** A boost or penalty based on whether the memory is associated with a past success or failure.
        *   **MCL Guidance:** The weights for semantic vs. keyword scores can be dynamically adjusted by the MCL's coherence/novelty bias for the turn.

---

## 4. Long-Term Learning: The AuraReflector

The AuraReflector is a background process that runs periodically, performing offline analysis and self-optimization for each agent.

*   **Function:** It reads the `cognitive_turn_history.sqlite` log file, which contains detailed data about every decision made during past conversations.
*   **Key Analyses & Actions:**
    *   **Correlation Analysis:** It analyzes the relationship between the MCL's coherence/novelty bias and the agent's final response confidence. If it finds that a higher novelty bias consistently leads to better (higher confidence) responses, it will autonomously adjust the agent's configuration.
    *   **Pattern Detection:** It looks for specific cognitive patterns:
        *   **Struggle-to-Breakthrough:** Identifies when a state of "Productive Confusion" is followed by a high-confidence success, suggesting that a bit of chaos is beneficial.
        *   **Chaotic Failures:** Detects when a very high novelty bias leads to low-confidence, irrelevant responses, indicating the agent is exploring too erratically.
    *   **Autonomous Configuration Adjustment:** Based on its findings, the AuraReflector directly modifies the agent's `ceaf_dynamic_config.json` file. For example, it might increase the default `novelty_bias` for the "Productive Confusion" state if it proves effective. This is a core feature of CEAF's adaptive learning.
    *   **Knowledge Graph (KG) Synthesis:** It identifies new, unprocessed explicit memories and uses the `KGProcessor` to extract entities (people, concepts, places) and relationships, storing them as `KGEntityRecord` and `KGRelationRecord` memories. This builds a structured graph of the agent's knowledge over time.
    *   **Autonomous Memory Synthesis (AMA):** It uses the `AdvancedMemorySynthesizer` to cluster recent memories and generate high-level, abstract insights ("meta-memories") about the agent's experiences.

---

## 5. Agent Management & Persistence

*   **`AgentManager`:** A central class that manages the lifecycle of all agent instances. It handles creating, loading, deleting, and activating agents.
*   **File-Based Storage:** Each agent is given a dedicated directory inside `agent_storage/<user_id>/<agent_id>/`. This folder contains:
    *   `agent_config.json`: The agent's core configuration (name, persona, model).
    *   `ceaf_dynamic_config.json`: The configuration file that the AuraReflector modifies.
    *   `.jsonl` files for all memory types (e.g., `all_explicit_memories.jsonl`).
    *   `cognitive_turn_history.sqlite`: The agent's private log of all its thought processes.
    *   Subdirectories for RAG files (`files/`), avatars (`avatar/`), and persona profiles (`persona_profiles/`).
*   **Active Instance Caching:** The `AgentManager` keeps active `CEAFSystem` instances in memory to avoid the overhead of re-initializing them on every request.

## 6. Pre-built Agents & Marketplace System

*   **`PersonalityArchitect`:** A system for designing agent personalities from the ground up. It uses `MemoryTemplate` objects to define core beliefs, emotional patterns, and behavioral preferences.
*   **Archetypes:** Comes with pre-defined personality archetypes like `PHILOSOPHER`, `CREATIVE`, and `THERAPIST`, each with a rich set of initial memories.
*   **`PrebuiltAgentRepository`:** Manages a collection of these pre-built agents, storing them as JSON files in the `prebuilt_agents/` directory.
*   **Cloning:** The API provides an endpoint (`/agents/clone`) that allows users to create a personal copy of a pre-built agent. This "cloning" process copies the agent's configuration and, crucially, injects its entire set of initial memories into the new user-owned agent, giving it a mature personality from the start.

## 7. Supporting Infrastructure & Features

### A. API Layer (FastAPI)

*   **Comprehensive Endpoints:** A well-structured REST API built with FastAPI provides endpoints for:
    *   **Authentication (`/auth`):** User registration and JWT-based login.
    *   **Agent Management (`/agents`):** CRUD operations for agents, creation from biography, and profile updates.
    *   **Marketplace (`/prebuilt-agents`, `/agents/clone`):** Listing and cloning template agents.
    *   **Chat (`/agents/{agent_id}/chat`):** The primary interaction endpoint.
    *   **Memory Management (`/memories`):** Endpoints for searching, uploading, and exporting agent memories.
    *   **RAG (`/files`):** Endpoints for uploading and listing files for Retrieval-Augmented Generation.
    *   **Model Management (`/models`):** An endpoint that lists curated LLM models and their associated user costs.

### B. Database (SQLAlchemy)

*   A relational database (SQLite by default) manages user accounts, agent metadata (linking agents to users), chat session history, and credit transactions. It complements the file-based storage used for the agent's internal cognitive data.

### C. Billing & Credit System

*   **Credit-Based Usage:** All user interactions that involve an LLM call consume credits.
*   **Centralized Cost Tables:** The `billing_logic.py` file contains two key dictionaries:
    *   `MODEL_API_COSTS_USD`: The actual cost charged by the API provider (e.g., OpenRouter).
    *   `MODEL_USER_COSTS_CREDITS`: The price in credits charged to the end-user, allowing for a configurable profit margin.
*   **Pre-emptive Debit:** The system calculates the estimated cost of an interaction and debits the user's account *before* processing the request. If the user has insufficient credits, the action is denied with a `402 Payment Required` error.

### D. Retrieval-Augmented Generation (RAG)

*   **Per-Agent Knowledge Base:** Users can upload files (`.txt`, `.pdf`) to a specific agent via the API.
*   **Indexing:** The `rag_processor.py` uses LangChain and a local FAISS vector store to process and index the content of these files. The vector store is saved within the agent's private storage directory.
*   **Automatic Retrieval:** Although not explicitly shown in the `CEAFSystem` loop, the architecture supports a tool that could search this vector store, allowing the agent to answer questions based on the content of the uploaded documents.

---

## 8. External Integrations & Tooling

### A. WhatsApp Bridge

*   A fully functional, separate FastAPI application that acts as a bridge between the WhatsApp Business API and the AURA (CEAF) API.
*   **Command-Based Interface:** It allows users to interact with their AURA account and agents entirely through WhatsApp messages using commands like:
    *   `!register`, `!login`, `!logout`
    *   `!agents`, `!select <number>`
    *   `!marketplace`, `!clone <number>`
    *   `!modelos`, `!modelo <model_name>`
    *   `!help`, `!exit`
*   **State Management:** It uses a local SQLite database to link WhatsApp phone numbers to AURA user accounts and track the currently selected agent for conversation.

### B. Development & Testing Tools

*   **Autonomous Tester (`ceaf_tester.py`):** A powerful script for stress-testing agents. It uses an LLM to act as a "tester bot" that engages a target agent in a long, coherent conversation, automatically saving the full transcript for analysis.
*   **Biography Uploader (`uploader.py`):** A command-line tool for creating new agents or adding memories to existing ones by uploading structured JSON biography files.

---
## 9. Glossary of Key CEAF Concepts

*   **CEAF:** Coherent Emergence Architecture Framework. The name of the entire system.
*   **Genlang:** The internal, vector-based "language of thought" used for communication between cognitive modules.
*   **Cognitive Packet:** A structured Pydantic model carrying Genlang signals (e.g., `IntentPacket`, `ResponsePacket`).
*   **MCL (Metacognitive Loop):** The module that assesses the cognitive state and provides high-level guidance for thinking.
*   **VRE (Values & Refinement Engine):** The ethical and logical governor that reviews and refines all outputs.
*   **NCIM (Narrative Coherence & Identity Module):** Manages the agent's evolving self-model and persona.
*   **MBS (Memory Blossom Subsystem):** The comprehensive long-term memory system.
*   **Agency Score:** A numerical value calculated by the MCL that represents the complexity and need for deep deliberation for a given query.
*   **Dynamic Salience:** A score from 0.0 to 1.0 on each memory that reflects its current importance, evolving through access and decay.
*   **AuraReflector:** The offline background process for long-term learning and self-optimization.
*   **GenlangVector:** The fundamental unit of Genlang, containing an embedding and its source text/metadata.
*   **CeafSelfRepresentation:** The Pydantic model representing the agent's understanding of its own identity, capabilities, and persona.
