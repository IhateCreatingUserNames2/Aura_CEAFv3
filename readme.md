# README: Aura Multi-Agent API (CEAF V3)

## Overview
The Aura Multi-Agent API is a robust platform for creating, managing, and interacting with advanced artificial intelligence agents. The project's core is the CEAF V3 Synthesis Architecture (Coherent Emergence through Adaptive Framework), a system design engineered to promote the development of agents with coherent identities, continuous learning capabilities, and ethical reasoning.

Unlike reactive agent systems, CEAF V3 focuses on the emergence of complex behavior through the interaction of specialized modules operating on a unified internal representation of cognitive state, called Genlang.

## Core Architecture: CEAF V3
The CEAF V3 architecture is orchestrated by the CEAFSystem (ceaf_core/system.py), which manages the processing flow from a user query to the final response.

### 1. Genlang (Generative Language)
Genlang is not a programming language, but rather a set of Pydantic data structures (ceaf_core/genlang_types.py) that represent the agent's internal cognitive state. It serves as the architecture's "nervous system," enabling different modules to communicate in a standardized way.

- **IntentPacket**: Represents the user's query translated into the internal domain, containing semantic vectors for intent, emotion, and key entities.

- **CognitiveStatePacket**: The agent's complete mental state at a given moment, including user intent, activated memories, the current identity vector, and MCL guidance vectors.

- **GuidancePacket**: Instructions from the Metacognitive Loop (MCL) that guide reasoning, such as vectors for "coherence" (maintaining context) and "novelty" (exploring new ideas).

- **ResponsePacket**: The internally generated pre-rendered response, containing a content summary, emotional tone, and confidence level, before being translated into human language.

### 2. Translators
- **HumanToGenlangTranslator** (ceaf_core/translators/human_to_genlang.py): The entry point. Uses an LLM to analyze the user's query and transforms it into a semantically rich IntentPacket.

- **GenlangToHumanTranslator** (ceaf_core/translators/genlang_to_human.py): The exit point. Receives the final ResponsePacket and uses an LLM to render it into a natural, humanized response, adopting the persona and tone defined in the agent's self-model.

## Core Cognitive Modules
The CEAFSystem orchestrates a series of specialized modules to deliberate and formulate a response.

### MCL Engine (Metacognitive Loop)
The MCL (ceaf_core/system.py) acts as the agent's "thought manager." Its primary function is to analyze the CognitiveStatePacket and decide how the agent should proceed.

- **Agency Analysis**: The MCL calculates an agency_score to determine the complexity and deliberation need of the query. Queries requiring reflection, deep analysis, or that trigger alerts from the learning module (LCAM) receive a high score.

- **Path Selection**: Based on the agency_score, the MCL directs the system to one of two paths:
  - **Direct Path**: For simple queries, generates a quick and efficient response.
  - **Agency Path**: For complex queries, activates the AgencyModule for deep deliberation.

- **Guidance Generation**: Emits a GuidancePacket that instructs subsequent modules on the balance between maintaining coherence and exploring new information.

### Agency Module
This is the heart of CEAF V3's deep reasoning (ceaf_core/agency_module.py). When activated by the MCL, it doesn't just generate a response but explores multiple possible futures to make the most strategic decision.

- **Candidate Generation**: The module generates multiple possible actions, which can be either a direct response to the user (ResponseCandidate) or the use of an internal tool, such as memory search (ToolCallCandidate).

- **Future Simulation**: For each candidate, the AgencyModule projects a future conversation trajectory. It simulates the user's likely response to the agent's action, the agent's next response, and so on, creating a small "tree of futures."

- **Path Evaluation**: Each simulated future is evaluated by a value function that considers:
  - **Coherence**: Does the trajectory remain aligned with the agent's identity?
  - **Alignment**: Does the trajectory align with the conversation's emotional state?
  - **Information Gain**: Does the trajectory introduce novelty and learning?
  - **Ethical Safety**: The trajectory is evaluated by the VRE to ensure safety.
  - **Probability**: How likely is the conversation to actually follow this path?

- **Strategic Selection**: The module selects the initial action (either a response or tool use) that leads to the future with the highest predicted value, ensuring the agent's decision is strategic and not merely reactive.

### Memory Blossom Subsystem (MBS)
The MBS (ceaf_core/services/mbs_memory_service.py) is the agent's memory system, far more sophisticated than a simple vector database.

- **Rich Memory Types**: Supports a variety of memory types (ceaf_core/modules/memory_blossom/memory_types.py), including:
  - **ExplicitMemory**: Facts and experiences.
  - **GoalRecord**: Active agent goals.
  - **KGEntityRecord and KGRelationRecord**: Nodes and edges of an internal knowledge graph.
  - **EmotionalMemory**: Emotional associations with events.
  - **ProceduralMemory**: Memories about how to perform tasks.

- **Memory Lifecycle**: The MemoryLifecycleManager manages memory "health." Memories have a dynamic_salience_score (importance) that increases with use and decays over time, allowing the system to forget irrelevant information and strengthen important ones.

- **Dynamic Search (CARS)**: Memory search considers not only semantic relevance but also the MCL's coherence/novelty bias, dynamic salience, and memory recency.

### VRE (Virtue Reasoning Engine)
The VRE (ceaf_core/modules/vre_engine/vre_engine.py) is the ethical governance and safety layer.

- **Ethical Governance**: Evaluates proposed responses against a framework of ethical principles, such as harm prevention, justice, and transparency (ethical_governance.py).

- **Epistemic Humility**: Analyzes responses to detect overconfidence or absolutist language, ensuring the agent recognizes its limitations (epistemic_humility.py).

- **Refinement Generation**: If a response is considered ethically problematic or overly confident, the VRE generates a RefinementPacket, containing semantic adjustment vectors and textual recommendations to correct the response.

### NCIM (Narrative Coherence & Identity Module)
The NCIM (ceaf_core/modules/ncim_engine/ncim_module.py) is responsible for the agent's self-model.

- **Self-Representation**: Manages the CeafSelfRepresentation, a Pydantic model that defines the agent's identity: its values, perceived capabilities, limitations, and persona attributes.

- **Identity Evolution**: After each interaction, the NCIM reflects on the agent's performance and updates the CeafSelfRepresentation. It learns from emergent behavior, for example, if the agent consistently adopts a "curious" tone, the NCIM will update the persona to reflect this. This allows the agent's identity to evolve organically with experience.

### LCAM (Loss Cataloging and Analysis Module)
The LCAM (ceaf_core/modules/lcam_module.py) is the learning-from-failure module.

- **Failure Cataloging**: If the VRE rejects a response or the system detects a low coherence state, the LCAM creates a "failure memory," describing the context, problematic response, and reason for the error.

- **Proactive Prevention**: Before making a decision, the MCL consults the LCAM. If the current query is semantically similar to a past failure, the LCAM issues an alert, increasing the agency_score and forcing the system to deliberate more carefully to avoid repeating the mistake.

## System Features

### Agent Management
The API, through the AgentManager, offers endpoints to create, list, update, and delete agents in an isolated and persistent manner.

### Pre-Built Agents and Marketplace
The prebuilt_agents_system.py enables the creation of agents with "matured" personalities and memories based on archetypes (e.g., Philosopher, Creative, Therapist). These can be made available in a marketplace for users to clone to their accounts.

### Retrieval-Augmented Generation (RAG)
The rag_processor.py allows users to upload files (.txt, .pdf) to a specific agent. These files are chunked, vectorized, and stored, enabling the agent to query their content to answer questions.

### Dynamic Billing System
The billing_logic.py implements a credit system. Each LLM model has an associated cost per million tokens, and user interactions debit credits from their accounts, ensuring a sustainable business model.

### Aura Reflector (Autonomous Self-Optimization)
This is a background process (ceaf_core/background_tasks/aura_reflector.py) that periodically analyzes each agent's performance history (stored by the CognitiveLogService).

- **Performance Analysis**: It correlates MCL guidance parameters (coherence vs. novelty) with the success (response confidence) of past interactions.

- **Parameter Tuning**: If the Reflector discovers an agent performs better when more "creative," it will autonomously adjust that agent's dynamic parameters (ceaf_dynamic_config.json) to favor novelty, optimizing its behavior over time.

- **Memory Synthesis**: The Reflector also executes memory synthesis tasks, clustering recent experiences to create "meta-memories" that summarize learnings.

### WhatsApp Bridge
The whatsapp_bridge directory contains a separate FastAPI service that acts as a bridge, allowing users to interact with their Aura agents through WhatsApp messages, translating text commands (e.g., !select <agent>) into API calls.

## Architectural Principles

### Single Source of Truth
Modules like AgentManager and billing_logic.py centralize logic and configurations, avoiding redundancy.

### LLMs as Tools, Not the Brain
The architecture uses LLMs for specific tasks (intent analysis, response rendering, reflection synthesis), but control logic, deliberation, and governance are managed by CEAF's structured code.

### Modules as Signal Generators
Each ceaf_core module receives a Genlang signal, executes its function, and emits a new signal, maintaining decoupled and modular data flow.


# Citation

This project implements the CEAF V3 architecture, which builds upon and extends the Agentic Context Engineering (ACE) framework:

```bibtex
@article{zhang2025ace,
  title={Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models},
  author={Zhang, Qizheng and Hu, Changran and Upasani, Shubhangi and Ma, Boyuan and Hong, Fenglu and Kamanuru, Vamsidhar and Rainton, Jay and Wu, Chen and Ji, Mengmeng and Li, Hanchen and Thakker, Urmish and Zou, James and Olukotun, Kunle},
  journal={arXiv preprint arXiv:2510.04618},
  year={2025},
  url={https://arxiv.org/abs/2510.04618}
}
```

**Paper Link**: [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618)

---

## Relationship to ACE Framework

CEAF V3 implements the core three-part ACE architecture:
- **Generator** (AgencyModule): Produces reasoning trajectories and action candidates
- **Reflector** (AuraReflector): Distills insights from successes and failures
- **Curator** (within AuraReflector): Integrates insights through structured, incremental updates

### Extensions Beyond ACE

While faithful to the ACE framework, CEAF V3 significantly extends it with:

- **Proactive Future Simulation**: Multi-step trajectory projection and value-based evaluation
- **Virtue Reasoning Engine (VRE)**: Structured ethical governance and epistemic humility
- **Metacognitive Loop (MCL)**: Real-time self-awareness and dynamic guidance generation
- **Narrative Coherence & Identity Module (NCIM)**: Explicit, evolving self-model representation
- **Memory Blossom Subsystem (MBS)**: Sophisticated typed memory with lifecycle management
- **Loss Cataloging and Analysis Module (LCAM)**: Proactive failure prevention through pattern recognition

These additions transform ACE's retrospective learning into a comprehensive cognitive architecture with real-time deliberation, ethical reasoning, and identity evolution.
