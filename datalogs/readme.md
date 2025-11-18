
# AuraCEAF V3: An Adaptive Cognitive Engine for Autonomous Forms

**AuraCEAF V3** is a sophisticated cognitive architecture designed to serve as an external scaffolding for Large Language Models (LLMs), enabling the development of persistent, adaptive, and self-regulating AI agents. Instead of treating the LLM as a monolithic black box, AuraCEAF wraps it with a suite of Python modules that simulate an "internal world," forcing the model to operate according to principles of cognitive self-awareness, metacognition, and continuous identity evolution.


## Core Philosophy: Simulating an Internal World

The fundamental premise of AuraCEAF is that advanced AI agency emerges not just from a model's pre-trained knowledge, but from its ability to **observe, regulate, and learn from its own internal state**. Since the true hidden states of proprietary LLMs are inaccessible, AuraCEAF constructs an explicit, computational "internal world" for the agent.

The agent's behavior is a direct product of this simulated state, which includes:

-   **A Virtual Body (`EmbodimentModule`):** Tracks `cognitive_fatigue` and `information_saturation`, simulating physical and mental limits.
-   **Motivational Drives (`MotivationalEngine`):** Manages drives like `Curiosity`, `Connection`, `Mastery`, and `Consistency`, which fluctuate based on interaction success and create behavioral biases.
-   **A Dynamic Identity (`NCIMModule`):** Maintains a JSON-based `CeafSelfRepresentation` (self-model) that evolves after each interaction, allowing the agent to learn "who it is" from its experiences.
-   **A Living Memory System (`MBSMemoryService`):** Treats memory not as a static database, but as a dynamic network where the relevance (`dynamic_salience_score`) of each memory decays over time and is reinforced by access.

## Architecture Overview

AuraCEAF processes a user query through a multi-stage cognitive cycle, orchestrated by `system.py`.

 <!-- You can create and host a diagram to make this even clearer -->

### Key Modules:

1.  **Agent Manager (`agent_manager.py`)**
    -   Manages the lifecycle of multiple, isolated agent instances.
    -   Handles agent creation, data persistence (file system and database), and deletion.
    -   Acts as the entry point for interacting with any specific agent.

2.  **Human-to-Genlang Translator (`htg_translator.py`)**
    -   **Input:** Raw user query (e.g., "what do you think about that?").
    -   **Process:** Uses a fast LLM to analyze the query and decompose it into an `IntentPacket`, a structured object containing vectors for the core query, emotional tone, and key entities.
    -   **Output:** The `IntentPacket`, which is the system's internal, machine-readable understanding of the user's intent.

3.  **Metacognitive Loop (`mcl_engine.py`)**
    -   **The "Supervisor Brain"**. This is the core of the agent's self-regulation.
    -   **Process:** It analyzes the full `CognitiveStatePacket` (user intent, current agent identity, active memories, virtual body state, and motivational drives) to determine *how* the agent should think in the current turn.
    -   **Output:** Issues a `GuidancePacket` that sets critical parameters for the next stage, including:
        -   `agency_score`: A measure of how complex the task is, determining whether to use a fast, direct response path or a slower, more deliberate one.
        -   `coherence_bias` vs. `novelty_bias`: A trade-off between staying on-topic (coherence) and exploring new ideas (novelty).
        -   `operational_advice`: Special, high-priority instructions, such as "ALERT: Information saturation is critical. Change the topic."

4.  **Agency Module (`agency_module.py`)**
    -   **The Deliberative Thinker**. This module is activated for complex tasks.
    -   **Process:** It generates multiple potential `ThoughtPathCandidate`s (response strategies). It may simulate the conversational future of each path to predict its outcome and value.
    -   **Output:** Selects a single `WinningStrategy` that best aligns with the agent's current goals and internal state.

5.  **Genlang-to-Human Translator (`gth_translator.py`)**
    -   **The "Voice" of the Agent**. This is the final and most complex prompt-engineering step.
    -   **Input:** The `WinningStrategy`, supporting memories, the full internal state report (drives, body state), the agent's self-model, and the conversation history.
    -   **Process:** It constructs a massive, highly detailed prompt that instructs a powerful LLM on how to generate a final response. The prompt tells the LLM not just *what* to say, but *how* to say it, guiding its tone, style, and even sentence structure based on the agent's simulated internal "feelings".
    -   **Output:** The final, human-readable text response that the user sees.

6.  **Background Processes (`AuraReflector`)**
    -   **The "Subconscious"**. This background task runs periodically on active agents.
    -   **Process:** It simulates "rest" (decaying fatigue) and "dreaming" by clustering recent memories to synthesize new insights, discover emergent goals, and consolidate the knowledge graph (`KGProcessor`). This is critical for long-term learning and preventing information saturation.

