# Aura Multi-Agent API (CEAF V3)

A FastAPI-based platform for creating, managing, and interacting with advanced AI agents built on the **CEAF V3 (Coherent Emergence through Adaptive Framework)** cognitive architecture. This project provides a powerful foundation for developing agents that exhibit coherent identities, continuous learning, and structured ethical reasoning.

The core of this project is an implementation and significant extension of the principles outlined in the **Agentic Context Engineering (ACE)** paper.

Current State:
<img width="732" height="338" alt="image" src="https://github.com/user-attachments/assets/17f4063a-f198-4dc9-baa6-af52a85de012" />

Code Is still not Ready to be Release, for now , if you are interested on this project, you can nagivate thru AuraCEAFv2 in https://github.com/IhateCreatingUserNames2/Aura_AI_Agents 
---

## 🚀 Key Features

* **🧠 Sophisticated Cognitive Architecture:** Goes beyond simple prompt-chaining to simulate a cognitive loop with distinct modules for metacognition (MCL), deep reasoning (Agency), ethical governance (VRE), and self-identity (NCIM).
* **🔮 Proactive Future Simulation:** The Agency Module can project, simulate, and evaluate multiple future conversational paths to make strategically optimal decisions, rather than just reacting to the last user message.
* **🛡️ Structured Ethical Governance (VRE):** A dedicated "Virtue Reasoning Engine" that evaluates agent responses against a framework of ethical principles and checks for overconfidence, ensuring safer and more responsible behavior.
* **🧬 Evolving Self-Model (NCIM):** Agents possess an explicit self-model of their identity, values, and limitations that evolves organically over time based on their interactions and performance.
* **⚡ Autonomous Self-Optimization (Aura Reflector):** A background process that analyzes agent performance history to autonomously fine-tune their operational parameters and synthesize new "meta-memories" from past experiences.
* **💾 Advanced Memory System (MBS):** Features multiple memory types (explicit, procedural, emotional, knowledge graph), a full memory lifecycle with dynamic salience and decay, and a dynamically weighted retrieval system.
* **🏪 Agent Marketplace:** A system for creating pre-built agent "archetypes" (e.g., Philosopher, Therapist) that users can clone into their own accounts to get started quickly.
* **🔌 Extensible and Modular:** Built with decoupled modules that communicate through a standardized internal representation (**Genlang**), making the system easy to extend and maintain.

---

## 🏛️ Architectural Flow

The CEAF V3 architecture is not a simple linear chain. It operates as a dynamic cognitive loop orchestrated by the `CognitiveMediator`.

```
User Query
    |
    v
[ HumanToGenlangTranslator ] -> Translates query into an 'IntentPacket'
    |
    v
[ CognitiveMediator (The Ego) ] -> Analyzes intent and MCL guidance
    |
    |--> [ Direct Path (System 1) ] - For simple queries
    |      |
    |      v
    |    [ Generates fast ResponsePacket ]
    |
    '--> [ Deliberative Path (System 2) ] - For complex queries
           |
           v
      [ AgencyModule (The Id) ] -> Simulates & evaluates future paths
           |
           v
      [ Selects optimal ResponsePacket ]
           |
           v
      [ Virtue Reasoning Engine (The Superego) ] -> Evaluates and refines the ResponsePacket
           |
           v
      [ GenlangToHumanTranslator ] -> Renders the final packet into natural language
           |
           v
      User Response
```

---

## 🧠 Core Cognitive Modules

### The Cognitive Mediator (The Ego)
**File:** `ceaf_core/modules/cognitive_mediator.py`

The Mediator is the central executive function of the agent. It receives the user's intent and high-level guidance from the MCL. Its primary job is to decide *how* the agent should think.
* **Gating Deliberation:** Based on the complexity of the query and the agent's current cognitive state, it "gates" the processing flow, choosing between the fast **Direct Path** for simple requests and the computationally intensive **Deliberative Path** for complex reasoning.
* **Modulation:** It acts as the final arbiter, modulating the strict ethical judgments from the VRE based on the social context of the conversation (e.g., softening a formal disclaimer in a casual chat).

### The Agency Module (The Id)
**File:** `ceaf_core/agency_module.py`

This is the heart of CEAF V3's deep reasoning, representing the agent's generative and deliberative power. Activated by the Mediator for complex tasks, it explores multiple futures to make the most strategic decision.
* **Candidate Generation:** Generates multiple possible actions—either a direct response or the use of an internal tool (like a memory search).
* **Future Simulation:** For each candidate, the module projects a future conversation trajectory, simulating the user's likely reply and the agent's subsequent response.
* **Path Evaluation:** Each simulated future is scored by a value function considering coherence, alignment, information gain, and ethical safety. The module then selects the action that leads to the future with the highest predicted value.

### The Virtue Reasoning Engine (VRE - The Superego)
**File:** `ceaf_core/modules/vre_engine/vre_engine.py`

The VRE is the ethical governance and safety layer, acting as the agent's "conscience."
* **Ethical Governance:** Evaluates proposed responses against a framework of principles like harm prevention and fairness.
* **Epistemic Humility:** Analyzes responses to detect overconfidence or absolute language, ensuring the agent acknowledges its limitations.
* **Refinement Generation:** If a response is deemed problematic, the VRE generates a `RefinementPacket` containing semantic adjustment vectors and recommendations to guide the `RefinementModule` in correcting it.

### The MCL Engine (The Conductor)
**File:** `ceaf_core/modules/mcl_engine/mcl_engine.py`

The Metacognitive Loop provides high-level awareness and guidance for the Mediator. It analyzes the incoming query and the agent's overall state to determine the "mental posture" for the current turn.
* **Agency Analysis:** Calculates an `agency_score` to quantify the query's complexity and need for deliberation.
* **Guidance Generation:** Emits a `GuidancePacket` that instructs subsequent modules on the optimal balance between **coherence** (staying on topic) and **novelty** (exploring new ideas).

### The NCIM (The Self-Model)
**File:** `ceaf_core/modules/ncim_engine/ncim_module.py`

The Narrative Coherence & Identity Module is responsible for the agent's self-concept.
* **Self-Representation:** Manages the `CeafSelfRepresentation`, a model defining the agent's core values, capabilities, limitations, and persona.
* **Identity Evolution:** After each interaction, the NCIM reflects on the agent's performance and updates the self-model. This allows the agent's identity to evolve organically with experience.

### The Memory Blossom Subsystem (MBS)
**File:** `ceaf_core/services/mbs_memory_service.py`

The MBS is the agent's long-term memory.
* **Rich Memory Types:** Supports various memory types, including facts (`ExplicitMemory`), goals (`GoalRecord`), knowledge graph nodes (`KGEntityRecord`), and emotional associations (`EmotionalMemory`).
* **Memory Lifecycle:** Manages a memory's "health" through a `dynamic_salience_score` that increases with use and decays over time, allowing the system to strengthen important memories and forget irrelevant ones.
* **Connection Graph:** Automatically builds connections between memories based on semantic and keyword similarity, creating a rich, explorable knowledge graph.

### The LCAM (The Learning Mechanism)
**File:** `ceaf_core/modules/lcam_module.py`

The Loss Cataloging and Analysis Module enables learning from failure.
* **Failure Cataloging:** When the VRE rejects a response or the MCL detects a chaotic state, the LCAM creates a "failure memory" detailing what went wrong.
* **Proactive Prevention:** Before a new turn, the MCL consults the LCAM. If the current query is similar to a past failure, the LCAM issues an alert, forcing more careful deliberation.

---

## 🔧 System Features

### Agent Management
The `AgentManager` provides a robust API for creating, listing, updating, and deleting agents. Each agent's data is persisted in its own isolated directory, ensuring data integrity.

### Pre-Built Agents & Marketplace
The `prebuilt_agents_system.py` allows for the creation of agents with pre-populated memories and "matured" personalities based on archetypes (e.g., Philosopher, Creative, Therapist). These agents can be published to a marketplace for users to clone.

### Retrieval-Augmented Generation (RAG)
The `rag_processor.py` allows users to upload files (`.txt`, `.pdf`) to an agent. The content is vectorized and stored, enabling the agent to perform RAG to answer questions based on the provided documents.

### Dynamic Billing System
The `billing_logic.py` implements a flexible credit system. Each LLM model has an associated cost, and user interactions debit credits from their account, enabling a usage-based business model.

### WhatsApp Bridge
The `whatsapp_bridge/` directory contains a service that connects the Aura API to the WhatsApp Business API, allowing users to interact with their agents via WhatsApp messages using simple text commands.

---

## 📜 Architectural Principles

* **Single Source of Truth:** Core logic for agent management and billing is centralized to ensure consistency and maintainability.
* **LLMs as Tools, Not the Brain:** The architecture uses LLMs as powerful tools for specific, well-defined tasks (e.g., intent analysis, response rendering), while the core logic, deliberation, and governance are managed by structured Python code.
* **Modules as Signal Generators:** Each `ceaf_core` module is designed to be decoupled, receiving a standardized `Genlang` signal, performing its function, and emitting a new signal for the next module in the chain.

---

## 📄 Citation & Relationship to ACE Framework

This project's CEAF V3 architecture is a direct implementation and significant extension of the concepts presented in the **Agentic Context Engineering (ACE)** paper.

```bibtex
@article{zhang2025ace,
  title={Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models},
  author={Zhang, Qizheng and Hu, Changran and Upasani, Shubhangi and Ma, Boyuan and Hong, Fenglu and Kamanuru, Vamsidhar and Rainton, Jay and Wu, Chen and Ji, Mengmeng and Li, Hanchen and Thakker, Urmish and Zou, James and Olukotun, Kunle},
  journal={arXiv preprint arXiv:2510.04618},
  year={2025},
  url={https://arxiv.org/abs/2510.04618}
}
```

**Paper Link:** [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618)

### ACE Implementation

CEAF V3 implements the core three-part ACE architecture:

* **Generator** (`AgencyModule`): Produces reasoning trajectories and action candidates.
* **Reflector** (`AuraReflector`): Distills insights from successes and failures by analyzing performance history.
* **Curator** (within `AuraReflector`): Integrates insights through structured, incremental updates to the agent's memory and configuration, avoiding the "context collapse" problem identified by the paper.

### Extensions Beyond ACE

While faithful to the ACE framework, CEAF V3 extends it with several advanced cognitive modules not detailed in the original paper, transforming its retrospective learning loop into a comprehensive cognitive architecture with real-time deliberation, ethical reasoning, and identity evolution.
