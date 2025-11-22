# 🚀 AuraCEAF V4: The Era of Dynamic Cognitive Configuration

This document describes the massive transformation carried out in the AuraCEAF architecture, evolving from a rigid logic system to a Configurable Intelligence Platform ("Data-Driven").

## 🎯 The Objective: Total "Twerkability"

Previously, the agent's behavior (whether it thinks fast or slow, whether it's creative or conservative, how it interprets the user) was "hardcoded" in scattered Python files.

V4 implements an Inversion of Control:

- Python logic now acts only as a generic engine.
- The agent's "Soul", "Personality", and "Biochemistry" are defined in a JSON file (cognitive_profile.json).
- This allows creating radically different agents without changing a single line of code, just by adjusting sliders and text ("Twerking").

## 🏗️ Main Architectural Changes

### 1. The Unified Cognitive Schema (ceaf_core/models.py)

We created a robust data structure (Pydantic) that defines all configurable aspects of the agent's mind:

- **LLMConfig**: Defines which models to use for fast thinking (Fast), reasoning (Smart), or creating (Creative).
- **MemoryConfig**: Controls how the agent remembers. (Ex: focus on exact keywords or semantic concepts? How fast does it forget?).
- **MCLConfig**: The brain's "thermostat". Defines when to pause for deep thinking (agency_threshold) and whether it prefers coherence or novelty.
- **DrivesConfig**: The motivational personality. (Ex: A curious agent that gets bored quickly vs. a focused, steady agent).
- **BodyConfig**: Virtual physiology. (Ex: Resistance to mental fatigue, capacity to absorb information before saturation).
- **SystemPrompts**: The biggest change. The prompt templates that control perception (HTG), deliberation (Agency), and response (GTH) are now user-editable.

### 2. Dependency Injection (ceaf_core/system.py)

The central orchestrator (CEAFSystem) was rewritten to:

- Load the cognitive_profile.json from disk on initialization.
- Inject specific configurations into each sub-module (MBS, MCL, Agency, Translators).
- Support Hot Reload: A reload_cognitive_profile method allows updating the agent's mind in real-time without restarting the server.

### 3. Configurable Sub-Modules

All logic engines were updated to stop using global constants and start obeying injected configuration:

- **LLMService**: No longer uses hardcoded LLM_MODEL_FAST. Uses self.config.fast_model.
- **MBSMemoryService**: Search weights (semantic vs. keyword) are now dynamic.
- **MCLEngine**: Coherence/novelty biases based on profile.
- **AgencyModule**: Strategy planning prompt is now a user template.
- **Translators (HTG/GTH)**: The complex logic (rules, user adaptation) was preserved, but is now calculated and passed as variables ({rules_block}, {user_adapt_block}) to a template the user controls.

### 4. Control Interface (api/routes.py & Frontend)

- **API**: New GET and PATCH endpoints `/agents/{id}/config` to read and write the agent's "soul".
- **Frontend**: New settings button (gear icon) that opens a Control Panel with tabs for LLM, Memory, Personality, and Prompts.

## 🧬 The New Life Cycle Flow of an Agent

**Creation**: When an agent is born, the AgentManager creates a cognitive_profile.json with optimized default values.

**Execution**: The CEAFSystem reads this JSON and configures the agent's brain.

**Interaction**:
- The agent senses "fatigue" based on BodyConfig.
- Decides whether to think fast or slow based on MCLConfig.
- Retrieves memories using weights from MemoryConfig.
- Generates the response by filling the gth_rendering template from SystemPrompts.

**Evolution (Twerking)**: The user adjusts a slider on the frontend → API receives the JSON → CEAFSystem does Hot Reload → The behavior changes instantly on the next message.

## 🧪 How to Test (A/B/C Strategy)

It's now possible to create clones of the same agent with distinct profiles to measure impact:

- **Low Config ("The Reflex")**: Fast models, low agency (doesn't think much), high fatigue (gets tired quickly), keyword focus. Expected result: Fast, superficial, conversational.
- **Medium Config ("The Balancer")**: The default balance.
- **High Config ("The Deep Thinker")**: Smart models (Claude/GPT-4), sensitive agency (thinks about everything), deep semantic memory, high curiosity. Expected result: Slow, deep, creative, analytical.